#!/usr/bin/env python3
"""Run the registered finite H800 workload inventory, one GPU job at a time.

No kernel changes, performance-based row removal, or implicit successes.
Failures remain in the denominator. Resume only reuses hash-bound complete
cells with the current workload authority, candidate and control images.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from sm90_workloads import WORKLOADS, GATES, FAMILIES, validate_inventory

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--ppu-source", type=Path, required=True)
    p.add_argument("--only", nargs="+", choices=[w.name for w in WORKLOADS],
                   help="bounded tranche; remaining registered rows remain PENDING")
    p.add_argument("--family", choices=FAMILIES)
    p.add_argument("--gate", type=float, choices=GATES)
    p.add_argument("--cell-timeout", type=int, default=1800)
    args = p.parse_args()
    validate_inventory(WORKLOADS)
    args.out.mkdir(parents=True, exist_ok=True)
    bindings = {"control": sha(args.control), "candidate": sha(args.candidate),
                "ppu-source": sha(args.ppu_source),
                "authority": sha(ROOT / "tools/sm90_workloads.py"),
                "adapters": sha(ROOT / "benchmarks/sm90_library_inputs.py"),
                "harness": sha(ROOT / "benchmarks/profile_sm90_libraries.py")}
    state_file = args.out / "matrix.json"
    # Reorder only: the anchor and genuine batch boundary must run first.
    order = sorted(WORKLOADS, key=lambda w: (w.name not in ("seq2048", "batch2"),
                                            w.name != "seq2048", WORKLOADS.index(w)))
    cells = [dict(workload=w.name, gate=g, family=f, status="PENDING",
                  directory=f"{w.name}-{f}-g{g}")
             for w in order for g in GATES for f in FAMILIES]
    if state_file.exists():
        old = json.loads(state_file.read_text())
        if old["bindings"] != bindings:
            raise RuntimeError("resume input/code/binary identity changed")
        index = {row["directory"]: row for row in old["cells"]}
        if set(index) != {row["directory"] for row in cells}:
            raise RuntimeError("resume denominator changed")
        cells = [index[row["directory"]] for row in cells]

    def save():
        counts = {status: sum(row["status"] == status for row in cells)
                  for status in ("PENDING", "RUNNING", "PASS", "FAIL")}
        data = dict(utc=utc(), scope="H800_NOT_NATIVE_PPU17", registered_cells=56,
                    registered_numerical_scenarios=28, bindings=bindings,
                    status_counts=counts, cells=cells, routing="UNCHANGED",
                    shutdown_authorized=False)
        temp = state_file.with_suffix(".json.new")
        temp.write_text(json.dumps(data, indent=2) + "\n")
        temp.replace(state_file)
        print(f"[matrix] {utc()} {counts} denominator=56", flush=True)

    save()
    for row in cells:
        if args.only and row["workload"] not in args.only:
            continue
        if args.family and row["family"] != args.family:
            continue
        if args.gate is not None and row["gate"] != args.gate:
            continue
        directory = args.out / row["directory"]
        if row["status"] == "PASS":
            for name, digest in row["evidence"].items():
                if sha(directory / name) != digest:
                    raise RuntimeError("resume evidence changed")
            continue
        if row["status"] != "PENDING" or directory.exists():
            raise RuntimeError(f"unclosed cell requires explicit diagnosis, not overwrite: {directory}")
        env = dict(os.environ, WORK=str(args.work), FAMILY=row["family"],
                   GATE=str(row["gate"]), WORKLOAD=row["workload"], OUT=str(directory),
                   CUDA_EXTENSION=str(args.control), PPU_SOURCE_EXTENSION=str(args.ppu_source),
                   CANDIDATE_EXTENSION=str(args.candidate), CANDIDATE_RAW_BIT="1")
        row.update(status="RUNNING", started_utc=utc())
        save()
        start = time.monotonic()
        log = args.out / f"{row['directory']}.driver.log"
        try:
            with log.open("x") as stream:
                completed = subprocess.run(["bash", str(ROOT / "tools/run_sm90_library_nsys.sh")],
                    env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=args.cell_timeout, check=False)
            if completed.returncode:
                raise RuntimeError(f"capture returned {completed.returncode}; {log}")
            receipt = json.loads((directory / "receipt.json").read_text())
            result = json.loads((directory / "result.json").read_text())
            if receipt["workload"]["name"] != row["workload"] or receipt["gate"] != row["gate"]:
                raise RuntimeError("capture bound to wrong workload")
            if receipt["incumbent_builds"]["ours-candidate"]["extension_sha256"] != bindings["candidate"]:
                raise RuntimeError("capture bound to wrong candidate")
            if receipt["device_watch"]["errors"] or not receipt["candidate_requires_raw_bit"]:
                raise RuntimeError("concurrency or raw-bit admission incomplete")
            row.update(status="PASS", candidate_beats_all=result["candidate_beats_all_reference_paths"],
                       vs_parent=result["comparisons"]["ours-candidate"]["verdict"],
                       references=result["candidate_vs_references"],
                       summary=result["summary"],
                       evidence={name: sha(directory / name) for name in
                                 ("receipt.json", "result.json", "forward.sqlite", "forward.nsys-rep")})
        except Exception as error:
            row.update(status="FAIL", error=repr(error))
            raise
        finally:
            row.update(elapsed_seconds=time.monotonic() - start, finished_utc=utc())
            save()


if __name__ == "__main__":
    main()
