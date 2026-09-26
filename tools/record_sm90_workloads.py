#!/usr/bin/env python3
"""Re-extract all multi-workload traces; never infer success from a subset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from analyze_sm90_nsys import extract
from sm90_workloads import WORKLOADS, GATES, FAMILIES, BY_NAME, validate_inventory

ROOT = Path(__file__).resolve().parents[1]
UUID = "GPU-1d5fdef3-4899-79d9-19e6-c9c815b2a59c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete_win(rows):
    expected = {(w.name, g, f) for w in WORKLOADS for g in GATES for f in FAMILIES}
    return (len(rows) == 56 and {(r["workload"], r["gate"], r["family"]) for r in rows} == expected
            and all(r["status"] == "PASS" and r.get("candidate_beats_all") for r in rows))


def record(root):
    validate_inventory(WORKLOADS)
    state = json.loads((root / "matrix.json").read_text())
    rows = state["cells"]
    expected = {(w.name, g, f) for w in WORKLOADS for g in GATES for f in FAMILIES}
    if len(rows) != 56 or {(r["workload"], r["gate"], r["family"]) for r in rows} != expected:
        raise ValueError("matrix denominator is not the registered 56 captures")
    bindings = state["bindings"]
    for role, file in (("authority", "tools/sm90_workloads.py"),
                       ("adapters", "benchmarks/sm90_library_inputs.py"),
                       ("harness", "benchmarks/profile_sm90_libraries.py")):
        if sha(ROOT / file) != bindings[role]:
            raise ValueError(f"changed measurement authority: {role}")
    scenarios = {}
    complete = []
    forwards = 0
    for row in rows:
        if row["status"] != "PASS":
            continue
        directory = root / row["directory"]
        for name, digest in row["evidence"].items():
            if sha(directory / name) != digest:
                raise ValueError("evidence modified")
        receipt = json.loads((directory / "receipt.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        workload = BY_NAME[row["workload"]]
        if receipt["workload"] != json.loads(json.dumps(workload.receipt())):
            raise ValueError("executed workload differs from registered input")
        if receipt["shape"] != [*workload.shape, 128] or receipt["gate"] != row["gate"]:
            raise ValueError("shape/gate changed")
        if receipt["workload_authority_sha256"] != bindings["authority"]:
            raise ValueError("executed authority differs")
        if receipt["input_adapter_sha256"] != bindings["adapters"]:
            raise ValueError("executed ABI adapters differ")
        for role, key in (("ours-cuda", "control"), ("ours-candidate", "candidate"),
                          ("ours-ppu-source-check", "ppu-source")):
            if receipt["incumbent_builds"][role]["extension_sha256"] != bindings[key]:
                raise ValueError("binary changed between cells")
        watch = receipt["device_watch"]
        if watch["errors"] or not watch["records"]:
            raise ValueError("missing/invalid exclusive device evidence")
        if any(r["telemetry"].split(",")[0] != UUID for r in watch["records"]):
            raise ValueError("physical device changed")
        if not receipt["candidate_requires_raw_bit"]:
            raise ValueError("raw-bit admission omitted")
        admission = receipt["admission"]
        if any(a["repeat"] != "8/8 RAW-BIT" for a in admission.values()):
            raise ValueError("replay admission omitted")
        if admission["ours-cuda"]["fingerprint"] != admission["ours-candidate"]["fingerprint"]:
            raise ValueError("parent/candidate raw fingerprint differs")
        with sqlite3.connect(f"file:{directory}/forward.sqlite?mode=ro", uri=True) as db:
            fresh = json.loads(json.dumps(extract(db, receipt)))
        if fresh != {k: v for k, v in result.items() if k != "evidence_sha256"}:
            raise ValueError("stored timing verdict differs from exact SQLite re-extraction")
        for path, digest in result["evidence_sha256"].items():
            local = ROOT / "tools/analyze_sm90_nsys.py" if Path(path).name == "analyze_sm90_nsys.py" else directory / Path(path).name
            if sha(local) != digest:
                raise ValueError("analyzer/evidence hash differs")
        identity = (receipt["input_sha256"], receipt["initial_sha256"], receipt["reference_sha256"])
        key = (row["workload"], row["gate"])
        if scenarios.setdefault(key, identity) != identity:
            raise ValueError("cross-library comparison did not use the same inputs/oracle")
        summary = fresh["summary"]
        reference_roles = [role for role in summary if not role.startswith("ours-")]
        fastest = min(reference_roles, key=lambda role: summary[role]["kernel_sum_us"]["median"])
        complete.append(dict(workload=row["workload"], gate=row["gate"], family=row["family"],
            candidate=summary["ours-candidate"]["kernel_sum_us"],
            control=summary["ours-cuda"]["kernel_sum_us"], fastest_reference=fastest,
            reference=summary[fastest]["kernel_sum_us"],
            verdict=fresh["candidate_vs_references"][fastest]["verdict"],
            versus_parent=fresh["comparisons"]["ours-candidate"]["verdict"],
            all_reference_verdicts=fresh["candidate_vs_references"],
            candidate_beats_all=fresh["candidate_beats_all_reference_paths"],
            route=receipt["route"], input_identity=identity, evidence=row["evidence"]))
        if row["candidate_beats_all"] != fresh["candidate_beats_all_reference_paths"]:
            raise ValueError("matrix verdict drift")
        forwards += len(fresh["forwards"])
    return dict(scope="H800_NOT_NATIVE_PPU17", bindings=bindings, device_uuid=UUID,
                registered_workloads=14, numerical_scenarios=28, registered_captures=56,
                counts=dict(Counter(r["status"] for r in rows)), complete_forwards=forwards,
                reanalysis="EXACT_SQLITE_ALL_COMPLETED_CAPTURES", rows=complete,
                unfinished=[r for r in rows if r["status"] != "PASS"],
                candidate_all_workloads_win=complete_win(rows),
                native_ppu17="SKIP_SDK_MODEL_UNAVAILABLE", routing="UNCHANGED")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = record(args.root)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k not in ("rows", "unfinished", "bindings")})
    for row in result["rows"]:
        print(f"{row['workload']:16s} g={row['gate']:4} {row['family']:10s} "
              f"S38={row['candidate']['median']:9.3f} S24={row['control']['median']:9.3f} "
              f"{row['fastest_reference']}={row['reference']['median']:9.3f} {row['verdict']}")


if __name__ == "__main__":
    main()
