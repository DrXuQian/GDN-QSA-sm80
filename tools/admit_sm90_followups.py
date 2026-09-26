#!/usr/bin/env python3
"""Admit the three registered followups, sequentially, after the frozen matrix.

No timing or routing decision is made here. A numerical failure stops that
admission sequence for diagnosis; it cannot be relabelled as an environment skip.
An independently reproduced reference-only failure stays FAIL in the matrix
but need not block correctness checks of our unrelated candidate kernels.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "benchmarks")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admit_reference_diagnosis(diagnosis, criterion):
    if diagnosis["status"] != "DIAGNOSIS_COMPLETE_NOT_PERFORMANCE":
        raise ValueError("reference failure not diagnosed")
    if diagnosis["criterion"] != criterion or diagnosis["device_watch"]["errors"]:
        raise ValueError("numeric criterion or device evidence changed")
    roles = diagnosis["roles"]
    if set(roles) != {"ours-cuda", "ours-candidate", "flashqla-auto", "flashqla-no-cp"}:
        raise ValueError("diagnostic role denominator changed")
    for name in ("ours-cuda", "ours-candidate"):
        if (roles[name]["verdict"] != "PASS" or not all(math.isfinite(e) and 0 <= e < criterion for e in roles[name]["errors"])
                or roles[name]["repeat"] != "8/8 RAW-BIT"):
            raise ValueError("our numerical failure cannot be classified as reference-only")
    if roles["ours-cuda"]["fingerprint"] != roles["ours-candidate"]["fingerprint"]:
        raise ValueError("our raw-bit failure cannot be classified as reference-only")
    if not any(max(roles[name]["errors"]) >= criterion for name in ("flashqla-auto", "flashqla-no-cp")):
        raise ValueError("reference failure not reproduced")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--value-parent", type=Path, required=True)
    p.add_argument("--stash-root", type=Path, required=True)
    p.add_argument("--loader-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference-diagnosis", type=Path, nargs="*", default=[],
                   help="explicit reproduced reference-only failures; never change the matrix verdict")
    a = p.parse_args()
    matrix = json.loads(a.matrix.read_text())
    from sm90_workloads import WORKLOADS, GATES, FAMILIES
    expected_cells = {(w.name, g, f) for w in WORKLOADS for g in GATES for f in FAMILIES}
    if (len(matrix["cells"]) != 56
            or {(r["workload"], r["gate"], r["family"]) for r in matrix["cells"]} != expected_cells
            or any(c["status"] not in ("PASS", "FAIL") for c in matrix["cells"])):
        raise RuntimeError("finish all frozen56 attempts before running another GPU task")
    if a.out.exists():
        raise RuntimeError("preserve previous admission artifacts; no implicit rerun/overwrite")
    a.out.mkdir(parents=True)
    import torch
    from bench_sm90_hopper import DeviceWatch
    from test_ppu_gdn_backend import MAX_RELATIVE_ERROR
    failures = [row for row in matrix["cells"] if row["status"] == "FAIL"]
    diagnoses = {}
    for path in a.reference_diagnosis:
        data = json.loads(path.read_text())
        admit_reference_diagnosis(data, MAX_RELATIVE_ERROR)
        for name, digest in data["tensor_artifacts"].items():
            if sha(path.parent / name) != digest:
                raise ValueError("diagnostic tensors changed")
        diagnoses[data["failed_receipt_sha256"]] = dict(path=str(path), sha256=sha(path), data=data)
    if len(diagnoses) != len(failures):
        raise ValueError("every failed matrix cell needs its own explicit reference diagnosis")
    for row in failures:
        receipt_path = a.matrix.parent / row["directory"] / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        entry = diagnoses.get(sha(receipt_path))
        if (entry is None or receipt["calls"] or receipt["device_watch"]["errors"]
                or receipt["comparison_family"] != "flashqla-candidate"
                or "original 2% criterion failed" not in receipt.get("error", "")):
            raise ValueError("matrix failure is not the diagnosed untimed reference failure")
        if (entry["data"]["input_sha256"] != receipt["input_sha256"]
                or entry["data"]["reference_sha256"] != receipt["reference_sha256"]):
            raise ValueError("diagnosis used different inputs/reference")
    torch.set_num_threads(1)
    DeviceWatch(0).sample(idle=True)
    expected = json.loads((a.parent / "relative-cases/cases.json").read_text())
    assert expected["denominator"] == expected["completed"] == len(expected["cases"]) == 14
    rows = [dict(id=name, root=str(root), build=str(root / f"{name}-build"), status="PENDING")
            for name, root in (("s39", a.stash_root), ("s40", a.stash_root), ("s41", a.loader_root))]

    def save():
        result = dict(updated_at=datetime.now(timezone.utc).isoformat(), denominator=3, rows=rows,
                      matrix_sha256=sha(a.matrix), harness_sha256=sha(__file__),
                      matrix_failures_preserved=len(failures),
                      reference_diagnoses={key: {k: v for k, v in entry.items() if k != "data"}
                                           for key, entry in diagnoses.items()},
                      performance="NOT_RUN", native_ppu17="SKIP_SDK_MODEL_UNAVAILABLE",
                      routing="UNCHANGED", shutdown_authorized=False)
        (a.out / "admission.json").write_text(json.dumps(result, indent=2) + "\n")

    def run(directory, label, args):
        with (directory / f"{label}.log").open("x") as output:
            completed = subprocess.run([sys.executable, *map(str, args)], stdout=output,
                                       stderr=subprocess.STDOUT, check=False, timeout=300)
        if completed.returncode:
            raise RuntimeError(f"{label} rc={completed.returncode}: {directory}")

    save()
    for row in rows:
        directory = a.out / row["id"]
        directory.mkdir()
        root = Path(row["root"])
        build = Path(row["build"])
        identity = json.loads((build / "build.json").read_text())
        binaries = list(build.glob("_gdn_fused_sm90*.so"))
        assert len(binaries) == 1 and identity["complete"]
        binary = binaries[0]
        assert sha(binary) == identity["extension_sha256"]
        assert "C7512" not in (build / "device.log").read_text()
        for relative, digest in identity["source_sha256"].items():
            assert sha(root / "source" / relative) == digest
        row.update(status="RUNNING", binary_sha256=sha(binary),
                   source=identity["repository_revision"], flags=identity["flags"])
        save()
        try:
            if row["id"] in ("s39", "s40"):
                mode = 1 if row["id"] == "s39" else 2
                assert f"-DGDN_SM90_STATE_OUTPUT_STASH={mode}" in identity["flags"]
                run(directory, "native", [root / "source/dev/backends/check_sm90_output_stash.py",
                    build / "codegen/image.sass", a.parent / "relative-build/codegen/image.sass",
                    "--mode", mode, "--device-log", build / "device.log", "--out", directory / "native.json"])
            else:
                assert "-DGDN_SM90_VALUE_LOADER_REGS=32" in identity["flags"]
                run(directory, "native", [root / "source/dev/backends/check_sm90_value_native.py",
                    build / "codegen/image.sass", a.value_parent / "s38-build/codegen/image.sass",
                    "--aux", 232, "--loader", 32])
            run(directory, "cases", [ROOT / "tests/run_sm90_hopper_cases.py", "--extension", binary,
                "--backend", "cuda_sm90", "--out", directory / "cases"])
            actual = json.loads((directory / "cases/cases.json").read_text())
            assert actual["denominator"] == actual["completed"] == len(actual["cases"]) == 14
            for got, want in zip(actual["cases"], expected["cases"]):
                assert got["case"] == want["case"] and got["rc"] == want["rc"] == 0
                for key in ("input_sha256", "output_sha256", "errors"):
                    assert got["result"][key] == want["result"][key], (got["case"], key)
                assert max(got["result"]["errors"]) < MAX_RELATIVE_ERROR
            for magnitude in (8, 10000):
                destination = directory / f"stress{magnitude}"
                run(directory, f"stress{magnitude}", [ROOT / "tests/run_sm90_aux_mask_stress.py",
                    "--extension", binary, "--gate", -magnitude, "--out", destination])
                x = torch.load(destination / "captured.pt", weights_only=True, map_location="cpu")
                y = torch.load(a.parent / f"relative-stress{magnitude}/captured.pt", weights_only=True, map_location="cpu")
                assert len(x) == len(y) == 2
                assert all(torch.equal(l.contiguous().view(torch.uint8), r.contiguous().view(torch.uint8))
                           for l, r in zip(x, y))
            row.update(status="PASS", numerical="14CPU_PARENT_FINGERPRINTS+2DIRECT_BYTE_STRESSES",
                       evidence=str(directory), performance="NOT_RUN")
        except Exception as error:
            row.update(status="FAIL", error=repr(error))
            raise
        finally:
            save()
        print(f"[followup admission] {row['id']} {row['status']} performance=NOT_RUN", flush=True)


if __name__ == "__main__":
    main()
