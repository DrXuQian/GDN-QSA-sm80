#!/usr/bin/env python3
"""Reproduce a reference admission failure without timing or altering its code."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "tools"), str(ROOT / "benchmarks")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failed", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir()
    os.environ.update(TILELANG_CACHE_DIR=str(a.work / "cache/tilelang"),
                      TILELANG_TMP_DIR=str(a.work / "scratch/tilelang"))
    import torch
    from test_ppu_gdn_backend import fixture, digest, error, MAX_RELATIVE_ERROR
    from gdn_qsa_sm80.reference.gdn_chunk_ref import torch_recurrent_gated_delta_rule as oracle
    from gdn_qsa_sm80.gdn_sm90_interface import gdn_chunk_sm90
    from sm90_library_inputs import make_inputs, expand_reference_heads
    from sm90_workloads import BY_NAME
    from profile_sm90_libraries import validate_archive, reference_binaries, build_receipt
    from bench_sm90_hopper import DeviceWatch
    failed = json.loads((a.failed / "receipt.json").read_text())
    if failed["comparison_family"] != "flashqla-candidate" or failed["calls"]:
        raise ValueError("this diagnostic requires an untimed FlashQLA admission failure")
    if "original 2% criterion failed" not in failed["error"]:
        raise ValueError("different failure mechanism")
    result = dict(scope="REFERENCE_NUMERICS_DIAGNOSTIC_NO_TIMING", status="INCOMPLETE",
                  failed_receipt_sha256=sha(a.failed / "receipt.json"),
                  failed_log_sha256=sha(a.failed / "run.log"), harness_sha256=sha(__file__),
                  source=validate_archive(a.work / "FlashQLA", a.work / "flashqla-source.tar.gz"),
                  workload=failed["workload"], gate=failed["gate"], criterion=MAX_RELATIVE_ERROR, roles={})
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    watch = DeviceWatch(0)
    try:
        for _ in range(3):
            watch.sample(idle=True); time.sleep(.2)
        watch.thread.start()
        workload = BY_NAME[failed["workload"]["name"]]
        cpu, state_cpu = make_inputs(workload, failed["gate"], fixture)
        q, k = expand_reference_heads(cpu, workload)
        want = oracle(q, k, *cpu[2:], initial_state=state_cpu, output_final_state=True)
        if digest(cpu) != failed["input_sha256"] or digest(want) != failed["reference_sha256"]:
            raise ValueError("failed fixture/oracle not reproduced exactly")
        result.update(input_sha256=digest(cpu), reference_sha256=digest(want))
        inputs = tuple(t.cuda() for t in cpu)
        initial = state_cpu.cuda() if state_cpu is not None else None
        sys.path.insert(0, str(a.work / "FlashQLA"))
        from flash_qla import chunk_gated_delta_rule as ref
        calls = {}
        for role in ("ours-cuda", "ours-candidate"):
            path = Path(failed["incumbent_builds"][role]["extension"])
            manifest = build_receipt(path, "cuda_sm90", "native")
            if manifest["extension_sha256"] != failed["incumbent_builds"][role]["extension_sha256"]:
                raise ValueError("frozen binary changed")
            def ours(path=path):
                os.environ["GDN_QSA_SM90_EXTENSION"] = str(path)
                return gdn_chunk_sm90(*inputs, initial_state=initial, backend="cuda_sm90")
            calls[role] = ours
        for name, cp in (("auto", True), ("no-cp", False)):
            calls[f"flashqla-{name}"] = lambda cp=cp: ref(*inputs, scale=128**-.5,
                initial_state=initial, output_final_state=True, use_qk_l2norm_in_kernel=False,
                state_v_first=False, auto_cp=cp, enable_fwd_cp_cache=False)
        for role, call in calls.items():
            pair = tuple(t.cpu() for t in call())
            if any(x.shape != y.shape for x, y in zip(pair, want)):
                raise ValueError("wrong output ABI")
            fingerprint = digest(pair)
            for _ in range(8):
                if digest(tuple(t.cpu() for t in call())) != fingerprint:
                    raise ValueError("diagnostic replay unstable")
            errs = [error(x, y) for x, y in zip(pair, want)]
            details = []
            for x, y in zip(pair, want):
                delta = (x.float()-y.float()).abs()
                index = int(delta.reshape(-1).argmax())
                details.append(dict(flat=index, got=float(x.reshape(-1)[index]), want=float(y.reshape(-1)[index]),
                    absolute_error=float(delta.reshape(-1)[index]), reference_absmax=float(y.abs().max()),
                    nonfinite=int((~torch.isfinite(x)).sum())))
            row = dict(errors=errs, verdict="PASS" if max(errs) < MAX_RELATIVE_ERROR else "NUMERICAL_FAIL",
                       fingerprint=fingerprint, repeat="8/8 RAW-BIT", worst=details)
            result["roles"][role] = row
            torch.save(pair, a.out / f"{role}.pt")
            print(f"[reference numeric] {role} {row}", flush=True)
        torch.save(dict(inputs=cpu, initial=state_cpu, reference=want), a.out / "fixture.pt")
        result["tensor_artifacts"] = {path.name: sha(path) for path in a.out.glob("*.pt")}
        result["images"] = reference_binaries("flashqla", a.out)
        if result["roles"]["ours-cuda"]["fingerprint"] != result["roles"]["ours-candidate"]["fingerprint"]:
            raise ValueError("candidate differs from parent")
        watch.sample()
        result["status"] = "DIAGNOSIS_COMPLETE_NOT_PERFORMANCE"
    except Exception as e:
        result.update(status="FAIL", error=repr(e)); raise
    finally:
        watch.stop.set()
        if watch.thread.ident is not None: watch.thread.join(timeout=22)
        if watch.errors or watch.thread.is_alive(): result.update(status="FAIL", error="device interference")
        result.update(utc=datetime.now(timezone.utc).isoformat(), device_watch=dict(records=watch.records, errors=watch.errors))
        (a.out / "diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] == "FAIL": raise RuntimeError(result["error"])


if __name__ == "__main__":
    main()
