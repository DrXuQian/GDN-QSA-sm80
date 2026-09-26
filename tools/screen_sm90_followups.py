#!/usr/bin/env python3
"""Bounded, paired graph screens after full matrix and numerical admission.

Screening is not performance admission: any selected finalist still requires
same-input nsys complete-forward comparison against both pinned references.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "benchmarks"), str(ROOT / "tools")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--admission", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    admission = json.loads(a.admission.read_text())
    if admission["denominator"] != 3 or any(row["status"] != "PASS" for row in admission["rows"]):
        raise ValueError("all three registered candidates must first pass numerical admission")
    a.out.mkdir()
    import torch
    from bench_sm90_hopper import DeviceWatch
    from test_ppu_gdn_backend import fixture, digest, assert_pair
    from gdn_qsa_sm80.gdn_sm90_interface import gdn_chunk_sm90
    from gdn_qsa_sm80.reference.gdn_chunk_ref import torch_recurrent_gated_delta_rule
    from sm90_library_inputs import make_inputs, expand_reference_heads
    from sm90_workloads import BY_NAME, GATES
    from profile_sm90_libraries import build_receipt
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    watch = DeviceWatch(0)
    rows = []
    result = dict(scope="H800_GRAPH_SCREEN_NOT_NSYS_ADMISSION", denominator=12,
                  admission_sha256=sha(a.admission), harness_sha256=sha(__file__),
                  rows=rows, status="INCOMPLETE", routing="UNCHANGED")
    try:
        for _ in range(3):
            watch.sample(idle=True)
            time.sleep(.2)
        watch.thread.start()
        for candidate in admission["rows"]:
            control = (Path("/workspace/gdn-sm90-value-split-20260926/s38-build") if candidate["id"] == "s41"
                       else Path("/workspace/gdn-sm90-win-20260926/relative-build"))
            paths = {"parent": next(control.glob("_gdn_fused_sm90*.so")),
                     "candidate": next(Path(candidate["build"]).glob("_gdn_fused_sm90*.so"))}
            identities = {role: build_receipt(path, "cuda_sm90", "native") for role, path in paths.items()}
            if identities["candidate"]["extension_sha256"] != candidate["binary_sha256"]:
                raise ValueError("candidate changed after numeric admission")
            workloads = ("seq2048", "seq8192") if candidate["id"] == "s41" else ("seq2048", "batch2")
            for name in workloads:
                workload = BY_NAME[name]
                for gate in GATES:
                    cpu, state_cpu = make_inputs(workload, gate, fixture)
                    assert state_cpu is None
                    q, k = expand_reference_heads(cpu, workload)
                    want = torch_recurrent_gated_delta_rule(q, k, *cpu[2:], output_final_state=True)
                    inputs = tuple(t.cuda() for t in cpu)
                    def call(role):
                        os.environ["GDN_QSA_SM90_EXTENSION"] = str(paths[role])
                        return gdn_chunk_sm90(*inputs, backend="cuda_sm90")
                    fingerprints, errors, graphs, captured = {}, {}, {}, {}
                    for role in paths:
                        first = None
                        for _ in range(8):
                            pair = tuple(t.cpu() for t in call(role))
                            errors[role] = assert_pair(pair, want)
                            fingerprint = digest(pair)
                            if first is not None and first != fingerprint:
                                raise ValueError("unstable direct launch")
                            first = fingerprint
                        fingerprints[role] = first
                    if fingerprints["parent"] != fingerprints["candidate"]:
                        raise ValueError("delivery candidate differs from parent raw bits")
                    for role in paths:
                        graphs[role] = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graphs[role]):
                            captured[role] = [call(role) for _ in range(16)]
                        for _ in range(3):
                            graphs[role].replay()
                    torch.cuda.synchronize()
                    samples = {role: [] for role in paths}
                    for sample in range(9):
                        for role in (("parent", "candidate") if sample % 2 == 0 else ("candidate", "parent")):
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record(); graphs[role].replay(); end.record(); end.synchronize()
                            samples[role].append(start.elapsed_time(end) * 1000 / 16)
                    for role in paths:
                        for pair in captured[role]:
                            actual = tuple(t.cpu() for t in pair)
                            assert_pair(actual, want)
                            if digest(actual) != fingerprints[role]:
                                raise ValueError("captured output differs from admitted direct result")
                    if digest(tuple(t.cpu() for t in inputs)) != digest(cpu):
                        raise ValueError("input mutated")
                    summary = {role: dict(median=statistics.median(times), range=[min(times), max(times)], samples=times)
                               for role, times in samples.items()}
                    verdict = ("CANDIDATE-WINS" if max(samples["candidate"]) < min(samples["parent"]) else
                               "PARENT-WINS" if max(samples["parent"]) < min(samples["candidate"]) else "UNRESOLVED")
                    row = dict(candidate=candidate["id"], workload=workload.receipt(), gate=gate,
                               binaries=identities, input_sha256=digest(cpu), errors=errors,
                               fingerprint=fingerprints["candidate"], replay="8_DIRECT+32_GRAPH_RESULTS",
                               summary_us=summary, verdict=verdict, admission="NOT_A_SPEED_VERDICT")
                    rows.append(row)
                    watch.sample()
                    if watch.errors:
                        raise RuntimeError(f"device interference: {watch.errors}")
                    (a.out / "screen.json").write_text(json.dumps(result, indent=2) + "\n")
                    print(f"[screen] {candidate['id']} {name} g={gate} parent={summary['parent']['median']:.3f} "
                          f"candidate={summary['candidate']['median']:.3f} {verdict} NSYS_REQUIRED", flush=True)
        if len(rows) != 12:
            raise ValueError("screen denominator changed")
        result["status"] = "PASS_SCREEN_COMPLETED_NOT_SPEED_ADMISSION"
    except Exception as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        watch.stop.set()
        if watch.thread.ident is not None:
            watch.thread.join(timeout=22)
        result.update(updated_at=datetime.now(timezone.utc).isoformat(), device_watch=dict(records=watch.records, errors=watch.errors))
        (a.out / "screen.json").write_text(json.dumps(result, indent=2) + "\n")
    if watch.thread.is_alive() or watch.errors:
        raise RuntimeError("incomplete exclusive device evidence")


if __name__ == "__main__":
    main()
