#!/usr/bin/env python3
"""Numerical stress outside the shipping [-1,0] admission; NEVER performance.

Keep run_sm90_gdn.py's admitted workload range unchanged. This separate
diagnostic exercises discarded upper-triangle Inf/NaN at g=-8/-10000.
One public invocation per process, independent CPU recurrence, fixed2% gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import torch
from test_ppu_gdn_backend import fixture, assert_pair, digest, MAX_RELATIVE_ERROR
from gdn_qsa_sm80.reference.gdn_chunk_ref import torch_recurrent_gated_delta_rule
from gdn_qsa_sm80.gdn_sm90_interface import gdn_chunk_sm90


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extension", type=Path, required=True)
    p.add_argument("--gate", type=float, choices=(-8., -10000.), required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(1)
    manifest = json.loads((a.extension.parent / "build.json").read_text())
    binary = hashlib.sha256(a.extension.read_bytes()).hexdigest()
    assert manifest["complete"] and manifest["extension_sha256"] == binary
    assert manifest["target"] == "cuda_sm90" and manifest["mode"] == "native"
    assert not a.out.exists(), "preserve old evidence"
    a.out.mkdir(parents=True)
    q,k,v,g,beta = fixture(1,129,1,2,a.gate)
    g = g.float()
    state = torch.randn(1,2,128,128,device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(17))*.005
    want = torch_recurrent_gated_delta_rule(q.repeat_interleave(2,2),k.repeat_interleave(2,2),
        v,g,beta,initial_state=state,output_final_state=True)
    os.environ["GDN_QSA_SM90_EXTENSION"] = str(a.extension.resolve())
    cpu = (q,k,v,g,beta)
    device = tuple(t.to("cuda:0") for t in cpu)
    got = gdn_chunk_sm90(*device,initial_state=state.to("cuda:0"),output_final_state=True,
                         backend="cuda_sm90",source_check=False)
    actual = tuple(t.detach().cpu() for t in got)
    errors = assert_pair(actual,want)
    assert all(torch.isfinite(t).all().item() for t in actual)
    torch.save(actual,a.out / "captured.pt")
    result = dict(scope="AUX-MASK-STRESS-NOT-SHIPPING-ADMISSION-OR-PERFORMANCE",
        gate=a.gate,errors=errors,limit=MAX_RELATIVE_ERROR,extension_sha256=binary,
        input_sha256=digest(cpu),output_sha256=digest(actual),public_calls=1,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out / "result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
