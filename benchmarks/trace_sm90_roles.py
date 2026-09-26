#!/usr/bin/env python3
"""H800 diagnostic calls with isolated CPU trace readback, not a speed verdict."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"tests"),str(ROOT/"benchmarks"),str(ROOT/"tools")]
import torch
from bench_sm90_hopper import DeviceWatch
from profile_sm90_cula import build_receipt, sha
from test_ppu_gdn_backend import fixture,assert_pair,digest
from gdn_qsa_sm80.reference.gdn_chunk_ref import torch_recurrent_gated_delta_rule
from gdn_qsa_sm80.backends.loading import _load
from analyze_sm90_roles import analyze


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension",type=Path,required=True)
    parser.add_argument("--control",type=Path,required=True)
    parser.add_argument("--gate",type=float,choices=(-.1,-1.),required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    receipt=dict(scope="DIAGNOSTIC_ONLY_NOT_PERFORMANCE",status="INCOMPLETE",gate=args.gate,
        builds={name:build_receipt(path,"cuda_sm90","native") for name,path in (("trace",args.extension),("control",args.control))},
        harness_sha256=sha(__file__))
    watch=DeviceWatch(0)
    try:
        for _ in range(3): watch.sample(idle=True); time.sleep(.2)
        watch.thread.start()
        props=torch.cuda.get_device_properties(0)
        assert props.major==9 and props.minor==0 and "PPU" not in props.name
        receipt.update(device=props.name,sms=props.multi_processor_count)
        torch.set_num_threads(1)
        cpu=fixture(1,2048,16,32,args.gate)
        q,k,v,g,beta=cpu
        want=torch_recurrent_gated_delta_rule(q.repeat_interleave(2,2),k.repeat_interleave(2,2),v,g,beta,output_final_state=True)
        inputs=tuple(t.cuda() for t in cpu)
        trace=_load("_gdn_fused_sm90",str(args.extension.resolve()))
        control=_load("_gdn_fused_sm90",str(args.control.resolve()))
        assert trace.role_trace_enabled and not hasattr(control,"role_trace_enabled")
        anchor=tuple(t.cpu() for t in control.forward(*inputs,None,True))
        assert_pair(anchor,want)
        rows=[]; raw=[]
        for sample in range(8):
            trace.reset_role_trace()
            actual=tuple(t.cpu() for t in trace.forward(*inputs,None,True))
            errors=assert_pair(actual,want)
            assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(actual,anchor))
            stamps=trace.read_role_trace()
            rows.append(analyze(stamps.flatten().tolist(),32,32))
            raw.append(stamps)
        torch.save(raw,args.out/"timestamps.pt")
        receipt.update(status="PASS",input_sha256=digest(cpu),output_sha256=digest(anchor),
            errors=errors,repeat="8/8 raw-equal-to-S24",analyses=rows,timestamps_sha256=sha(args.out/"timestamps.pt"))
        print(json.dumps(rows[-1],indent=2),flush=True)
    finally:
        watch.stop.set()
        if watch.thread.is_alive(): watch.thread.join(timeout=22)
        receipt["device_watch"]=dict(records=watch.records,errors=watch.errors)
        if watch.errors: receipt["status"]="INVALID"
        (args.out/"result.json").write_text(json.dumps(receipt,indent=2)+"\n")
    if watch.errors: raise RuntimeError(str(watch.errors))
    print("[SM90 role trace] PASS diagnostic only; nsight perturbation verdict separate")


if __name__=="__main__": main()
