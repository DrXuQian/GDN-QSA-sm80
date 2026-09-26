#!/usr/bin/env python3
"""Paired graph-event screening only; final admission remains complete nsys.

Both gates, alternating graph order, CPU oracle and direct parent byte checks.
No reference-library result, selector update or speed verdict is made here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks'),str(ROOT/'tests')]
import torch
from bench_sm90_hopper import DeviceWatch
from profile_sm90_cula import build_receipt
from test_ppu_gdn_backend import fixture,assert_pair,digest
from gdn_qsa_sm80.reference.gdn_chunk_ref import torch_recurrent_gated_delta_rule
from gdn_qsa_sm80.gdn_sm90_interface import gdn_chunk_sm90


def raw_equal(got,want):
    if len(got)!=len(want) or not all(torch.equal(x.contiguous().view(torch.uint8),
            y.contiguous().view(torch.uint8)) for x,y in zip(got,want)):
        raise AssertionError('candidate/parent raw-byte mismatch')


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('preserve existing screening receipt')
    a.out.mkdir(parents=True)
    builds={r:build_receipt(path,'cuda_sm90','native')
            for r,path in [('candidate',a.candidate),('control',a.control)]}
    if os.getenv('CUDA_VISIBLE_DEVICES') not in (None,'0'):
        raise ValueError('require the unremapped single-Hopper host')
    watch=DeviceWatch(0)
    result=dict(scope='SCREENING_ONLY_NOT_SPEED_ADMISSION',builds=builds,screen={})
    try:
        for _ in range(3):watch.sample(idle=True);time.sleep(.2)
        watch.thread.start()
        props=torch.cuda.get_device_properties(0)
        if (props.major,props.minor)!=(9,0) or 'PPU' in props.name:
            raise ValueError('physical H800 control only')
        result.update(device=props.name,sms=props.multi_processor_count)
        torch.set_num_threads(1)
        for gate in (-.1,-1.):
            cpu=fixture(1,2048,16,32,gate)
            q,k,v,g,b=cpu
            want=torch_recurrent_gated_delta_rule(q.repeat_interleave(2,2),
                k.repeat_interleave(2,2),v,g,b,output_final_state=True)
            tensors=tuple(t.cuda() for t in cpu)
            def call(role):
                os.environ['GDN_QSA_SM90_EXTENSION']=str(getattr(a,role).resolve())
                return gdn_chunk_sm90(*tensors,backend='cuda_sm90')
            anchors={r:tuple(t.cpu() for t in call(r)) for r in ('control','candidate')}
            errors={r:assert_pair(out,want) for r,out in anchors.items()}
            raw_equal(anchors['candidate'],anchors['control'])
            for _ in range(8):
                for role in ('control','candidate'):
                    raw_equal(tuple(t.cpu() for t in call(role)),anchors[role])
            graphs,captured={},{}
            for role in ('control','candidate'):
                graphs[role]=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[role]):captured[role]=[call(role) for _ in range(8)]
            for _ in range(3):
                graphs['control'].replay();graphs['candidate'].replay()
            torch.cuda.synchronize()
            samples={r:[] for r in graphs}
            for i in range(8):
                watch.sample()
                for role in (('control','candidate') if i%2==0 else ('candidate','control')):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();graphs[role].replay();end.record();end.synchronize()
                    samples[role].append(start.elapsed_time(end)*1000/8)
            for role,outputs in captured.items():
                for out in outputs:
                    actual=tuple(t.cpu() for t in out)
                    assert_pair(actual,want);raw_equal(actual,anchors[role])
            stats={r:dict(median_us=statistics.median(xs),range_us=[min(xs),max(xs)],samples_us=xs)
                   for r,xs in samples.items()}
            stats.update(candidate_over_control=stats['candidate']['median_us']/stats['control']['median_us'],
                         errors=errors,input_sha256=digest(cpu),output_sha256=digest(anchors['candidate']),
                         admission='CPU+PARENT_DIRECT_BYTES+8REPEAT+EVERY_CAPTURED_OUTPUT/PASS')
            result['screen'][str(gate)]=stats
            print(f"[stage screening] gate={gate} candidate={stats['candidate']['median_us']:.3f}us "
                  f"control={stats['control']['median_us']:.3f}us ratio={stats['candidate_over_control']:.5f} "
                  'SCREENING_ONLY',flush=True)
        watch.sample();watch.stop.set();watch.thread.join(timeout=22)
        if watch.thread.is_alive() or watch.errors:raise RuntimeError('invalid device watch')
        result['status']='PASS'
        result['harness_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    except Exception as e:
        watch.errors.append(str(e));raise
    finally:
        watch.stop.set()
        if watch.thread.is_alive():watch.thread.join(timeout=22)
        (a.out/'device-watch.json').write_text(json.dumps(dict(records=watch.records,errors=watch.errors),indent=2)+'\n')


if __name__=='__main__':main()
