#!/usr/bin/env python3
"""S58: native FIFO retirement and exact source/lifetime transformation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
from check_sm90_binary import inspect
from check_sm90_paired_tail import progress

ROOT=Path(__file__).resolve().parents[2]
FILE='csrc/backends/sm90/scalar_gdn_state.cuh'


def strip(s): return re.sub(r'\s+','',re.sub(r'//[^\n]*','',s))


def source(old,new):
    marker='            gemm(kv_mma,operand_scaled,kt_desc(_,_,_,kr.index()),h);'
    prefix,tail=old.split(marker)
    tail=tail.replace('warpgroup_wait<0>();\n            warpgroup_fence_operand(h);',
                      'warpgroup_wait<1>();',1)
    tail=tail.replace('            kp.consumer_release(kr); ++kr;',
        '            warpgroup_wait<0>();\n            warpgroup_fence_operand(h);\n'
        '            kp.consumer_release(kr); ++kr;',1)
    if strip(new)!=strip(prefix+marker+tail):
        raise ValueError('not the exact staged-retirement transformation')


def split(sass):
    inspect(sass)
    p=re.split(r'Function\s*:\s*(\S+)',sass)
    return dict(zip(p[1::2],p[2::2]))


def rows(body):
    return re.findall(r'/\*[0-9a-f]+\*/\s*(.*?)\s*;\s*/\*',body)


def native(old,new):
    a,b=split(old),split(new)
    if a.keys()!=b.keys():raise ValueError('specialization denominator changed')
    result={}
    for name in b:
        control,subject=rows(a[name]),rows(b[name])
        def families(xs):
            return Counter(op for r in xs for op in re.findall(r'^(?:@!?\w+\s+)?([\w.]+)',r)
                if op.startswith(('HGMMA','HMMA','UTMA','SYNCS','BAR.','WARPGROUP')))
        want=families(control);want['WARPGROUP.DEPBAR.LE']+=4
        if families(subject)!=want:raise ValueError('math/data/retirement count changed')
        pending=[];interval=None;pairs=[]
        for i,r in enumerate(subject):
            m=re.match(r'HGMMA\.64x(64|128)x16\.F32\.BF16 (R\d+), R(\d+),',r)
            if m:pending.append((int(m[1]),m[2],int(m[3])))
            if 'WARPGROUP.DEPBAR.LE' not in r:continue
            limit=int(re.search(r', (0x[0-9a-f]+)$',r)[1],16)
            if limit==1:
                if interval is not None or [x[0] for x in pending]!=[64]*4+[128]*4:
                    raise ValueError('wait1 without exactly O2 then KV')
                first={x[2]+j for x in pending[:4] for j in range(4)}
                second={x[2]+j for x in pending[4:] for j in range(4)}
                if first & second:raise ValueError('live unscaled/scaled operands alias')
                if pending[0][1]==pending[4][1]:raise ValueError('output and H accumulators alias')
                interval=i;pending=[]
            elif limit==0:
                if interval is not None:
                    between=subject[interval+1:i]
                    if pending or not any('STSM.' in r for r in between):
                        raise ValueError('KV not independently live during output store')
                    # Full and tail both publish output under their live barrier.
                    if not any('SYNCS.ARRIVE.TRANS64.A1T0' in r for r in between):
                        raise ValueError('output publication not between retirements')
                    pairs.append((interval,i));interval=None
                pending=[]
            else:raise ValueError('unregistered completion allowance')
        if interval is not None or len(pairs)!=4:
            raise ValueError('missing four completed O2/KV epochs')
        result[name]=dict(staged_epochs=pairs,math_protocol=dict(families(subject)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--device-log',type=Path,required=True);a=p.parse_args()
    old=subprocess.check_output(['git','show',f'170f34f:{FILE}'],cwd=ROOT,text=True)
    new=(ROOT/FILE).read_text();source(old,new)
    for plant in (old,new.replace('warpgroup_wait<1>();','warpgroup_wait<2>();',1),
        new.replace('warpgroup_wait<0>();\n            warpgroup_fence_operand(h);','',1),
        new.replace('operand_scaled(i) =','operand_delta(i) =',1)):
        try:source(old,plant)
        except ValueError:continue
        raise AssertionError('source lifetime negative escaped')
    c,p=a.candidate.read_text(),a.parent.read_text()
    if 'C7512' in a.device_log.read_text():raise ValueError('serialized WGMMA')
    result=native(p,c)
    for plant in (p,c.replace('gsb0, 0x1','gsb0, 0x2',1),
                  c.replace('WARPGROUP.DEPBAR.LE','REMOVED.DEPBAR',1)):
        try:native(p,plant)
        except ValueError:continue
        raise AssertionError('native retirement negative escaped')
    print(json.dumps(dict(source_negatives=4,native_negatives=3,bodies=result,
        data_progress={str(n):progress(n) for n in range(1,9)},
        scope='FIFO completion/unchanged data-ring; not device memory-order proof'),indent=2))


if __name__=='__main__':main()
