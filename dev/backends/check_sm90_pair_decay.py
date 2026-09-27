#!/usr/bin/env python3
"""Bind S72 exact exponent producer and unchanged alpha-ring lifetime."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from check_sm90_packed_newv import bodies

ROOT=Path(__file__).resolve().parents[2]
FILE='csrc/backends/sm90/scalar_gdn_aux.cuh'


def source(aux,helper):
    parent=subprocess.check_output(['git','show','d29ba16:'+FILE],cwd=ROOT,text=True)
    expected=parent.replace('#include "relative_gate_layout.cuh"',
        '#include "relative_gate_layout.cuh"\n#include "pair_decay.cuh"').replace(
        'cute::array_aligned<float, 64 * Base::StagesAlpha::value> relative_gate;',
        'cute::array_aligned<float, 64 * Base::StagesAlpha::value> relative_gate;\n        cute::array_aligned<float, 4096 * Base::StagesAlpha::value> pair_decay;').replace(
        'smem.relative_gate[relative_gate_index(stage,lane+32)] = rhi;',
        'smem.relative_gate[relative_gate_index(stage,lane+32)] = rhi;\n            publish_pair_decay(lo,hi,stage,smem.pair_decay.data());').replace(
        '                float row_log = alpha(row,0,ar.index());\n                float col_log = alpha(col,0,ar.index());\n','').replace(
        'float decay = exp2f(row_log-col_log);',
        'float decay = smem.pair_decay[pair_decay_index(ar.index(),row,col)];')
    if aux!=expected:raise ValueError('unregistered source/arithmetic/lifetime change')
    if '__exp2f' in helper or helper.count('exp2f(__fsub_rn(')!=4 or 'col<32' not in helper:
        raise ValueError('standard rounded exponent or producer denominator changed')
    for expr in ('stage,lane,col','stage,lane,col+32','stage,lane+32,col','stage,lane+32,col+32'):
        if 'target[pair_decay_index('+expr+')]' not in helper:
            raise ValueError('missing producer quadrant')
    gate=(ROOT/'csrc/backends/sm90/scalar_gate.cuh').read_text()
    if not gate.index('pipeline.producer_acquire(stage)')<gate.index('publish_factors(lane, lo, hi, stage.index())')<gate.index('pipeline.producer_commit(stage)'):
        raise ValueError('factor stores not inside alpha ownership')
    body=aux[aux.index('CUTE_DEVICE void compute_aux_safe'):]
    if not body.index('ap.consumer_wait(ar)')<body.index('smem.pair_decay[')<body.index('ap.consumer_release(ar)'):
        raise ValueError('reader outside alpha lifetime')


def native(candidate,parent,log):
    if re.search(r'C751[02]',log):raise ValueError('WGMMA serialization')
    stacks=list(map(int,re.findall(r'(\d+) bytes stack frame',log)))
    if len(stacks)!=4 or max(stacks)>128:raise ValueError('registered stack limit / body count')
    c,p=bodies(candidate),bodies(parent)
    if c.keys()!=p.keys() or len(c)!=4:raise ValueError('four-body denominator changed')
    result={}
    fixed=('HMMA','HGMMA','UTMA','WARPGROUP','USETMAXREG','BAR.','SYNCS')
    for key,rows in c.items():
        cc,pc=Counter(o for _,o,_ in rows),Counter(o for _,o,_ in p[key])
        if {o:n for o,n in cc.items() if o.startswith(fixed)} != {o:n for o,n in pc.items() if o.startswith(fixed)}:
            raise ValueError('matrix/TMA/barrier/register role protocol changed')
        intervals=[]
        for data,expected in ((p[key],30),(rows,0)):
            indices=[i for i,(_,o,_) in enumerate(data) if o.startswith('HGMMA')]
            for start in (indices[0],indices[16]):
                end=next(i for i in range(start,len(data)) if data[i][1].startswith('BAR.SYNC') and '0xd, 0x80' in data[i][2])
                count=sum(op=='MUFU.EX2' for _,op,_ in data[start:end])
                if count!=expected:raise ValueError('auxiliary exponent placement unchanged/missing')
                intervals.append(dict(begin=data[start][0],end=data[end][0],exponents=count))
        if cc['MUFU.EX2'] != pc['MUFU.EX2']-60+4:
            raise ValueError('actual rolled producer not four new exponent sites')
        result[key]=dict(aux_intervals_parent_then_candidate=intervals,sites=len(rows),
                         exponents=cc['MUFU.EX2'],parent_exponents=pc['MUFU.EX2'],
                         local_loads=cc['LDL'],local_stores=cc['STL'],opcodes=dict(cc))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','parent','device-log'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); aux=(ROOT/FILE).read_text(); helper=(ROOT/'csrc/backends/sm90/pair_decay.cuh').read_text()
    source(aux,helper)
    c,b,l=a.candidate.read_text(),a.parent.read_text(),a.device_log.read_text()
    got=native(c,b,l)
    def red(fn):
        try:fn()
        except (ValueError,StopIteration):return
        raise AssertionError('negative escaped')
    red(lambda:source(aux.replace('publish_pair_decay(lo,hi,stage,smem.pair_decay.data());',''),helper))
    red(lambda:source(aux.replace('pair_decay_index(ar.index(),row,col)','pair_decay_index(0,row,col)'),helper))
    red(lambda:source(aux,helper.replace('exp2f(','__exp2f(')))
    red(lambda:source(aux,helper.replace('col<32','col<31')))
    red(lambda:native(b,b,l))
    red(lambda:native(c.replace('MUFU.EX2','REMOVED.EX2',1),b,l))
    print(json.dumps(dict(status='PASS',scope='SOURCE_LIFETIME_AND_STATIC_NATIVE_NOT_SPEED',
        bodies=got,negatives=6,hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (a.candidate,a.parent,a.device_log)}),indent=2))


if __name__=='__main__':main()
