#!/usr/bin/env python3
"""Bind paired issue to real state source and exhaust changed data dependencies.

The progress model is not a memory-order proof. Native retirement plus the
independent device oracle/replay remain mandatory.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
FILE='csrc/backends/sm90/scalar_gdn_state.cuh'
PARENT='06b7471'


def norm(s):
    return re.sub(r'\s+','',re.sub(r'//[^\n]*','',s))


def bind(old,new):
    start='            qp.consumer_wait(qr);'
    end='            vp.consumer_wait(vr);'
    before,rest=old.split(start,1);oldpart,after=rest.split(end,1)
    prefix,rest=new.split(start,1);newpart,suffix=rest.split(end,1)
    if before!=prefix or after!=suffix:raise ValueError('unrelated arithmetic/source changed')
    calls=re.findall(r'gemm_zero_acc\([^;]+;',newpart)
    oldcalls=re.findall(r'gemm_zero_acc\([^;]+;',oldpart)
    if calls!=oldcalls or len(calls)!=2:raise ValueError('dot order/operand/destination changed')
    if newpart.count('auto operand_h =')!=1 or newpart.count('warpgroup_wait<0>();')!=1:
        raise ValueError('H cast or retirement count is not paired')
    if newpart.count('warpgroup_commit_batch();')!=1:raise ValueError('missing paired commit')
    for token in ('kp.consumer_wait(kr);','warpgroup_fence_operand(acc_o);',
                  'warpgroup_fence_operand(acc_sk);','warpgroup_fence_operand(operand_h);',
                  'order.ordered_or_wait(wg);','order.notify_next_blocked(wg);',
                  'qp.consumer_release(qr); ++qr;'):
        if token not in newpart:raise ValueError('missing dependency: '+token)
    waits=[newpart.index('kp.consumer_wait(kr);'),newpart.index(calls[0]),newpart.index(calls[1]),
           newpart.index('warpgroup_commit_batch();'),newpart.index('warpgroup_wait<0>();'),
           newpart.index('qp.consumer_release(qr); ++qr;')]
    if waits!=sorted(waits):raise ValueError('early input release or use before wait')
    # Compare the real O1 scalar arithmetic verbatim after the paired wait.
    gain='''CUTE_UNROLL
                for (int i=0; i<size(acc_o); ++i) {
                    auto [dv,t] = c_output(i);
                    acc_o(i) *= smem.gate_factors[ar.index()*128+64+t];
                }'''
    if norm(gain) not in norm(oldpart) or norm(gain) not in norm(newpart):
        raise ValueError('O1 rounded gain changed')
    if 'typename Base::TiledMmaSK::LayoutA_TV>' not in newpart:
        raise ValueError('missing exact operand layout type assertion')


def explore(chunks):
    # P(Q,K,V), aux(QK/KK), two state groups. Acquires/completions are
    # atomic in this reduced dependency graph; native async waits separate.
    programs=[['cQ','cK','cV'],
              ['wK','wQ','rK','rQ','aKK','aQK','cQK','cKK'],
              ['wQ','wK','rQ','wV','wKK','rV','rKK','wQK','rQK','rK']]
    programs=[tuple(p*chunks) for p in (programs[0],programs[1],programs[2],programs[2])]
    caps={'Q':2,'K':2,'V':1,'QK':2,'KK':2}
    readers={'Q':(1,2,3),'K':(1,2,3),'V':(2,3),'QK':(2,3),'KK':(2,3)}
    producer={'Q':0,'K':0,'V':0,'QK':1,'KK':1}
    counts=[]
    for prog in programs:
        rows=[{}]
        for op in prog:
            row=rows[-1].copy();row[op]=row.get(op,0)+1;rows.append(row)
        counts.append(rows)
    def n(pc,actor,op):return counts[actor][pc[actor]].get(op,0)
    def ready(pc,actor,op):
        kind,pipe=op[0],op[1:]
        if kind=='w':return n(pc,producer[pipe],'c'+pipe)>n(pc,actor,op)
        if kind=='r':return n(pc,actor,'w'+pipe)>n(pc,actor,op)
        if kind=='c' and actor==1:return n(pc,actor,'a'+pipe)>n(pc,actor,op)
        return n(pc,actor,op)<min(n(pc,r,'r'+pipe) for r in readers[pipe])+caps[pipe]
    start=(0,0,0,0);seen={start};todo=deque([start]);terminal=0
    while todo:
        pc=todo.popleft();successors=[]
        if all(pc[i]==len(programs[i]) for i in range(4)):
            terminal+=1;continue
        for actor,prog in enumerate(programs):
            if pc[actor]<len(prog) and ready(pc,actor,prog[pc[actor]]):
                nxt=list(pc);nxt[actor]+=1;nxt=tuple(nxt);successors.append(nxt)
                if nxt not in seen:seen.add(nxt);todo.append(nxt)
        if not successors:raise ValueError(f'deadlock: {pc}')
    assert terminal==1
    return len(seen)


def main():
    old=subprocess.check_output(['git','show',f'{PARENT}:{FILE}'],cwd=ROOT,text=True)
    new=(ROOT/FILE).read_text();bind(old,new)
    for plant in (old,new.replace('warpgroup_wait<0>();','',1),
                  new.replace('kp.consumer_wait(kr);','',1),
                  new.replace('qp.consumer_release(qr); ++qr;', '',1).replace(
                      'gemm_zero_acc(o1_mma,','qp.consumer_release(qr); ++qr; gemm_zero_acc(o1_mma,',1),
                  new.replace('c_output(i);','c_output(i+1);',1)):
        try:bind(old,plant)
        except ValueError:pass
        else:raise AssertionError('paired source negative escaped')
    base=(ROOT/'csrc/backends/sm90/cula/kda/sm90/collective/mainloop_kda_fwd.hpp').read_text()
    for name,cap in [('Q',2),('K',2),('V',1)]:
        assert f'Tag::kStages{name}, Int<{cap}>' in base
    for name in ('QK','KK'):
        assert f'using Stages{name} = cutlass::gemm::collective::StageCount<2>;' in base
    print(json.dumps(dict(status='PASS',scope='SOURCE_BOUND_PROGRESS_NOT_MEMORY_ORDER',
        states_per_chunk_count={str(n):explore(n) for n in range(1,9)},
        source_sha256=hashlib.sha256(new.encode()).hexdigest(),negatives=5),indent=2))


if __name__=='__main__':main()
