#!/usr/bin/env python3
"""Bind the actual O1 move and exhaust its changed data-pipeline subgraph.

This is not a proof of CUDA memory ordering. The native waits and device
numeric/replay admission remain separate and mandatory.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
FILE='csrc/backends/sm90/scalar_gdn_state.cuh'
PARENT='c5c0584'


def bind_move(old,new):
    begin=old.index('            qp.consumer_wait(qr);',old.index('auto body ='))
    end=old.index('            kp.consumer_wait(kr);',begin)
    block=old[begin:end].strip()
    if new.count(block)!=1:
        raise ValueError('O1 arithmetic/waits/releases changed, not a pure move')
    if not (new.index('auto operand_delta =') < new.index(block) < new.index('qkp.consumer_wait(qkr)')):
        raise ValueError('O1 did not move between NewV conversion and O2')
    clean=lambda s:re.sub(r'\s+','',re.sub(r'//[^\n]*','',s))
    if clean(old[:begin]+old[end:])!=clean(new.replace(block,'',1)):
        raise ValueError('something besides independent O1 placement changed')


def explore(chunks):
    # Per-actor PCs determine every acquire/commit/release count. Q/K/V and
    # QK/KK are the only inter-role dependency whose release timing changed.
    programs=[['cQ','cK','cV'],
              ['wK','wQ','rK','rQ','aKK','aQK','cQK','cKK'],
              ['wK','wV','wKK','rV','rKK','wQ','rQ','wQK','rQK','rK']]
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
        # P has an atomic acquire/complete step. Async completion may delay
        # readiness, but adds no reverse dependency to this subgraph.
        return n(pc,actor,op)<min(n(pc,r,'r'+pipe) for r in readers[pipe])+caps[pipe]
    start=(0,0,0,0); seen={start}; todo=deque([start]);terminal=0
    while todo:
        pc=todo.popleft(); successors=[]
        if all(pc[i]==len(programs[i]) for i in range(4)):
            terminal+=1;continue
        for actor,prog in enumerate(programs):
            if pc[actor]<len(prog) and ready(pc,actor,prog[pc[actor]]):
                nxt=list(pc);nxt[actor]+=1;nxt=tuple(nxt);successors.append(nxt)
                if nxt not in seen:seen.add(nxt);todo.append(nxt)
        if not successors:raise ValueError(f'data pipeline deadlock: {pc}')
    assert terminal==1
    return len(seen)


def main():
    old=subprocess.check_output(['git','show',f'{PARENT}:{FILE}'],cwd=ROOT,text=True)
    new=(ROOT/FILE).read_text();bind_move(old,new)
    # These plants act on the actual implementation, not another formula.
    for plant in (new.replace('kkp.consumer_wait(kkr);','',1),
                  new.replace('qp.consumer_release(qr); ++qr;','',1),old):
        try:bind_move(old,plant)
        except ValueError:pass
        else:raise AssertionError('removed wait/release or unmoved body escaped')
    base=(ROOT/'csrc/backends/sm90/cula/kda/sm90/collective/mainloop_kda_fwd.hpp').read_text()
    for name,cap in [('Q',2),('K',2),('V',1)]:
        assert f'Tag::kStages{name}, Int<{cap}>' in base
    for name in ('QK','KK'):assert f'using Stages{name} = cutlass::gemm::collective::StageCount<2>;' in base
    rows={str(n):explore(n) for n in range(1,9)}
    print(json.dumps(dict(status='PASS',scope='SOURCE_BOUND_QKV_QKKK_PROGRESS_NOT_MEMORY_ORDER',
        states_per_chunk_count=rows,source_sha256=hashlib.sha256(new.encode()).hexdigest(),
        negatives=3,device_numerics='REQUIRED_SEPARATELY'),indent=2))


if __name__=='__main__':main()
