#!/usr/bin/env python3
"""Bind inherited arithmetic and exhaust the new finite pipeline dependency graph.

This checks progress and ownership, not CUDA memory ordering. Native async
retirement and actual repeated device numerics are separate admission gates.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PARENT='c5c0584'


def bind_sources(state,prepare,kernel):
    required=('qkp.producer_acquire(qw);','kkp.producer_acquire(kw);',
              'cp_async_fence();','cp_async_wait<0>();','__syncwarp();',
              'cutlass::arch::fence_view_async_shared();','qkp.producer_commit(qw);',
              'kkp.producer_commit(kw);')
    positions=[state.index(x) for x in required]
    if positions!=sorted(positions):raise ValueError('publication before async completion/acquire')
    if 'using MainloopQKPipeline = cutlass::PipelineAsync<1>;' not in state or \
       'using MainloopKKPipeline = cutlass::PipelineAsync<1>;' not in state:
        raise ValueError('model stage capacity differs')
    if kernel.count('producer_arv_count = PrecomputedAuxiliary ? 32 : NumAuxMathThreads;')!=2:
        raise ValueError('prepared operand arrivals differ from actual32-lane producer')
    if prepare.index('wait_barrier(s.inputs_ready,0);')>prepare.index('Arithmetic{}.compute_aux_safe'):
        raise ValueError('arithmetic before TMA completion')
    segment=prepare[prepare.index('Arithmetic{}.compute_aux_safe'):]
    if segment.index('__syncthreads();')>segment.index('auto* dst ='):
        raise ValueError('global publication before all owners complete')


def explore(chunks):
    # qkv TMA, scalar metadata, prepared-plane copy, output store, state.
    programs=[['cQ','cK','cV'], ['aA','cA','wA','aL','cL','rA'],
              ['aQK','aKK','cQK','cKK','aB','cB'], ['wO','rO'],
              ['wA','wQ','rQ','wK','wV','wKK','wB','rV','rKK','rB',
               'wQK','rQK','aO','cO','wL','rK','rA','rL']]
    programs.append(programs[-1].copy()) # the second existing V64 state WG
    programs=[tuple(p*chunks) for p in programs]
    capacities={'Q':2,'K':2,'V':1,'A':2,'L':2,'QK':1,'KK':1,'B':2,'O':1}
    readers={'Q':(4,5),'K':(4,5),'V':(4,5),'A':(1,4,5),'L':(4,5),
             'QK':(4,5),'KK':(4,5),'B':(4,5),'O':(3,)}
    producers={'Q':(0,),'K':(0,),'V':(0,),'A':(1,),'L':(1,),
               'QK':(2,),'KK':(2,),'B':(2,),'O':(4,5)}
    counts=[]
    for program in programs:
        rows=[{}]
        for op in program:
            row=rows[-1].copy();row[op]=row.get(op,0)+1;rows.append(row)
        counts.append(rows)
    def n(pc,actor,op):return counts[actor][pc[actor]].get(op,0)
    def ready(pc,actor,op):
        kind,pipe=op[0],op[1:]
        if kind=='w':return min(n(pc,p,'c'+pipe) for p in producers[pipe])>n(pc,actor,op)
        if kind=='r':return n(pc,actor,'w'+pipe)>n(pc,actor,op)
        if kind=='c' and actor!=0:return n(pc,actor,'a'+pipe)>n(pc,actor,op)
        return n(pc,actor,op)<min(n(pc,r,'r'+pipe) for r in readers[pipe])+capacities[pipe]
    start=(0,)*len(programs);seen={start};todo=deque([start]);terminals=0
    while todo:
        pc=todo.popleft();successors=[]
        if all(pc[i]==len(programs[i]) for i in range(len(programs))):terminals+=1;continue
        for actor,program in enumerate(programs):
            if pc[actor]<len(program) and ready(pc,actor,program[pc[actor]]):
                next_pc=list(pc);next_pc[actor]+=1;next_pc=tuple(next_pc);successors.append(next_pc)
                if next_pc not in seen:seen.add(next_pc);todo.append(next_pc)
        if not successors:raise ValueError(f'pipeline deadlock {pc}')
    if terminals!=1:raise ValueError('missing terminal')
    return len(seen)


def main():
    sources={}
    for name in ('scalar_gdn_aux.cuh','scalar_gdn_state.cuh','scalar_gate.cuh'):
        file='csrc/backends/sm90/'+name
        old=subprocess.check_output(['git','show',f'{PARENT}:{file}'],cwd=ROOT)
        actual=(ROOT/file).read_bytes()
        if old!=actual:raise ValueError(f'inherited arithmetic changed: {name}')
        sources[name]=hashlib.sha256(actual).hexdigest()
    state=(ROOT/'csrc/backends/sm90/precomputed_state.cuh').read_text()
    prepare=(ROOT/'csrc/backends/sm90/prepare_aux_kernel.cuh').read_text()
    kernel=(ROOT/'csrc/backends/sm90/cula/kda/sm90/kernel/kernel_kda_fwd.hpp').read_text()
    bind_sources(state,prepare,kernel)
    for s,p,k in ((state.replace('cp_async_wait<0>();','REMOVED',1),prepare,kernel),
                  (state,prepare.replace('wait_barrier(s.inputs_ready,0);','REMOVED',1),kernel),
                  (state,prepare,kernel.replace('PrecomputedAuxiliary ? 32 : NumAuxMathThreads',
                                                'PrecomputedAuxiliary ? 128 : NumAuxMathThreads',1))):
        try:bind_sources(s,p,k)
        except ValueError:pass
        else:raise AssertionError('missing async wait or phantom producer escaped')
    rows={str(n):explore(n) for n in range(1,9)}
    print(json.dumps(dict(status='PASS',scope='SOURCE_BOUND_PROGRESS_NOT_MEMORY_ORDER',
                         arithmetic_sha256=sources,states_per_chunk_count=rows,negative_controls=3),indent=2))


if __name__=='__main__':main()
