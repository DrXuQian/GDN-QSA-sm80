#!/usr/bin/env python3
"""Actual native/types and bounded ring-progress gate for auxiliary depths."""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def keyed_native(path):
    result={}
    for section in path.read_text().split('Function :')[1:]:
        symbol=section.splitlines()[0].strip()
        if 'FlatKernelTmaWarpSpecializedKdaFwd' not in symbol: continue
        name=subprocess.check_output(['c++filt',symbol],text=True)
        gates=[g for g in ('float','cutlass::bfloat16_t') if f'(kda::sm90::kernel::Tag)11, {g}>' in name]
        initials=[i for i in ('false','true') if f'(kda::sm90::kernel::Tag)8, cute::C<{i}>' in name]
        if len(gates)!=1 or len(initials)!=1: raise ValueError('unknown specialization')
        key=(gates[0],initials[0])
        if key in result: raise ValueError('duplicate specialization')
        result[key]=re.findall(r'/\*[0-9a-f]+\*/\s*(.*?)\s*;\s*/\*',section)
    if len(result)!=4: raise ValueError('missing specialization')
    return result


def native(got,parent,config,qk,kk,log):
    if 'C7512' in log or 'C7510' in log: raise ValueError('serialized WGMMA')
    if got.keys()!=parent.keys() or len(got)!=4: raise ValueError('native denominator changed')
    if config==0 and got!=parent: raise ValueError('default native instructions/operands changed')
    def count(rows):
        return Counter(re.match(r'(?:@!?\w+\s+)?([A-Z][\w.]*)',r)[1] for r in rows)
    for key,rows in got.items():
        a,b=count(rows),count(parent[key])
        for prefix in ('HGMMA','HMMA','WARPGROUP','UTMA','USETMAXREG'):
            if {o:n for o,n in a.items() if o.startswith(prefix)} != {o:n for o,n in b.items() if o.startswith(prefix)}:
                raise ValueError('changed matrix/data protocol: '+prefix)
        if a['SYNCS.EXCH.64']!=2*(12+qk+kk): raise ValueError('native ring-init count differs from actual depths')
        if not all('gsb0, 0x0' in r for r in rows if 'WARPGROUP.DEPBAR' in r):
            raise ValueError('incomplete retirement')
    return dict(bodies=4,sites=sum(map(len,got.values())),default_native_identical=config==0)


def admit_types(rows,config):
    if len(rows)!=4 or {(r['gate_fp32'],r['initial']) for r in rows}!={(0,0),(0,1),(1,0),(1,1)}:
        raise ValueError('actual type denominator')
    expected=(1 if config&1 else 2,1 if config&2 else 2)
    for r in rows:
        if r['config']!=config or (r['qk'],r['kk'])!=expected:
            raise ValueError('actual axes differ from registered tuple')
        if (r['qk_cells'],r['kk_cells'])!=(4096*r['qk'],4096*r['kk']):
            raise ValueError('storage not tied to own pipeline depth')
        if r['threads']!=512 or not 0<r['shared_bytes']<=168960:
            raise ValueError('unexpected resources')
    return expected


def progress(chunks,qk,kk):
    state=['wQ','rQ','wK','wV','wKK','rV','rKK','wQK','rQK','aO','cO','rK']
    programs=[['cQ','cK','cV']*chunks,
              ['wK','wQ','rK','rQ','aKK','aQK','cQK','cKK']*chunks,
              state*chunks,state*chunks,['wO','rO']*chunks]
    caps={'Q':2,'K':2,'V':1,'QK':qk,'KK':kk,'O':1}
    producers={'Q':(0,),'K':(0,),'V':(0,),'QK':(1,),'KK':(1,),'O':(2,3)}
    readers={'Q':(1,2,3),'K':(1,2,3),'V':(2,3),'QK':(2,3),'KK':(2,3),'O':(4,)}
    counts=[]
    for program in programs:
        rows=[{}]
        for op in program:
            row=rows[-1].copy();row[op]=row.get(op,0)+1;rows.append(row)
        counts.append(rows)
    def n(pc,a,op): return counts[a][pc[a]].get(op,0)
    def ready(pc,a,op):
        kind,pipe=op[0],op[1:]
        if kind=='w': return min(n(pc,p,'c'+pipe) for p in producers[pipe])>n(pc,a,op)
        if kind=='r': return n(pc,a,'w'+pipe)>n(pc,a,op)
        if kind=='c' and a!=0: return n(pc,a,'a'+pipe)>n(pc,a,op)
        return n(pc,a,op)<min(n(pc,r,'r'+pipe) for r in readers[pipe])+caps[pipe]
    start=(0,)*5;seen={start};todo=deque([start]);terminal=0
    while todo:
        pc=todo.popleft();following=0
        if all(pc[i]==len(p) for i,p in enumerate(programs)): terminal+=1;continue
        for a,p in enumerate(programs):
            if pc[a]<len(p) and ready(pc,a,p[pc[a]]):
                following+=1;nxt=list(pc);nxt[a]+=1;nxt=tuple(nxt)
                if nxt not in seen:seen.add(nxt);todo.append(nxt)
        if not following: raise ValueError('deadlocked data ring')
    if terminal!=1: raise ValueError('terminal denominator')
    return len(seen)


def source():
    # No candidate changes a producer/consumer action, operand or arithmetic.
    for file in ('scalar_gdn_aux.cuh','scalar_gdn_state.cuh','aux_chunk_loop.cuh','ordered_pair.cuh'):
        rel='csrc/backends/sm90/'+file
        old=subprocess.check_output(['git','show','170f34f:'+rel],cwd=ROOT)
        if (ROOT/rel).read_bytes()!=old: raise ValueError('changed delivery/math source: '+file)
    path='csrc/backends/sm90/cula/kda/sm90/collective/mainloop_kda_fwd.hpp'
    old=subprocess.check_output(['git','show','170f34f:'+path],cwd=ROOT,text=True)
    expected=old.replace('StageCount<2>;\n    using StagesKK = cutlass::gemm::collective::StageCount<2>;',
        'StageCount<find_option_t<Tag::kStagesQK, Int<2>, Options>::value>;\n'
        '    using StagesKK = cutlass::gemm::collective::StageCount<find_option_t<Tag::kStagesKK, Int<2>, Options>::value>;')
    pos=expected.index('    using SmemLayoutKK =')
    expected=expected[:pos]+expected[pos:].replace('Int<StagesQK::value>','Int<StagesKK::value>',1)
    if expected!=(ROOT/path).read_text(): raise ValueError('unregistered collective change')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--device-log',type=Path,required=True);p.add_argument('--types',type=Path,required=True)
    p.add_argument('--config',type=int,choices=range(4),required=True);p.add_argument('--out',type=Path)
    a=p.parse_args();source()
    rows=[json.loads(line) for line in a.types.read_text().splitlines()]
    qk,kk=admit_types(rows,a.config)
    got,parent=keyed_native(a.candidate),keyed_native(a.parent);log=a.device_log.read_text()
    result=dict(status='PASS',config=a.config,types=rows,native=native(got,parent,a.config,qk,kk,log),
                progress={str(n):progress(n,qk,kk) for n in range(1,9)},
                progress_scope='SOURCE_BOUND_DATA_RING_NOT_MEMORY_ORDER',performance='NOT_INFERRED')
    for plant in (rows[:-1],rows[:1]*4,[r|{'kk_cells':r['kk_cells']-1} for r in rows]):
        try: admit_types(plant,a.config)
        except ValueError: pass
        else: raise AssertionError('type-denominator negative escaped')
    key=next(iter(got))
    for plant,plog in (({k:v for k,v in got.items() if k!=key},log),(got,log+' C7512'),
                       ({**got,key:[r for r in got[key] if 'WARPGROUP.DEPBAR' not in r]},log)):
        try: native(plant,parent,a.config,qk,kk,plog)
        except ValueError: pass
        else: raise AssertionError('native negative escaped')
    result['negatives']=6
    text=json.dumps(result,indent=2)+'\n'
    if a.out:a.out.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
