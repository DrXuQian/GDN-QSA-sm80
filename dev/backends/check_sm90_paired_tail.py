#!/usr/bin/env python3
"""Bind actual O2/KV async lifetime; matrix arithmetic is unchanged."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import subprocess
from check_sm90_binary import inspect

ROOT=Path(__file__).resolve().parents[2]
FILE='csrc/backends/sm90/scalar_gdn_state.cuh'


def clean(s):return re.sub(r'\s+','',re.sub(r'//[^\n]*','',s))


def source(old,new):
    start='            auto operand_delta =';end='            kp.consumer_release(kr); ++kr;'
    begin=old.index(start);begin2=new.index(start)
    finish=old.index(end,begin);finish2=new.index(end,begin2)
    if old[:begin]!=new[:begin2] or old[finish:]!=new[finish2:]:raise ValueError('unrelated source changed')
    a,b=old[begin:finish],new[begin2:finish2]
    # The publishing block and rounded gain expressions are preserved exactly.
    pub=a[a.index('            {\n                auto output'):a.index('\n\n            // Consume')]
    if b.count(pub)!=1:raise ValueError('output arithmetic/publication changed')
    if not b.index('warpgroup_wait<0>();')<b.index('qkp.consumer_release(qkr);')<b.index(pub):
        raise ValueError('early QK release or output read')
    ops=re.findall(r'(?:gemm_zero_acc|gemm)\([^;]+;',b)
    if [s.replace('operand_scaled','operand_delta') for s in ops]!=re.findall(r'(?:gemm_zero_acc|gemm)\([^;]+;',a):
        raise ValueError('matrix inputs or dot order changed')
    if b.count('warpgroup_wait<0>();')!=1 or b.count('warpgroup_commit_batch();')!=2:
        raise ValueError('wrong completion/commit count')
    if b.index('warpgroup_wait<0>();')<b.index('gemm(kv_mma,'):raise ValueError('retired before KV issue')
    for fragment in ('float decay_h = smem.gate_factors[ar.index()*128+valid-1];',
        'for (int i=0; i<size(h); ++i) h(i) *= decay_h;',
        'float gain = smem.relative_gate[relative_gate_index(ar.index(),t)];',
        'operand_scaled(i) = Element(float(operand_delta(i))*gain);',
        'auto operand_scaled = make_fragment_like<Element>(operand_delta);',
        'alp.consumer_wait(alr);','warpgroup_fence_operand(acc_o);',
        'warpgroup_fence_operand(h);','warpgroup_fence_operand(operand_scaled);'):
        if clean(fragment) not in clean(b):raise ValueError('lifetime/gain obligation missing')
    if re.search(r'operand_delta\(i\)\s*=',b):raise ValueError('async O2 operand overwritten')


def native(cand,old,log):
    if 'C7512' in log or 'C7510' in log:raise ValueError('serialized WGMMA')
    cm,pm=inspect(cand),inspect(old)
    if cm.keys()!=pm.keys():raise ValueError('changed specialization inventory')
    def split(text):
        p=re.split(r'Function\s*:\s*(\S+)',text);return dict(zip(p[1::2],p[2::2]))
    cb,pb=split(cand),split(old);result={}
    for key in cm:
        for op in ('HGMMA','HMMA','UTMALDG','UTMASTG','STG','state_stores','tail_stores'):
            if cm[key][op]!=pm[key][op]:raise ValueError('matrix/TMA/publication changed')
        rows=re.findall(r'/\*[0-9a-f]+\*/\s*(.*?)\s*;\s*/\*',cb[key])
        epochs=[];current=[]
        for i,r in enumerate(rows):
            m=re.match(r'HGMMA\.64x(64|128)x16\.F32\.BF16 (R\d+), R(\d+),',r)
            if m:current.append((i,int(m[1]),m[2],int(m[3])))
            if 'WARPGROUP.DEPBAR' in r:
                if 'gsb0, 0x0' not in r:raise ValueError('incomplete retirement')
                if current:epochs.append(current);current=[]
        if current:raise ValueError('unretired matrix')
        pairs=[e for e in epochs if any(r[1]==128 for r in e)]
        if len(pairs)!=4:raise ValueError('missing output/update epoch')
        for p in pairs:
            if [r[1] for r in p]!=[64]*4+[128]*4:raise ValueError('O2 and KV still separated')
            regsets=[{r[3]+j for r in part for j in range(4)} for part in (p[:4],p[4:])]
            if regsets[0]&regsets[1]:raise ValueError('live O2/KV operands alias')
            if len({r[2] for r in p[:4]})!=1 or len({r[2] for r in p[4:]})!=1 or p[0][2]==p[4][2]:
                raise ValueError('O/H accumulator aliases')
        if pb[key].count('WARPGROUP.DEPBAR.LE')-cb[key].count('WARPGROUP.DEPBAR.LE')!=4:
            raise ValueError('retirement delta not four')
        result[key]=dict(paired_epochs=4,sites=len(rows),counts=cm[key])
    return result


def progress(chunks):
    # Source-bound shared data ring, including BOTH state publishers and the
    # output TMA drainer. Metadata has a separate monotone progress lemma:
    # a producer blocked by a full alpha-last ring has already published the
    # current consumer's slot; it cannot block that same slot's first publish.
    # Moving its wait before O acquire introduces no reverse data dependency.
    # Async completion time and CUDA memory visibility are not modelled.
    state=['wQ','rQ','wK','wV','wKK','rV','rKK','wQK','rQK','aO','cO','rK']
    programs=[['cQ','cK','cV'],['wK','wQ','rK','rQ','aKK','aQK','cQK','cKK'],
              state,state,['wO','rO']]
    programs=[p*chunks for p in programs]
    caps={'Q':2,'K':2,'V':1,'QK':2,'KK':2,'O':1}
    producers={'Q':(0,),'K':(0,),'V':(0,),'QK':(1,),'KK':(1,),'O':(2,3)}
    readers={'Q':(1,2,3),'K':(1,2,3),'V':(2,3),'QK':(2,3),'KK':(2,3),'O':(4,)}
    counts=[]
    for prog in programs:
        rows=[{}]
        for op in prog:
            d=rows[-1].copy();d[op]=d.get(op,0)+1;rows.append(d)
        counts.append(rows)
    def n(pc,a,op):return counts[a][pc[a]].get(op,0)
    def ready(pc,a,op):
        kind,p=op[0],op[1:]
        if kind=='w':return min(n(pc,x,'c'+p) for x in producers[p])>n(pc,a,op)
        if kind=='r':return n(pc,a,'w'+p)>n(pc,a,op)
        if kind=='c' and a!=0:return n(pc,a,'a'+p)>n(pc,a,op)
        return n(pc,a,op)<min(n(pc,x,'r'+p) for x in readers[p])+caps[p]
    initial=(0,)*5;seen={initial};todo=deque([initial]);terminal=0
    while todo:
        pc=todo.popleft();following=[]
        if all(pc[i]==len(programs[i]) for i in range(5)):terminal+=1;continue
        for actor,prog in enumerate(programs):
            if pc[actor]<len(prog) and ready(pc,actor,prog[pc[actor]]):
                nxt=list(pc);nxt[actor]+=1;nxt=tuple(nxt);following.append(nxt)
                if nxt not in seen:seen.add(nxt);todo.append(nxt)
        if not following:raise ValueError('data ring deadlock')
    if terminal!=1:raise ValueError('terminal coverage')
    return len(seen)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path);parser.add_argument('--parent',type=Path)
    parser.add_argument('--device-log',type=Path);args=parser.parse_args()
    old=subprocess.check_output(['git','show',f'06b7471:{FILE}'],cwd=ROOT,text=True)
    new=(ROOT/FILE).read_text();source(old,new)
    plants=(old,new.replace('operand_scaled(i) =','operand_delta(i) =',1),
        new.replace('warpgroup_wait<0>();\n            warpgroup_fence_operand(h);','warpgroup_fence_operand(h);',1),
        new.replace('qkp.consumer_release(qkr); ++qkr;', '',1).replace(
            'gemm(kv_mma,operand_scaled,','qkp.consumer_release(qkr); ++qkr; gemm(kv_mma,operand_scaled,',1))
    for plant in plants:
        try:source(old,plant)
        except ValueError:pass
        else:raise AssertionError('source negative escaped')
    base=(ROOT/'csrc/backends/sm90/cula/kda/sm90/collective/mainloop_kda_fwd.hpp').read_text()
    for name,cap in [('Q',2),('K',2),('V',1)]:
        if f'Tag::kStages{name}, Int<{cap}>' not in base:raise ValueError('pipeline capacity changed')
    if 'using StagesO = cutlass::gemm::collective::StageCount<1>;' not in base:
        raise ValueError('output pipeline capacity changed')
    result=dict(source='PASS',source_sha256=hashlib.sha256(new.encode()).hexdigest(),source_negatives=4,
        progress={str(n):progress(n) for n in range(1,9)},progress_scope='DATA_RING_WITH_METADATA_LEMMA_NOT_MEMORY_ORDER')
    if args.candidate:
        c,p,l=args.candidate.read_text(),args.parent.read_text(),args.device_log.read_text()
        result['native']=native(c,p,l)
        for plant,log in ((p,l),(c,l+' C7512'),(c.replace('HGMMA.64x128','REMOVED.64x128',1),l)):
            try:native(plant,p,log)
            except ValueError:pass
            else:raise AssertionError('native negative escaped')
        result['native_negatives']=3
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
