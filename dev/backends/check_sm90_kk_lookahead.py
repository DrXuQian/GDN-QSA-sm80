#!/usr/bin/env python3
"""Source-bound next-KK lifetime and exhaustive bounded ring progress.

This is not a CUDA memory-order proof: native retirement and device RAW
admission remain separate requirements.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import subprocess
from check_sm90_binary import inspect

ROOT = Path(__file__).resolve().parents[2]
FILE = 'csrc/backends/sm90/scalar_gdn_aux.cuh'


def plain(text):
    return re.sub(r'\s+', '', re.sub(r'//[^\n]*', '', text))


def bind(old, new):
    helper_start = '        // One resident future-KK fragment'
    begin = new.index(helper_start)
    loop = new.index('        for_each_aux_chunk', begin)
    helper = new[begin:loop]
    old_loop = old.index('        for_each_aux_chunk')
    if old[:old_loop] != new[:begin]:
        raise ValueError('unrelated auxiliary prefix changed')
    acc = '            auto acc_kk = partition_fragment_C(mma, Shape<_64,_64>{});'
    issue = old[old.index('            kp.consumer_wait(kr);', old_loop):
                old.index('            qp.consumer_wait(qr);', old_loop)]
    expected = ('auto acc_kk = partition_fragment_C(mma, Shape<_64,_64>{});'
                'auto issue_kk = [&]() __attribute__((always_inline)) {' + issue +
                '}; issue_kk();')
    if plain(helper) != plain(expected):
        raise ValueError('priming group/operand/readiness changed')
    body = new[loop:]
    prefetch = body[body.index('            // All reads of current acc_kk'):
                    body.index('            if constexpr (AuxInverse)')]
    if plain(prefetch) != plain('if (chunk + 1 < aux_chunk_count(int(work.seq_len))) issue_kk();'):
        raise ValueError('prefetch is not exactly the next existing chunk')
    restored = body.replace(prefetch, '').replace(
        '[&](int chunk, auto valid_tag)', '[&](int, auto valid_tag)').replace(
        '            qp.consumer_wait(qr);', acc + '\n\n' + issue +
        '            qp.consumer_wait(qr);', 1)
    if plain(restored) != plain(old[old_loop:]):
        raise ValueError('changed math, retirement or input/output lifetime')
    base = (ROOT/'csrc/backends/sm90/cula/kda/sm90/collective/mainloop_kda_fwd.hpp').read_text()
    for name, cap in [('Q', 2), ('K', 2), ('V', 1)]:
        if f'Tag::kStages{name}, Int<{cap}>' not in base:
            raise ValueError('unmodelled pipeline depth')
    for name, cap in [('QK', 2), ('KK', 2), ('O', 1)]:
        if f'using Stages{name} = cutlass::gemm::collective::StageCount<{cap}>;' not in base:
            raise ValueError('unmodelled output depth')
    state = (ROOT/'csrc/backends/sm90/scalar_gdn_state.cuh').read_bytes()
    parent = subprocess.check_output(['git', 'show', '170f34f:csrc/backends/sm90/scalar_gdn_state.cuh'], cwd=ROOT)
    if state != parent:
        raise ValueError('state consumer changed')


def progress(chunks, k_capacity=2, omit_last=False):
    auxiliary = ['wK']
    for chunk in range(chunks):
        auxiliary += ['wQ', 'rK', 'rQ', 'aKK', 'aQK']
        if chunk+1 < chunks:
            auxiliary += ['wK']
        auxiliary += ['cQK', 'cKK']
    if omit_last:
        auxiliary = auxiliary[:-2]
    state = ['wQ','rQ','wK','wV','wKK','rV','rKK','wQK','rQK','aO','cO','rK']
    programs = [['cQ','cK','cV']*chunks, auxiliary, state*chunks,
                state*chunks, ['wO','rO']*chunks]
    caps = {'Q':2,'K':k_capacity,'V':1,'QK':2,'KK':2,'O':1}
    producers = {'Q':(0,),'K':(0,),'V':(0,),'QK':(1,),'KK':(1,),'O':(2,3)}
    readers = {'Q':(1,2,3),'K':(1,2,3),'V':(2,3),'QK':(2,3),'KK':(2,3),'O':(4,)}
    counts = []
    for program in programs:
        rows = [{}]
        for op in program:
            row = rows[-1].copy(); row[op] = row.get(op,0)+1; rows.append(row)
        counts.append(rows)
    def count(pc, actor, op):
        return counts[actor][pc[actor]].get(op,0)
    def ready(pc, actor, op):
        kind, pipe = op[0], op[1:]
        if kind == 'w':
            return min(count(pc,p,'c'+pipe) for p in producers[pipe]) > count(pc,actor,op)
        if kind == 'r':
            return count(pc,actor,'w'+pipe) > count(pc,actor,op)
        if kind == 'c' and actor != 0:
            return count(pc,actor,'a'+pipe) > count(pc,actor,op)
        return count(pc,actor,op) < min(count(pc,r,'r'+pipe) for r in readers[pipe])+caps[pipe]
    start = (0,)*len(programs); seen = {start}; todo = deque([start]); terminals = 0
    while todo:
        pc = todo.popleft(); following = 0
        if all(pc[i] == len(p) for i,p in enumerate(programs)):
            terminals += 1; continue
        for actor,program in enumerate(programs):
            if pc[actor] < len(program) and ready(pc,actor,program[pc[actor]]):
                following += 1; nxt = list(pc); nxt[actor] += 1; nxt = tuple(nxt)
                if nxt not in seen:
                    seen.add(nxt); todo.append(nxt)
        if not following:
            raise ValueError(f'deadlocked data ring at {pc}')
    if terminals != 1:
        raise ValueError('terminal denominator differs')
    return len(seen)


def native(candidate, parent, log):
    if 'C7512' in log or 'C7510' in log:
        raise ValueError('WGMMA serialized')
    cm, pm = inspect(candidate), inspect(parent)
    if cm.keys() != pm.keys():
        raise ValueError('specialization denominator changed')
    pieces = re.split(r'Function\s*:\s*(\S+)', candidate)
    bodies = dict(zip(pieces[1::2], pieces[2::2])); result = {}
    for name, body in bodies.items():
        for op in ('HGMMA','HMMA','UTMALDG','UTMASTG','STG','state_stores','tail_stores'):
            if cm[name][op] != pm[name][op]:
                raise ValueError('matrix/data work changed: '+op)
        rows = re.findall(r'/\*([0-9a-f]+)\*/\s*(.*?)\s*;\s*/\*',body)
        lookahead = []
        for i,(pc,line) in enumerate(rows):
            if not line.startswith('HGMMA') or 'gsb0' not in line:
                continue
            following = next((j for j in range(i+1,len(rows))
                if rows[j][1].startswith(('WARPGROUP.DEPBAR','HGMMA'))),len(rows))
            between = rows[i+1:following]
            if not any(re.search(r'\bHMMA\.', r) for _,r in between):
                continue
            dest = int(re.match(r'HGMMA\S* R(\d+)',line)[1])
            protected = set(range(dest,dest+32))
            if any(protected & {int(x) for x in re.findall(r'\bR(\d+)\b',r)} for _,r in between):
                raise ValueError('inverse touches live KK accumulator')
            lookahead.append(dict(issue_pc=pc, next_group_pc=rows[following][0],
                                  protected_registers=[dest,dest+31],
                                  inverse_hmma_sites=sum(bool(re.search(r'\bHMMA\.',r)) for _,r in between)))
        if len(lookahead) != 1:
            raise ValueError('expected exactly one full-chunk future KK/inverse overlap')
        result[name] = dict(overlap=lookahead[0],counts=cm[name])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    p.add_argument('--candidate', type=Path)
    p.add_argument('--parent', type=Path)
    p.add_argument('--device-log', type=Path)
    args = p.parse_args()
    old = subprocess.check_output(['git','show',f'170f34f:{FILE}'],cwd=ROOT,text=True)
    new = (ROOT/FILE).read_text()
    bind(old,new)
    plants = [new.replace('warpgroup_wait<0>();','',1),
              new.replace('kp.consumer_release(kr); ++kr;', '',1).replace(
                  'qp.consumer_wait(qr);', 'kp.consumer_release(kr); ++kr; qp.consumer_wait(qr);',1),
              new.replace('chunk + 1 < aux_chunk_count','chunk + 2 < aux_chunk_count',1)]
    for plant in plants:
        try: bind(old,plant)
        except ValueError: pass
        else: raise AssertionError('source negative escaped')
    result = dict(source='PASS', source_sha256=hashlib.sha256(new.encode()).hexdigest(),
                  progress={str(n):progress(n) for n in range(1,9)},
                  scope='SOURCE_BOUND_ALL_INTERLEAVINGS_NOT_MEMORY_ORDER')
    for kwargs in ({'k_capacity':1}, {'omit_last':True}):
        try: progress(3,**kwargs)
        except ValueError: pass
        else: raise AssertionError('deadlock negative escaped')
    result['negatives'] = 5
    if args.candidate:
        cand, parent, log = args.candidate.read_text(), args.parent.read_text(), args.device_log.read_text()
        result['native'] = native(cand,parent,log)
        early_wait = cand.replace('BAR.SYNC.DEFER_BLOCKING 0xd, 0x80', 'WARPGROUP.DEPBAR.LE gsb0, 0x0',1)
        overlap = re.sub(r'HMMA\.1688\.F32 R\d+,', 'HMMA.1688.F32 R56,', cand, count=1)
        for plant, planted_log in [(parent,log),(cand,log+' C7512'),(early_wait,log),
                                   (overlap,log),(cand.replace('HGMMA.64x64','REMOVED.64x64',1),log)]:
            try: native(plant,parent,planted_log)
            except ValueError: pass
            else: raise AssertionError('native negative escaped')
        result['native_negatives'] = 5
    text = json.dumps(result,indent=2)+'\n'
    if args.out: args.out.write_text(text)
    print(text,end='')


if __name__ == '__main__':
    main()
