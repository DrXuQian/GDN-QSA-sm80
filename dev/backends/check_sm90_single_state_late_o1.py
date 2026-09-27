#!/usr/bin/env python3
"""Bind S67's sole source move and its one-state data-ring dependencies."""
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'csrc/backends/sm90/scalar_gdn_state.cuh'
PARENT = 'bbddadd'


def bind_move(old, new):
    start = old.index('            qp.consumer_wait(qr);', old.index('auto body ='))
    end = old.index('            kp.consumer_wait(kr);', start)
    block = old[start:end].strip()
    if new.count(block) != 1:
        raise ValueError('O1 arithmetic/wait/release changed')
    if not new.index('auto operand_delta =') < new.index(block) < new.index('qkp.consumer_wait(qkr)'):
        raise ValueError('O1 not after BF16 NewV and before O2')
    def clean(s):
        return re.sub(r'\s+', '', re.sub(r'//[^\n]*', '', s))
    if clean(old[:start] + old[end:]) != clean(new.replace(block, '', 1)):
        raise ValueError('unregistered change beyond O1 placement')


def progress(chunks, omit_q_release=False):
    programs = [
        ['cQ', 'cK', 'cV'],
        ['wK', 'wQ', 'rK', 'rQ', 'aKK', 'aQK', 'cQK', 'cKK'],
        ['wK', 'wV', 'wKK', 'rV', 'rKK', 'wQ', 'rQ', 'wQK', 'rQK', 'rK'],
    ]
    if omit_q_release:
        programs[2].remove('rQ')
    programs = [p * chunks for p in programs]
    caps = {'Q': 2, 'K': 2, 'V': 1, 'QK': 2, 'KK': 2}
    readers = {'Q': (1, 2), 'K': (1, 2), 'V': (2,), 'QK': (2,), 'KK': (2,)}
    producers = {'Q': 0, 'K': 0, 'V': 0, 'QK': 1, 'KK': 1}
    prefixes = []
    for program in programs:
        rows = [{}]
        for op in program:
            row = rows[-1].copy()
            row[op] = row.get(op, 0) + 1
            rows.append(row)
        prefixes.append(rows)
    def n(pc, actor, op):
        return prefixes[actor][pc[actor]].get(op, 0)
    def ready(pc, actor, op):
        kind, pipe = op[0], op[1:]
        if kind == 'w':
            return n(pc, producers[pipe], 'c' + pipe) > n(pc, actor, op)
        if kind == 'r':
            return n(pc, actor, 'w' + pipe) > n(pc, actor, op)
        if kind == 'c' and actor == 1:
            return n(pc, actor, 'a' + pipe) > n(pc, actor, op)
        return n(pc, actor, op) < min(n(pc, r, 'r' + pipe) for r in readers[pipe]) + caps[pipe]
    start = (0, 0, 0)
    pending, seen, terminals = deque([start]), {start}, 0
    while pending:
        pc = pending.popleft()
        if all(pc[a] == len(p) for a, p in enumerate(programs)):
            terminals += 1
            continue
        advanced = False
        for a, program in enumerate(programs):
            if pc[a] < len(program) and ready(pc, a, program[pc[a]]):
                advanced = True
                following = list(pc)
                following[a] += 1
                following = tuple(following)
                if following not in seen:
                    seen.add(following)
                    pending.append(following)
        if not advanced:
            raise ValueError('data pipeline deadlock')
    if terminals != 1:
        raise ValueError('terminal denominator changed')
    return len(seen)


def main():
    old = subprocess.check_output(['git', 'show', PARENT + ':' + SOURCE], cwd=ROOT, text=True)
    new = (ROOT / SOURCE).read_text()
    bind_move(old, new)
    for file in (ROOT / 'csrc/backends/sm90').rglob('*'):
        if file.is_file() and str(file.relative_to(ROOT)) != SOURCE:
            original = subprocess.check_output(['git', 'show', PARENT + ':' + str(file.relative_to(ROOT))], cwd=ROOT)
            if file.read_bytes() != original:
                raise ValueError('changed unrelated source: ' + str(file))
    for plant in (old, new.replace('qp.consumer_release(qr); ++qr;', '', 1),
                  new.replace('kkp.consumer_wait(kkr);', '', 1)):
        try:
            bind_move(old, plant)
        except ValueError:
            continue
        raise AssertionError('source negative escaped')
    states = {str(n): progress(n) for n in range(1, 9)}
    try:
        progress(3, omit_q_release=True)
    except ValueError:
        pass
    else:
        raise AssertionError('missing release deadlock negative escaped')
    print(json.dumps(dict(status='PASS', scope='SOURCE_BOUND_DATA_PROGRESS_NOT_MEMORY_ORDER',
        states_per_chunks=states, negatives=4, source_sha256=hashlib.sha256(new.encode()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
