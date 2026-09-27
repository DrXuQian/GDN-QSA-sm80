#!/usr/bin/env python3
"""S68 source/lifetime and actual native admission; not a speed verdict."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import subprocess

from check_sm90_value_native import bodies, count

ROOT = Path(__file__).resolve().parents[2]
STATE = 'csrc/backends/sm90/scalar_gdn_state.cuh'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_check(source):
    body = source[source.index('auto body ='):]
    ordered = ['auto acc_sk =', 'auto acc_delta =', 'auto operand_delta =',
               'auto acc_o =', 'op.producer_commit(ow)',
               'auto h = partition_fragment_C', 'state_park_copy<false>',
               'gemm(kv_mma', 'warpgroup_fence_operand(h);\n            if constexpr (!last)',
               'state_park_copy<true>', 'publish_h_operand(h);', '\n            kp.consumer_release']
    positions = [body.index(token) for token in ordered]
    require(positions == sorted(positions), 'state live/publication order changed')
    require('if constexpr (first) clear(h);\n            else state_park_copy<false>' in body,
            'zero-state first chunk may read uninitialized scratch')
    publish = source[source.index('auto publish_h_operand ='):source.index('if constexpr (Base::kInitStateFromInput)')]
    require('copy(h,rounded_h)' in publish and 'copy(store_h,' in publish,
            'rounded H operand publication missing')
    require(publish.index('copy(store_h,') < publish.index('fence_view_async_shared') <
            publish.index('arrive_and_wait(128,Barriers::StateMathWG0)'),
            'SS producer fence or participants changed')
    initial = source[source.index('if constexpr (Base::kInitStateFromInput)'):source.index('auto state_product =')]
    require('state_park_copy<true>' in initial and 'publish_h_operand(h)' in initial,
            'initial state not published')
    # The state math expressions, precision and matrix operands remain those
    # of the late-O1 parent. Only H placement/lifetime/publication changes.
    old = subprocess.check_output(['git', 'show', '3eb5d75:' + STATE], cwd=ROOT, text=True)
    for pattern in (r'(?m)^\s*(?:gemm|gemm_zero_acc)\([^;]+;',
                    r'residual\(i\) = [^;]+;', r'acc_o\(i\) \*= [^;]+;',
                    r'h\(i\) \*= [^;]+;', r'operand_delta\(i\) = [^;]+;'):
        require(re.findall(pattern, source) == re.findall(pattern, old),
                'parent arithmetic changed: ' + pattern)


def progress(chunks, missing_release=False):
    # Q/K/V loader, aux, state, output drainer. Metadata producers are
    # unchanged/disjoint; this proves bounded data-ring progress, not CUDA
    # memory order. O's drainer has no Q/K dependency. Native/data gates cover
    # retirement/fences separately.
    programs = [
        ['cQ', 'cK', 'cV'],
        ['wK', 'wQ', 'rK', 'rQ', 'aKK', 'aQK', 'cQK', 'cKK'],
        ['wK', 'wV', 'wKK', 'rV', 'rKK', 'wQ', 'rQ', 'wQK', 'rQK', 'aO', 'cO', 'rK'],
        ['wO', 'rO'],
    ]
    if missing_release:
        programs[2].remove('rQ')
    programs = [p * chunks for p in programs]
    caps = dict(Q=2, K=2, V=1, QK=2, KK=1, O=1)
    readers = dict(Q=(1, 2), K=(1, 2), V=(2,), QK=(2,), KK=(2,), O=(3,))
    producers = dict(Q=0, K=0, V=0, QK=1, KK=1, O=2)
    prefixes = []
    for program in programs:
        rows = [{}]
        for op in program:
            row = rows[-1].copy(); row[op] = row.get(op, 0) + 1; rows.append(row)
        prefixes.append(rows)
    def n(pc, a, op): return prefixes[a][pc[a]].get(op, 0)
    def ready(pc, actor, op):
        kind, pipe = op[0], op[1:]
        if kind == 'w': return n(pc, producers[pipe], 'c'+pipe) > n(pc, actor, op)
        if kind == 'r': return n(pc, actor, 'w'+pipe) > n(pc, actor, op)
        if kind == 'c' and actor in (1, 2): return n(pc, actor, 'a'+pipe) > n(pc, actor, op)
        return n(pc, actor, op) < min(n(pc, r, 'r'+pipe) for r in readers[pipe]) + caps[pipe]
    pending, seen, terminals = deque([(0,)*4]), {(0,)*4}, 0
    while pending:
        pc = pending.popleft()
        if all(pc[a] == len(p) for a, p in enumerate(programs)):
            terminals += 1; continue
        advanced = False
        for a, program in enumerate(programs):
            if pc[a] < len(program) and ready(pc, a, program[pc[a]]):
                advanced = True; nxt = list(pc); nxt[a] += 1; nxt = tuple(nxt)
                if nxt not in seen: seen.add(nxt); pending.append(nxt)
        require(advanced, 'data-ring deadlock')
    require(terminals == 1, 'terminal count changed')
    return len(seen)


def native_check(image, log):
    require(not re.search(r'C751[02]', log), 'compiler serialized async groups')
    stacks = list(map(int, re.findall(r'(\d+) bytes stack frame', log)))
    require(len(stacks) == 4 and max(stacks) <= 104, 'stack bound / body denominator')
    output = {}
    for key, rows in image.items():
        counts = count(rows)
        initial = key[1] == 'true'
        matrix = sum(n for op, n in counts.items() if op.startswith('HGMMA'))
        waits = counts['WARPGROUP.DEPBAR.LE']
        require(matrix == (256 if initial else 192), 'matrix work/body denominator')
        require(waits == (22 if initial else 18), 'per-MMA serialization / missing wait')
        require(counts['LDS.128'] >= 64 and counts['STS.128'] >= 64, 'FP32 park not emitted')
        require(all('gsb0, 0x0' in r for r in rows if 'WARPGROUP.DEPBAR' in r), 'retirement changed')
        expected = {'USETMAXREG.DEALLOC.CTAPOOL 0x18',
                    'USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xf8',
                    'USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xe8'}
        require({r for r in rows if r.startswith('USETMAXREG')} == expected, 'wrong role budgets')
        if not initial:
            require(not any(op.startswith(('LDL', 'STL')) for op in counts), 'zero-state spill')
        output[str(key)] = dict(sites=len(rows), hgmma=matrix, waits=waits,
                               park_loads=counts['LDS.128'], park_stores=counts['STS.128'])
    require(len(output) == 4, 'missing specialization')
    return output


def negative(call):
    try: call()
    except (ValueError, AssertionError): return
    raise AssertionError('negative escaped')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--device-log', type=Path, required=True)
    a = p.parse_args()
    source = (ROOT / STATE).read_text(); source_check(source)
    negative(lambda: source_check(source.replace('publish_h_operand(h);', '', 2)))
    negative(lambda: source_check(source.replace('if constexpr (first) clear(h);', 'clear(h);')))
    negative(lambda: source_check(subprocess.check_output(['git', 'show', '3eb5d75:'+STATE], cwd=ROOT, text=True)))
    states = {n: progress(n) for n in range(1, 9)}
    negative(lambda: progress(3, True))
    image, log = bodies(a.candidate), a.device_log.read_text()
    result = native_check(image, log)
    negative(lambda: native_check(image, log+'C7512'))
    negative(lambda: native_check(dict(list(image.items())[1:]), log))
    key = next(iter(image))
    negative(lambda: native_check(image | {key: image[key] + ['WARPGROUP.DEPBAR.LE gsb0, 0x0']}, log))
    print(json.dumps(dict(status='PASS', scope='SOURCE_LIFETIME_BOUNDED_PROGRESS_NATIVE_NOT_SPEED',
        bodies=result, progress_states=states, negatives=7,
        hashes={str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in (a.candidate, a.device_log, ROOT/STATE)}), indent=2))


if __name__ == '__main__': main()
