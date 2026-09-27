#!/usr/bin/env python3
"""Bind S56's real CuTe visitor to CUDA12.8's native warp predicates.

This is not a cycle model: predication suppresses EX2 evaluation but does not
prove that issue slots or instruction fetch disappear. Tail correctness is
also covered by the exhaustive actual-layout and device gates.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
from check_sm90_binary import inspect

ROOT = Path(__file__).resolve().parents[2]
PARENT = '170f34f'


def source_check():
    changed = subprocess.check_output(
        ['git', 'diff', PARENT, '--name-only', '--', 'csrc'], cwd=ROOT, text=True).splitlines()
    if set(changed) - {'csrc/backends/sm90/scalar_gdn_aux.cuh',
                       'csrc/backends/sm90/aux_causal_sectors.cuh'}:
        raise ValueError('unregistered kernel change')
    path = 'csrc/backends/sm90/scalar_gdn_aux.cuh'
    old = subprocess.check_output(['git', 'show', f'{PARENT}:{path}'], cwd=ROOT, text=True)
    new = (ROOT / path).read_text()
    arithmetic = old.split('                auto [row, col] = coords(i);', 1)[1].split('\n            }', 1)[0]
    if new.count(arithmetic) != 1:
        raise ValueError('live arithmetic/mask/rounding differs from S50')
    helper = (ROOT / 'csrc/backends/sm90/aux_causal_sectors.cuh').read_text()
    if 'if(warp>=column_band)' not in helper or 'int warp=local_tid/32;' not in helper:
        raise ValueError('causal owner condition changed')


def bodies(sass):
    inspect(sass)
    parts = re.split(r'Function\s*:\s*(\S+)', sass)
    return dict(zip(parts[1::2], parts[2::2]))


def instructions(body):
    return [(int(m[1], 16), m[2], m[3], m[4].strip()) for m in re.finditer(
        r'/\*([0-9a-f]+)\*/\s+(?:(@!?P\d+)\s+)?(\w+(?:\.\w+)*)\s*([^;]*);', body)]


def check(old, new):
    control, subject = bodies(old), bodies(new)
    if control.keys() != subject.keys() or len(subject) != 4:
        raise ValueError('actual specialization denominator changed')
    result = {}
    for name, body in subject.items():
        rows = instructions(body)
        def math_protocol(rs):
            return Counter(op for _, _, op, _ in rs if op.startswith(
                ('HGMMA', 'HMMA', 'WARPGROUP', 'UTMA', 'SYNCS', 'BAR.')))
        if math_protocol(rows) != math_protocol(instructions(control[name])):
            raise ValueError('matrix/TMA/data/issue protocol changed')
        # Track native definitions, not a fixed physical register number.
        regs, predicates, matched, skipped_tail = {}, {}, Counter(), 0
        for pc, guard, op, args in rows:
            words = args.split(', ')
            if op == 'MUFU.EX2' and guard:
                definition = predicates.get(guard.lstrip('@!'))
                if definition:
                    sector, positive = definition
                    if (not guard.startswith('@!')) == positive:
                        matched[sector] += 1
            if op == 'BRA' and guard:
                definition = predicates.get(guard.lstrip('@!'))
                if definition and definition[0] == 1 and guard.startswith('@!'):
                    stop = int(args, 16)
                    skipped_tail += sum(x[2] == 'MUFU.EX2' for x in rows if pc < x[0] < stop)
            # Any predicate destination invalidates the preceding definition.
            if re.fullmatch(r'P\d+', words[0]):
                predicates.pop(words[0], None)
                if op == 'ISETP.NE.AND' and words[1:] == ['PT', next(
                        (r for r, kind in regs.items() if kind == 'local_warp'), ''), 'RZ', 'PT']:
                    predicates[words[0]] = (1, True)
                elif op == 'ISETP.GT.U32.AND' and words[1:] == ['PT', next(
                        (r for r, kind in regs.items() if kind == 'local_tid'), ''), '0x3f', 'PT']:
                    predicates[words[0]] = (2, True)
                elif op == 'ISETP.NE.AND' and words[1:] == ['PT', next(
                        (r for r, kind in regs.items() if kind == 'local_warp'), ''), '0x3', 'PT']:
                    predicates[words[0]] = (3, False)
            if re.fullmatch(r'R\d+', words[0]):
                kind = None
                if op == 'S2R' and words[1:] == ['SR_TID.X']:
                    kind = 'cta_tid'
                elif op == 'LOP3.LUT' and len(words) == 6 and regs.get(words[1]) == 'cta_tid' and words[2:] == ['0x7f', 'RZ', '0xc0', '!PT']:
                    kind = 'local_tid'
                elif op == 'SHF.R.U32.HI' and words[1:3] == ['RZ', '0x5'] and regs.get(words[3]) == 'local_tid':
                    kind = 'local_warp'
                regs.pop(words[0], None)
                if kind and guard is None:
                    regs[words[0]] = kind
        if matched != Counter({1: 8, 2: 8, 3: 6}) or skipped_tail != 8:
            raise ValueError(f'actual warp-dependent EX2 suppression absent: {matched}, tail={skipped_tail}')
        result[name] = dict(full_chunk_predicated_EX2_by_sector=dict(matched),
                            tail_first_sector_bypassed_EX2=skipped_tail,
                            fixed_math_protocol=dict(math_protocol(rows)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--device-log', type=Path, required=True)
    a = p.parse_args()
    source_check()
    if 'C7512' in a.device_log.read_text():
        raise ValueError('WGMMA was serialized')
    old, new = a.parent.read_text(), a.candidate.read_text()
    result = check(old, new)
    plants = [old, new.replace('@P1 MUFU.EX2', '    MUFU.EX2', 1),
              new.replace('0x3f, PT', '0x1f, PT', 1),
              new.replace('WARPGROUP.DEPBAR.LE', 'REMOVED.DEPBAR', 1)]
    for plant in plants:
        try:
            check(old, plant)
        except ValueError:
            continue
        raise AssertionError('stale/unguarded/wrong-sector/missing-retirement negative escaped')
    print(json.dumps(dict(bodies=result, negative_controls=len(plants),
        scope='CUDA native evaluation suppression; NOT issue-slot/latency saving'), indent=2))


if __name__ == '__main__':
    main()
