#!/usr/bin/env python3
"""Actual native clone/cast/completion contract, not a runtime-work estimate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from check_sm90_aux_final_mask import bodies


def keyed(path):
    result = {}
    for symbol, rows in bodies(path).items():
        name = subprocess.check_output(["c++filt", symbol], text=True)
        gates = [g for g in ("float", "cutlass::bfloat16_t")
                 if f"(kda::sm90::kernel::Tag)11, {g}>" in name]
        initial = [i for i in ("true", "false")
                   if f"(kda::sm90::kernel::Tag)8, cute::C<{i}>" in name]
        assert len(gates) == len(initial) == 1
        key = (gates[0], initial[0])
        assert key not in result
        result[key] = rows
    assert len(result) == 4
    return result


def batches(rows):
    result, matrices, arrivals = [], 0, 0
    for _, opcode, operands in rows:
        if opcode.startswith("HGMMA"):
            matrices += 1
        elif opcode == "WARPGROUP.ARRIVE":
            arrivals += 1
        elif opcode.startswith("WARPGROUP.DEPBAR"):
            assert opcode == "WARPGROUP.DEPBAR.LE"
            assert operands.strip() == "gsb0, 0x0", "dependent batch lost wait-all"
            result.append((matrices, arrivals))
            matrices = arrivals = 0
    assert matrices == arrivals == 0, "unretired matrix batch"
    return result


def check(candidate, parent):
    assert candidate.keys() == parent.keys() and len(candidate) == 4
    results = {}
    for key, rows in candidate.items():
        initial = key[1] == "true"
        c, p = Counter(o for _, o, _ in rows), Counter(o for _, o, _ in parent[key])
        state = [(8, 1), (8, 1), (4, 1), (4, 1), (4, 1)]
        first = state if initial else state[2:]
        # Auxiliary has two static bodies with two products sharing wait0.
        assert batches(rows) == [(16, 2)] * 2 + first + state * 2
        assert batches(parent[key]) == [(16, 2)] * 2 + first * 2 + state * 2
        assert c['F2F.BF16.F32'] == 0 and p['F2F.BF16.F32'] == 128
        assert c['F2FP.BF16.F32.PACK_AB'] > 0
        for prefix in ('HMMA', 'UTMA', 'USETMAXREG', 'SYNCS.EXCH'):
            assert {o:n for o,n in c.items() if o.startswith(prefix)} == \
                   {o:n for o,n in p.items() if o.startswith(prefix)}, prefix
        results[str(key)] = dict(sites=len(rows), parent_sites=len(parent[key]),
                                matrix_batches=batches(rows), opcodes=dict(c))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('parent', type=Path)
    args = parser.parse_args()
    c, p = keyed(args.candidate), keyed(args.parent)
    result = check(c, p)
    key = next(iter(c))
    missing = dict(c)
    missing.pop(key)
    lost = dict(c)
    index = next(i for i, r in enumerate(c[key]) if r[1].startswith('WARPGROUP.DEPBAR'))
    lost[key] = c[key][:index] + c[key][index+1:]
    relaxed = dict(c)
    relaxed[key] = c[key][:index] + [(c[key][index][0], c[key][index][1], 'gsb0, 0x1')] + c[key][index+1:]
    for label, plant in [('old-clones-and-casts', p), ('missing-type', missing),
                         ('lost-completion', lost), ('relaxed-completion', relaxed)]:
        try:
            check(plant, p)
        except AssertionError:
            continue
        raise AssertionError(f'negative escaped: {label}')
    print(json.dumps(dict(status='PASS', scope='STATIC_WRITING_NOT_DYNAMIC_WORK_OR_SPEED',
                          bodies=result, negatives='4/4 EXPECTED_RED'), indent=2))


if __name__ == '__main__':
    main()
