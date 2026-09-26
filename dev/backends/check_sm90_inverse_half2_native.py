#!/usr/bin/env python3
"""Bind packed-half final inverse reduction in all four native bodies."""
import argparse
from collections import Counter
import json
from pathlib import Path
from check_sm90_aux_final_mask import bodies

PRESERVED = ("HGMMA", "HMMA", "LDSM", "STSM", "WARPGROUP", "SYNCS", "BAR")


def intervals(rows):
    barriers = [i for i, (_, op, arg) in enumerate(rows)
                if op.startswith("BAR.SYNC") and "0xd, 0x80" in arg]
    assert len(barriers) == 14, "full/tail inverse denominator"
    return [rows[barriers[s+5]:barriers[s+6]+1] for s in (0,7)]


def compare(candidate, parent):
    assert len(candidate) == len(parent) == 4 and candidate.keys() == parent.keys()
    result = {}
    for symbol, rows in candidate.items():
        work = lambda body: Counter(op for _, op, _ in body if op.startswith(PRESERVED))
        assert work(rows) == work(parent[symbol]), "matrix/copy/completion changed"
        parts = []
        for got, old in zip(intervals(rows), intervals(parent[symbol])):
            count = lambda part: Counter(op.split('.')[0] for _,op,_ in part)
            actual, previous = count(got), count(old)
            assert previous['HADD2'] == 16 and previous['PRMT'] == 8
            assert actual['HADD2'] == 8 and actual['PRMT'] == 0, "packed lowering absent"
            for _, op, arg in got:
                if op.startswith('HADD2'):
                    assert '.H0_H0' not in arg and '.H1_H1' not in arg, "scalar broadcast remains"
            parts.append(dict(candidate=dict(actual), parent=dict(previous),
                              candidate_sites=len(got), parent_sites=len(old)))
        result[symbol] = dict(reduction=parts, whole_sites=len(rows), parent_sites=len(parent[symbol]))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('parent',type=Path)
    args=parser.parse_args()
    candidate,parent=bodies(args.candidate),bodies(args.parent)
    result=compare(candidate,parent)
    removed=dict(candidate)
    first=next(iter(removed))
    removed[first]=[r for r in removed[first] if not r[1].startswith('HMMA')]
    for name,plant in [('old-scalar',parent),('lost-mma',removed)]:
        try: compare(plant,parent)
        except AssertionError: continue
        raise AssertionError(f'negative escaped: {name}')
    print(json.dumps(dict(status='PASS',scope='NATIVE_MECHANISM_NOT_SPEED',bodies=result,
                          negatives='2/2 EXPECTED_RED'),indent=2))


if __name__=='__main__': main()
