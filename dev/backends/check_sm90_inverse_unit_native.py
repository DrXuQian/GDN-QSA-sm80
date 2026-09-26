#!/usr/bin/env python3
"""Verify removal of inverse input selects in four actual native bodies.

Static intervals identify the mechanism, not dynamic instructions or speed.
Register allocation and all new work in the publication epilogue still matter.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from check_sm90_aux_final_mask import bodies

PRESERVED = ("HGMMA", "HMMA", "LDSM", "STSM", "WARPGROUP", "SYNCS", "BAR")


def parts(rows):
    barriers = [i for i, (_, op, arg) in enumerate(rows)
                if op.startswith("BAR.SYNC") and "0xd, 0x80" in arg]
    assert len(barriers) == 14, "full/tail inverse denominator"
    return [rows[barriers[start]:barriers[start + 1] + 1] for start in (0, 7)]


def compare(candidate, parent):
    assert len(candidate) == len(parent) == 4 and candidate.keys() == parent.keys()
    result = {}
    for symbol, rows in candidate.items():
        count_work = lambda body: Counter(op for _, op, _ in body if op.startswith(PRESERVED))
        assert count_work(rows) == count_work(parent[symbol]), "matrix/copy/protocol changed"
        intervals = []
        for got, old in zip(parts(rows), parts(parent[symbol])):
            counts = lambda part: Counter(op.split(".")[0] for _, op, _ in part)
            current, original = counts(got), counts(old)
            for op in ("SHFL", "FFMA", "LDS", "STS"):
                assert current[op] == original[op], (symbol, op)
            assert current["FSEL"] < original["FSEL"], "solver normalization remains"
            intervals.append(dict(candidate_sites=len(got), parent_sites=len(old),
                                  candidate_counts=dict(current), parent_counts=dict(original)))
        result[symbol] = dict(diagonal_intervals=intervals, whole_sites=len(rows),
                              parent_whole_sites=len(parent[symbol]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("parent", type=Path)
    args = parser.parse_args()
    candidate, parent = bodies(args.candidate), bodies(args.parent)
    result = compare(candidate, parent)
    removed = dict(candidate)
    symbol = next(iter(removed))
    removed[symbol] = [row for row in removed[symbol] if not row[1].startswith("HMMA")]
    for name, plant in (("old-native", parent), ("lost-matrix-work", removed)):
        try:
            compare(plant, parent)
        except AssertionError:
            continue
        raise AssertionError(f"negative escaped: {name}")
    print(json.dumps(dict(status="PASS", scope="NATIVE_MECHANISM_NOT_SPEED",
                          bodies=result, negatives="2/2 EXPECTED_RED"), indent=2))


if __name__ == "__main__":
    main()
