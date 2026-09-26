#!/usr/bin/env python3
"""Bind the relative-decay cache to all four compiled SM90 function bodies.

The register-budget transitions identify the state role in this fixed shipping
specialization. Counts are static sites, not executed instructions or speed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from check_sm90_aux_final_mask import bodies


def state_interval(rows):
    alloc = [i for i, (_, op, arg) in enumerate(rows)
             if op == "USETMAXREG.TRY_ALLOC.CTAPOOL" and "0xc0" in arg]
    assert len(alloc) == 1, "state register-allocation boundary is ambiguous"
    begin = alloc[0]
    ends = [i for i, (_, op, arg) in enumerate(rows)
            if i > begin and op == "USETMAXREG.DEALLOC.CTAPOOL" and "0x18" in arg]
    assert len(ends) == 1, "producer register-allocation boundary is ambiguous"
    return begin, ends[0]


def inspect(rows):
    begin, end = state_interval(rows)
    counts = lambda rs: Counter(op for _, op, _ in rs)
    return dict(begin=rows[begin][0], end_exclusive=rows[end][0],
                sites=end-begin, state_ex2=counts(rows[begin:end])["MUFU.EX2"],
                outside_ex2=counts(rows[:begin] + rows[end:])["MUFU.EX2"])


def compare(candidate, parent):
    assert len(candidate) == 4 and candidate.keys() == parent.keys()
    fixed = ("HGMMA", "HMMA", "LDSM", "STSM", "WARPGROUP", "SYNCS", "BAR", "UTMA")
    counts = lambda rows: Counter(op for _, op, _ in rows if op.startswith(fixed))
    result = {}
    for symbol, rows in candidate.items():
        current, old = inspect(rows), inspect(parent[symbol])
        assert old["state_ex2"] == 96 and current["state_ex2"] == 0, (symbol, current, old)
        assert current["outside_ex2"] == old["outside_ex2"] + 2
        assert current["sites"] < old["sites"]
        assert counts(rows) == counts(parent[symbol]), "matrix/data-completion work changed"
        result[symbol] = dict(candidate=current, parent=old)
    return result


def expect_red(candidate, parent, label):
    try:
        compare(candidate, parent)
    except AssertionError:
        return
    raise AssertionError(label + " negative escaped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("parent", type=Path)
    args = parser.parse_args()
    current, parent = bodies(args.candidate), bodies(args.parent)
    result = compare(current, parent)
    expect_red(parent, parent, "old-native")
    symbol = next(iter(current))
    broken = dict(current)
    rows = list(broken[symbol])
    victim = next(i for i, (_, op, _) in enumerate(rows) if op.startswith("BAR.SYNC"))
    del rows[victim]
    broken[symbol] = rows
    expect_red(broken, parent, "lost-barrier")
    print(json.dumps(dict(status="PASS", scope="STATIC-NOT-SPEED", bodies=result,
        negatives={"old-native": "EXPECTED_RED", "lost-barrier": "EXPECTED_RED"},
        sha256={"candidate": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                "parent": hashlib.sha256(args.parent.read_bytes()).hexdigest()}), indent=2))


if __name__ == "__main__":
    main()
