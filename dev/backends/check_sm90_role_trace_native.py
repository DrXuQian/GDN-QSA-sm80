#!/usr/bin/env python3
"""Check diagnostic stamps against actual full native device bodies.

This is not an instruction-count performance claim. Individual stamp ordering
against the inverse's barriers is additionally recorded in the experiment
report; arithmetic without a dependency may move across a timer instruction.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from check_sm90_aux_final_mask import bodies

PRESERVED = ("HGMMA", "HMMA", "LDSM", "STSM", "BAR", "SYNCS", "WARPGROUP")


def compare(candidate, parent):
    assert len(candidate) == len(parent) == 4, "specialization denominator"
    assert candidate.keys() == parent.keys(), "specialization identities"
    report = {}
    for symbol, rows in candidate.items():
        original = parent[symbol]
        counts = lambda stream: Counter(op for _, op, _ in stream if op.startswith(PRESERVED))
        assert counts(rows) == counts(original), "matrix/copy/sync work changed"
        assert not any("GLOBALTIMER" in arg for _, _, arg in original), "instrumented control"
        stamps = [(pc, op, arg) for pc, op, arg in rows if "GLOBALTIMER" in arg]
        # Two auxiliary full/tail bodies *9 points, four state clones *13.
        assert len(stamps) == 2 * 9 + 4 * 13, "static stamp denominator"
        assert all(op == "CS2R" and "SR_GLOBALTIMERLO" in arg for _, op, arg in stamps)
        report[symbol] = dict(stamp_sites=stamps, preserved_counts=dict(counts(rows)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("parent", type=Path)
    args = parser.parse_args()
    candidate, parent = bodies(args.candidate), bodies(args.parent)
    result = compare(candidate, parent)
    symbol = next(iter(candidate))
    missing_stamp = dict(candidate)
    removed = False
    kept = []
    for row in candidate[symbol]:
        if not removed and "GLOBALTIMER" in row[2]:
            removed = True
        else:
            kept.append(row)
    missing_stamp[symbol] = kept
    missing_barrier = dict(candidate)
    missing_barrier[symbol] = [row for row in candidate[symbol] if not row[1].startswith("BAR")]
    for name, plant in (("old-uninstrumented", parent), ("missing-stamp", missing_stamp),
                        ("missing-data-barrier", missing_barrier)):
        try:
            compare(plant, parent)
        except AssertionError:
            continue
        raise AssertionError(f"negative escaped: {name}")
    print(json.dumps(dict(status="PASS", scope="DIAGNOSTIC_NATIVE_NOT_SPEED",
                          bodies=result, negatives="3/3 EXPECTED_RED"), indent=2))


if __name__ == "__main__":
    main()
