#!/usr/bin/env python3
"""Exact C64/D128 SASS publication order; static sites, not executed counts.

0x29060/70/80 are the actual QK-full/QK-empty/KK-full mbarriers in this
fixed SharedStorage. A changed storage ABI requires re-derivation, not a
looser match. Both auxiliary full/final bodies must satisfy the order in
each of the four gate/initial-state device specializations.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def bodies(path):
    text = path.read_text()
    markers = list(re.finditer(r"\.type\s+(\S+),@function", text))
    result = {}
    for i, marker in enumerate(markers):
        end = markers[i+1].start() if i+1 < len(markers) else len(text)
        lines = text[text.index(marker[1]+":", marker.end()):end]
        result[marker[1]] = re.findall(
            r"/\*([0-9a-f]+)\*/\s+(?:@!?\w+\s+)?([A-Z][\w.]*)\s*(.*);", lines)
    assert len(result) == 4, "require all four real specializations"
    return result


def inspect(rows):
    events = []
    retries = []
    for i, (pc, op, args) in enumerate(rows):
        if op == "SYNCS.ARRIVE.TRANS64.A1T0" and "+0x29080]" in args:
            events.append((pc, "KK-ready"))
        elif op == "SYNCS.PHASECHK.TRANS64.TRYWAIT" and "+0x29070]" in args:
            if rows[i+1][1] == "NANOSLEEP.SYNCS":
                # nvcc outlines each blocking retry after the role bodies.
                # Retain/count it; it is not another independent acquire.
                assert rows[i+2][1] == "SYNCS.PHASECHK.TRANS64"
                assert "+0x29070]" in rows[i+2][2]
                retries.append(pc)
            else:
                events.append((pc, "QK-empty"))
        elif op == "SYNCS.ARRIVE.TRANS64.A1T0" and "+0x29060]" in args:
            events.append((pc, "QK-ready"))
    assert [e[1] for e in events] == ["KK-ready", "QK-empty", "QK-ready"]*2, events
    assert len(retries) == 2, retries
    return dict(events=events, outlined_retry_pcs=retries)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate", type=Path); p.add_argument("parent", type=Path)
    args = p.parse_args()
    cand, parent = bodies(args.candidate), bodies(args.parent)
    assert cand.keys() == parent.keys()
    result = {}
    for symbol, rows in cand.items():
        events = inspect(rows)
        try:
            inspect(parent[symbol])
        except AssertionError:
            pass
        else:
            raise AssertionError("old ordering negative unexpectedly passed")
        # No removed matrix work, named/data barrier or asynchronous completion.
        families = ("HGMMA", "HMMA", "LDSM", "STSM", "SYNCS", "WARPGROUP", "BAR")
        counts = lambda rs: Counter(op for _, op, _ in rs if op.startswith(families))
        assert counts(rows) == counts(parent[symbol]), "matrix/lifetime op change"
        result[symbol] = dict(events=events, preserved_counts=dict(counts(rows)))
    print(json.dumps(dict(status="PASS", old_native_negative="EXPECTED_RED",
        scope="STATIC-PUBLICATION-ORDER-NOT-SPEED", bodies=result,
        sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest()), indent=2))


if __name__ == "__main__":
    main()
