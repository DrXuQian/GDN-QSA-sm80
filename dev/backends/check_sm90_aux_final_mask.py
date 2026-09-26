#!/usr/bin/env python3
"""Exact native pre-inverse intervals; does not estimate executed work/time."""
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
        end = markers[i+1].start() if i+1<len(markers) else len(text)
        body = text[text.index(marker[1]+":", marker.end()):end]
        result[marker[1]] = re.findall(
            r"/\*([0-9a-f]+)\*/\s+(?:@!?\w+\s+)?([A-Z][\w.]*)\s*(.*);", body)
    assert len(result) == 4
    return result


def intervals(rows):
    starts = [i for i, (_,op,_) in enumerate(rows) if op.startswith("HGMMA")]
    result = []
    for start in (starts[0], starts[16]):
        end = next(i for i in range(start,len(rows))
                   if rows[i][1].startswith("BAR.SYNC") and "0xd, 0x80" in rows[i][2])
        selected = rows[start:end+1]
        result.append(dict(begin=rows[start][0], end=rows[end][0],
            sites=len(selected), counts=dict(Counter(o.split(".")[0] for _,o,_ in selected))))
    return result


def compare(candidate, parent):
    assert candidate.keys() == parent.keys()
    result = {}
    for symbol, rows in candidate.items():
        pairs = list(zip(intervals(rows), intervals(parent[symbol])))
        for c,p in pairs:
            for op in ("HGMMA", "MUFU", "FSETP", "FMUL", "F2FP", "LDS", "STSM"):
                assert c["counts"][op] == p["counts"][op], (symbol,op)
            assert c["counts"]["FSEL"] < p["counts"]["FSEL"]
            assert c["sites"] < p["sites"]
        # Full-body comparisons fall48->30; tail rises63->64 due to different
        # predicate packing. The input FSEL disappears in BOTH. Do not turn
        # the observed full-body ISETP reduction into a false tail claim.
        assert sum(c["counts"]["ISETP"] for c,_ in pairs) < sum(p["counts"]["ISETP"] for _,p in pairs)
        fixed = ("HGMMA", "HMMA", "LDSM", "STSM", "WARPGROUP", "SYNCS", "BAR")
        count = lambda rs: Counter(o for _,o,_ in rs if o.startswith(fixed))
        assert count(rows) == count(parent[symbol])
        result[symbol] = pairs
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate", type=Path); p.add_argument("parent", type=Path)
    a = p.parse_args(); candidate, parent = bodies(a.candidate), bodies(a.parent)
    result = compare(candidate,parent)
    try:
        compare(parent,parent)
    except AssertionError:
        pass
    else:
        raise AssertionError("old native negative passed")
    print(json.dumps(dict(status="PASS", scope="STATIC-NOT-SPEED", bodies=result,
        old_native_negative="EXPECTED_RED", sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest()),indent=2))


if __name__ == "__main__":
    main()
