#!/usr/bin/env python3
"""Bind staged epilogue experiment to actual native bodies; not a speed gate."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from check_sm90_aux_final_mask import bodies, intervals


def compare(candidate,parent):
    assert len(candidate)==4 and candidate.keys()==parent.keys()
    fixed=("HGMMA","HMMA","LDSM","STSM","WARPGROUP","SYNCS","BAR","UTMA",
           "MUFU","FMUL","FFMA","FADD","HADD2")
    count=lambda rows: Counter(op for _,op,_ in rows if op.startswith(fixed))
    result={}
    for symbol,rows in candidate.items():
        assert count(rows)==count(parent[symbol]), "matrix/arithmetic/protocol changed"
        current,old=intervals(rows),intervals(parent[symbol])
        assert any(a["counts"]!=b["counts"] for a,b in zip(current,old)), "inert native change"
        result[symbol]=dict(candidate=current,parent=old)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate",type=Path);p.add_argument("parent",type=Path);a=p.parse_args()
    c,b=bodies(a.candidate),bodies(a.parent);result=compare(c,b)
    try: compare(b,b)
    except AssertionError: pass
    else: raise AssertionError("old native negative escaped")
    print(json.dumps(dict(status="PASS",scope="STATIC_NOT_SPEED_OR_NUMERICS",bodies=result,
        old_native_negative="EXPECTED_RED",sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest()),indent=2))


if __name__=="__main__":main()
