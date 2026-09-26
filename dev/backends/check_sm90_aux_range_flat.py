#!/usr/bin/env python3
"""Validate compact native guard, without claiming compare instructions vanish."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from check_sm90_aux_final_mask import bodies, intervals


def guarded_checks(rows):
    return [(pc,op,arg) for pc,op,arg in rows
            if op == "FSETP.LT.AND" and ", -126," in arg]


def compare(candidate,parent):
    assert len(candidate)==4 and candidate.keys()==parent.keys()
    fixed=("HGMMA","HMMA","LDSM","STSM","WARPGROUP","SYNCS","BAR","UTMA",
           "MUFU","FMUL","FFMA")
    count=lambda rows:Counter(op for _,op,_ in rows if op.startswith(fixed))
    result={}
    for symbol,rows in candidate.items():
        assert count(rows)==count(parent[symbol]), "matrix/exponent/protocol work changed"
        assert Counter(op for _,op,_ in rows)["MUFU.EX2"]==64
        checks=guarded_checks(rows)
        assert len(checks)==60
        assert all(arg.split(",")[-1].strip().startswith("!P") for _,_,arg in checks), "range conjunction lost"
        result[symbol]=dict(candidate=intervals(rows),parent=intervals(parent[symbol]),
                            whole_sites=len(rows),parent_sites=len(parent[symbol]),
                            gated_comparison_pcs=[pc for pc,_,_ in checks])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate",type=Path);p.add_argument("parent",type=Path);a=p.parse_args()
    c,b=bodies(a.candidate),bodies(a.parent);result=compare(c,b)
    try:compare(b,b)
    except AssertionError:pass
    else:raise AssertionError("old native escaped")
    broken={k:list(v) for k,v in c.items()};symbol=next(iter(broken))
    victim=next(i for i,(_,op,arg) in enumerate(broken[symbol]) if op=="FSETP.LT.AND" and ", -126," in arg)
    pc,op,arg=broken[symbol][victim];broken[symbol][victim]=(pc,op,arg.rsplit(",",1)[0]+", PT")
    try:compare(broken,b)
    except AssertionError:pass
    else:raise AssertionError("lost range predicate escaped")
    print(json.dumps(dict(status="PASS",scope="NATIVE_COMPACTNESS_NOT_SPEED",bodies=result,
        interpretation="Compiler folds the range into FSETP AND; per-element comparisons remain, not skipped",
        negatives=["old-native EXPECTED_RED","lost-range-conjunction EXPECTED_RED"],
        sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest()),indent=2))


if __name__=="__main__":main()
