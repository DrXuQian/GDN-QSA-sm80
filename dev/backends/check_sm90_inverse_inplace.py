#!/usr/bin/env python3
"""All four native bodies: diagonal updates stay predicated and in-place."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def bodies(path):
    text=path.read_text(); markers=list(re.finditer(r"\.type\s+(\S+),@function",text))
    result={}
    for i,m in enumerate(markers):
        end=markers[i+1].start() if i+1<len(markers) else len(text)
        body=text[text.index(m[1]+":",m.end()):end]
        result[m[1]]=re.findall(r"/\*([0-9a-f]+)\*/\s+(@!?\w+\s+)?([A-Z][\w.]*)\s*(.*);",body)
    assert len(result)==4
    return result


def inspect(rows, require_inplace):
    sh=[i for i,(_,_,o,a) in enumerate(rows) if o.startswith("SHFL") and "0x181f" in a]
    assert len(sh)==42, "two complete21-shuffle inverse bodies"
    result=[]
    for first in (0,21):
        end=next(i for i in range(sh[first+20],len(rows)) if rows[i][2].startswith("BAR.SYNC"))
        selected=rows[sh[first]:end]
        assert all(not pred for _,pred,o,_ in selected if o.startswith("SHFL")), "shuffle source lanes must participate"
        fmas=[(pred,a) for _,pred,o,a in selected if o=="FFMA"]
        assert len(fmas)==21 and all(pred.strip().startswith("@") for pred,_ in fmas)
        in_place=sum(a.split(",")[0].strip()==a.split(",")[-1].strip() for _,a in fmas)
        copies=sum(o=="MOV" or o.startswith("IMAD.MOV") for _,_,o,a in selected)
        if require_inplace:
            assert in_place==21 and copies==0,(in_place,copies)
        result.append(dict(begin=selected[0][0],end=selected[-1][0],sites=len(selected),
            in_place_fma=in_place,copies=copies,counts=dict(Counter(o.split(".")[0] for _,_,o,_ in selected))))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate",type=Path);p.add_argument("parent",type=Path);a=p.parse_args()
    c,b=bodies(a.candidate),bodies(a.parent);assert c.keys()==b.keys()
    result={}
    for symbol,rows in c.items():
        current,old=inspect(rows,True),inspect(b[symbol],False)
        # A lost predicate changes the algorithm even with identical counts.
        plant=list(rows)
        victim=next(i for i,(_,pred,o,_) in enumerate(plant) if o=="FFMA" and pred)
        pc,_,op,operand=plant[victim];plant[victim]=(pc,"",op,operand)
        try:inspect(plant,True)
        except AssertionError:pass
        else:raise AssertionError("unconditional update negative escaped")
        try:inspect(b[symbol],True)
        except AssertionError:pass
        else:raise AssertionError("old body negative escaped")
        assert all(x["sites"]<y["sites"] for x,y in zip(current,old))
        fixed=("HGMMA","HMMA","SHFL","LDSM","STSM","WARPGROUP","SYNCS","BAR")
        count=lambda rs:Counter(o for _,_,o,_ in rs if o.startswith(fixed))
        assert count(rows)==count(b[symbol])
        result[symbol]=dict(candidate=current,parent=old)
    print(json.dumps(dict(status="PASS",scope="STATIC-NOT-SPEED",bodies=result,
        old_native_negative="EXPECTED_RED",sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest()),indent=2))


if __name__=="__main__":main()
