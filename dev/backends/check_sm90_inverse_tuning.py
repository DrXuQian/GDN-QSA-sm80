#!/usr/bin/env python3
"""Compare exact SM90 inverse / register allocation seams, not executed work."""
import argparse
from collections import Counter
import json
from pathlib import Path
from check_sm90_aux_final_mask import bodies

FIXED = ("HGMMA", "HMMA", "LDSM", "STSM", "WARPGROUP", "SYNCS", "BAR")

def inverse_spans(rows):
    barriers = [i for i,(_,op,arg) in enumerate(rows)
                if op.startswith("BAR.SYNC") and "0xd, 0x80" in arg]
    assert len(barriers) == 14, "aux full+tail inverse denominator"
    return [rows[barriers[start]:barriers[start+6]+1] for start in (0,7)]

def compare(candidate, parent, mode):
    assert len(candidate) == 4 and candidate.keys() == parent.keys()
    fixed = lambda rows: Counter(op for _,op,_ in rows if op.startswith(FIXED))
    report = {}
    for symbol, rows in candidate.items():
        assert fixed(rows) == fixed(parent[symbol]), "matrix/copy/data-protocol changed"
        item = dict(whole_sites=len(rows), parent_sites=len(parent[symbol]))
        if mode == "coords":
            pairs = []
            for c,p in zip(inverse_spans(rows),inverse_spans(parent[symbol])):
                cc,pc = (Counter(op.split(".")[0] for _,op,_ in x) for x in (c,p))
                for op in ("FFMA","HMMA","LDSM","STSM","BAR","F2FP"):
                    assert cc[op] == pc[op], (symbol,op)
                assert cc["SHFL"] == 21 and pc["SHFL"] == 22, "canonical coordinate shuffle remains"
                assert len(c) < len(p), "coordinate simplification absent"
                pairs.append(dict(candidate_sites=len(c),parent_sites=len(p),
                                  candidate_counts=dict(cc),parent_counts=dict(pc)))
            item["inverse"] = pairs
        else:
            allocation = [(op,arg) for _,op,arg in rows if op.startswith("USETMAXREG")]
            assert sum("DEALLOC" in op and arg.strip()=="0x78" for op,arg in allocation) == 1
            assert sum("DEALLOC" in op and arg.strip()=="0x18" for op,arg in allocation) == 1
            assert sum("TRY_ALLOC" in op and arg.strip().endswith(", 0xb8") for op,arg in allocation) == 1
            item["register_allocation"] = allocation
            item["local_operations"] = {op:sum(o.startswith(op) for _,o,_ in rows) for op in ("LDL","STL")}
        report[symbol] = item
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("coords","registers"))
    p.add_argument("candidate",type=Path);p.add_argument("parent",type=Path)
    a=p.parse_args()
    c,b=bodies(a.candidate),bodies(a.parent)
    result=compare(c,b,a.mode)
    try: compare(b,b,a.mode)
    except AssertionError: pass
    else: raise AssertionError("old native escaped targeted postcondition")
    broken={k:list(v) for k,v in c.items()}; key=next(iter(broken))
    broken[key]=[r for r in broken[key] if not r[1].startswith("HMMA")]
    try: compare(broken,b,a.mode)
    except AssertionError: pass
    else: raise AssertionError("lost matrix work escaped")
    print(json.dumps(dict(status="PASS",scope="STATIC_NOT_SPEED",bodies=result,
        negatives=["old native EXPECTED_RED","lost HMMA EXPECTED_RED"]),indent=2))

if __name__=="__main__": main()
