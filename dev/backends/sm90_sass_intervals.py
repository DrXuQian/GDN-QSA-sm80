#!/usr/bin/env python3
"""Count an explicitly selected native symbol/PC interval, NEVER executions.

Input is nvdisasm -c -g output. Cold out-of-line wait blocks, branch predicates,
and loop trips are not accounted by linear interval counts. No timing inference.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def inspect(text, symbol, begin, end):
    functions=list(re.finditer(r"\.type\s+(\S+),@function",text))
    selected=[(i,m) for i,m in enumerate(functions) if m[1]==symbol]
    if len(selected)!=1: raise ValueError("require one exact symbol, not a prefix")
    index,marker=selected[0]
    start=text.index(symbol+":",marker.end())
    finish=functions[index+1].start() if index+1<len(functions) else len(text)
    source=None; rows=[]
    for line in text[start:finish].splitlines():
        if match:=re.search(r'//## File "([^"]+)", line (\d+)',line):
            source=dict(file=match[1],line=int(match[2]))
        if match:=re.search(r"/\*([0-9a-f]+)\*/\s+(?:@!?\w+\s+)?([A-Z][\w.]*)\s*(.*);",line):
            rows.append(dict(pc=int(match[1],16),opcode=match[2],operands=match[3],source=source))
    pcs=[r['pc'] for r in rows]
    if len(set(pcs))!=len(pcs): raise ValueError("duplicate PC: mixed symbols or corrupt input")
    if begin not in pcs or end not in pcs or begin>end: raise ValueError("interval endpoints must be actual PCs")
    chosen=[r for r in rows if begin<=r['pc']<=end]
    counts=Counter(r['opcode'].split('.')[0] for r in chosen)
    return dict(symbol=symbol,begin=hex(begin),end=hex(end),static_instructions=len(chosen),
                counts=dict(sorted(counts.items())),local_memory=[r for r in chosen if r['opcode'].startswith(('LDL','STL'))],
                matrix=[r for r in chosen if r['opcode'].startswith(('HGMMA','HMMA'))],
                scope="STATIC-PC-INTERVAL-NOT-DYNAMIC-INSTRUCTION-COUNT")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sass",type=Path)
    parser.add_argument("--symbol",required=True)
    parser.add_argument("--begin",type=lambda s:int(s,0),required=True)
    parser.add_argument("--end",type=lambda s:int(s,0),required=True)
    args=parser.parse_args()
    raw=args.sass.read_bytes()
    result=inspect(raw.decode(),args.symbol,args.begin,args.end)
    result['sass_sha256']=hashlib.sha256(raw).hexdigest()
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
