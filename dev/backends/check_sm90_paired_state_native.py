#!/usr/bin/env python3
"""Check actual paired register-source WGMMA epochs in all four S49 bodies.

Native sequence/retirement evidence, not a speed claim or a CUDA race proof.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


def bodies(path):
    result={}
    for section in path.read_text().split('Function :')[1:]:
        symbol=section.splitlines()[0].strip()
        if 'FlatKernelTmaWarpSpecializedKdaFwd' not in symbol:continue
        name=subprocess.check_output(['c++filt',symbol],text=True)
        gates=[g for g in ('float','cutlass::bfloat16_t') if f'(kda::sm90::kernel::Tag)11, {g}>' in name]
        initial=[i for i in ('false','true') if f'(kda::sm90::kernel::Tag)8, cute::C<{i}>' in name]
        if len(gates)!=1 or len(initial)!=1:raise ValueError('unbound kernel type')
        key=(gates[0],initial[0])
        if key in result:raise ValueError('duplicate kernel type')
        result[key]=re.findall(r'/\*[0-9a-f]+\*/\s*(.*?)\s*;\s*/\*',section)
    return result


def count(rows):
    return Counter(re.match(r'(?:@!?\w+\s+)?([A-Z][\w.]*)',r)[1] for r in rows)


def rs_epochs(rows):
    epochs=[];current=[]
    for index,row in enumerate(rows):
        match=re.match(r'HGMMA\.64x64x16\.F32\.BF16 (R\d+), (R\d+), gdesc\[',row)
        if match:current.append((index,match[1],match[2]))
        if 'WARPGROUP.DEPBAR' in row:
            if current:epochs.append(current);current=[]
            if 'gsb0, 0x0' not in row:raise ValueError('changed required full retirement')
    if current:raise ValueError('unretired register-source epoch')
    return epochs


def check(candidate,parent,log):
    if 'C7512' in log or 'C7510' in log:raise ValueError('serialized WGMMA')
    if len(candidate)!=4 or candidate.keys()!=parent.keys():raise ValueError('kernel denominator changed')
    results={}
    for key,rows in candidate.items():
        old=parent[key];a,b=count(rows),count(old)
        fixed=('HGMMA','HMMA','UTMA')
        if {o:n for o,n in a.items() if o.startswith(fixed)}!={o:n for o,n in b.items() if o.startswith(fixed)}:
            raise ValueError('matrix or TMA work changed')
        oldpairs=[e for e in rs_epochs(old) if len(e)==8]
        pairs=[e for e in rs_epochs(rows) if len(e)==16]
        expected=2 if key[1]=='false' else 4
        if len(oldpairs)!=2*expected or len(pairs)!=expected:
            raise ValueError('old split or new paired epoch count wrong')
        if b['WARPGROUP.DEPBAR.LE']-a['WARPGROUP.DEPBAR.LE']!=expected:
            raise ValueError('retirement delta wrong')
        for pair in pairs:
            first,second=pair[:8],pair[8:]
            if len({r[1] for r in first})!=1 or len({r[1] for r in second})!=1 or first[0][1]==second[0][1]:
                raise ValueError('O1/SK accumulator ownership not disjoint')
            if [r[2] for r in first]!=[r[2] for r in second]:
                raise ValueError('paired native H operands differ')
        regs=[r for r in rows if r.startswith('USETMAXREG.')]
        if regs!=[r for r in old if r.startswith('USETMAXREG.')]:raise ValueError('unregistered role-budget change')
        results[str(key)]=dict(sites=len(rows),parent_sites=len(old),paired_epochs=expected,
            retirements=a['WARPGROUP.DEPBAR.LE'],parent_retirements=b['WARPGROUP.DEPBAR.LE'],
            local_opcodes={o:n for o,n in a.items() if o.startswith(('LDL','STL'))},
            bf16_conversion_sites={o:(b.get(o,0),n) for o,n in a.items() if 'BF16' in o and o.startswith('F2')})
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate',type=Path);parser.add_argument('parent',type=Path)
    parser.add_argument('--device-log',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();c,p=bodies(args.candidate),bodies(args.parent);log=args.device_log.read_text()
    result=check(c,p,log)
    def red(image,text=log):
        try:check(image,p,text)
        except ValueError:return
        raise AssertionError('native paired negative escaped')
    red(p);red(dict(list(c.items())[1:]));red(c,log+' C7512')
    key=next(iter(c));rows=list(c[key]);pair=next(e for e in rs_epochs(rows) if len(e)==16)
    i=pair[8][0];rows.insert(i,'WARPGROUP.DEPBAR.LE gsb0, 0x0');red(c|{key:rows})
    rows=list(c[key]);rows[i]=rows[i].replace(', '+pair[8][2]+',',', R0,',1);red(c|{key:rows})
    rows=list(c[key]);rows.pop(pair[0][0]);red(c|{key:rows})
    output=json.dumps(dict(status='PASS_NATIVE_NOT_SPEED',bodies=result,negatives=6,
        hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (args.candidate,args.parent,args.device_log)}),indent=2)+'\n'
    if args.out:args.out.write_text(output)
    print(output,end='')


if __name__=='__main__':main()
