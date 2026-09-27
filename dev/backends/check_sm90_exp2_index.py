#!/usr/bin/env python3
"""Bind four actual EX2 jump tables per SM90 body, including full chunks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
from check_sm90_exp2_batch import bodies


def tables(elf):
    return {m[1]: [int(x,16) for x in re.findall(r'0x[0-9a-f]+',m[2])]
        for m in re.finditer(r'^\.nv.constant2\.(\S+)\n((?:0x[0-9a-f]+(?: 0x[0-9a-f]+)*\n)+)',elf,re.M)}


def native(candidate,parent,table):
    c,b=bodies(candidate),bodies(parent)
    if c.keys()!=b.keys() or c.keys()!=table.keys():
        raise ValueError('specialization or constant table denominator changed')
    result={}
    fixed=('HMMA','HGMMA','UTMA','SYNCS','BAR.','WARPGROUP')
    def protocol(rows):
        return Counter(op for _,r in rows
            for op in [re.sub(r'^@!?\w+\s+','',r).split()[0]] if op.startswith(fixed))
    for name,rows in c.items():
        if protocol(rows)!=protocol(b[name]):raise ValueError('matrix/data protocol changed')
        index={pc:i for i,(pc,_) in enumerate(rows)}
        # The final inverse dispatch has its own table; only four EX2 tuple
        # tables are ours. Verify the target bytes, not merely BRX count.
        brx=[i for i,(_,r) in enumerate(rows) if r.startswith('BRX ')]
        if len(brx)!=5 or len(table[name])!=12:
            raise ValueError('four tuples plus inherited inverse dispatch not present')
        spans=[]
        for slot,i in enumerate(brx[:4]):
            pc,row=rows[i]
            slow_pc,fast_pc=table[name][slot*2:slot*2+2]
            if slow_pc!=pc+16 or fast_pc not in index:
                raise ValueError('tuple jump-table target mismatch')
            branch=re.fullmatch(r'BRX (R\d+) -0x([0-9a-f]+)',row)
            if not branch or int(branch[2],16)!=pc+16:
                raise ValueError('BRX does not resolve to table-relative absolute PCs')
            reg=branch[1];offset='' if slot==0 else r'\+0x'+format(8*slot,'x')
            # ptxas hoists later constant-table loads across the preceding
            # tuple's common arithmetic. Follow the actual last definition;
            # requiring adjacency would incorrectly reject this native CFG.
            definitions=[(p,r) for p,r in rows[:i]
                if re.match(r'^(?:@!?\w+\s+)?[\w.]+ '+reg+r'(?:,| )',r)]
            if not definitions or not re.fullmatch(
                    r'LDC '+reg+r', c\[0x2\]\[R\d+'+offset+r'\]',definitions[-1][1]):
                raise ValueError('wrong table offset or index delivery')
            j=index[fast_pc]
            merge=re.fullmatch(r'BRA (0x[0-9a-f]+)',rows[j-1][1])
            if not merge or int(merge[1],16) not in index:
                raise ValueError('fallback does not bypass the direct branch')
            k=index[int(merge[1],16)]
            slow=[r for _,r in rows[i+1:j-1]];fast=[r for _,r in rows[j:k]]
            n=sum(r.startswith('MUFU.EX2') for r in fast)
            # Fourteen values only in each diagonal-containing final tuple: two
            # exact zero differences are folded; their final values are one.
            expected=14 if slot%2==1 else 16
            if n!=expected or sum('MUFU.EX2' in r for r in slow)!=n:
                raise ValueError('fast/fallback exponent denominator changed')
            if sum('-126' in r for r in slow)!=n:
                raise ValueError('fallback underflow comparisons missing')
            if any('-126' in r or r.startswith(('FMUL','FSEL','LDL','STL','BRA','BRX')) for r in fast):
                raise ValueError('fast tuple still executes correction or local memory')
            if sum('FMUL' in r and '0.5' in r for r in slow)!=n:
                raise ValueError('fallback half-input correction missing')
            spans.append(dict(index=slot,branch=hex(pc),slow=hex(slow_pc),
                              fast=hex(fast_pc),join=merge[1],values=n,
                              fast_sites=len(fast),slow_sites=len(slow)))
        result[name]=spans
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--device-log',type=Path,required=True)
    p.add_argument('--cuobjdump',default='/usr/local/cuda-12.8/bin/cuobjdump')
    a=p.parse_args()
    log = a.device_log.read_text()
    stacks = list(map(int,re.findall(r'(\d+) bytes stack frame',log)))
    if 'C7512' in log or 'C7510' in log:raise ValueError('WGMMA serialization')
    if len(stacks)!=4 or max(stacks)>104:raise ValueError('stack ceiling / body inventory')
    obj=a.candidate.parent.parent/'launch.o'
    elf=subprocess.check_output([a.cuobjdump,'--dump-elf',str(obj)],text=True)
    (a.candidate.parent/'elf.txt').write_text(elf)
    c,b=a.candidate.read_text(),a.parent.read_text();t=tables(elf)
    got=native(c,b,t)
    swapped={k:list(v) for k,v in t.items()};key=next(iter(swapped))
    swapped[key][0],swapped[key][1]=swapped[key][1],swapped[key][0]
    fast=got[key][0]['fast'][2:]
    missing_fast,n=re.subn(r'(/\*0*'+fast+r'\*/\s*)MUFU.EX2',r'\1REMOVED.EX2',c,count=1)
    if n!=1:raise AssertionError('fast-path negative not planted')
    bad_offset,n=re.subn(r'(LDC R\d+, c\[0x2\]\[R\d+)\+0x8\]',r'\1+0x10]',c,count=1)
    if bad_offset==c:raise AssertionError('table offset negative not planted')
    for bad,tab in ((b,t),(c.replace('MUFU.EX2','REMOVED.EX2',1),t),
                    (missing_fast,t),(c,swapped),(bad_offset,t)):
        try:native(bad,b,tab)
        except ValueError:continue
        raise AssertionError('native negative escaped')
    print(json.dumps(dict(status='PASS',scope='CUDA_NATIVE_NOT_SPEED',
        negatives=['old body','missing fallback exponent','missing fast exponent',
                   'swapped branch targets','wrong table offset'],bodies=got),indent=2))


if __name__=='__main__':main()
