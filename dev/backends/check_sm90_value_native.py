#!/usr/bin/env python3
"""Check real V64 image roles and matrix retirement, not a speed predictor."""
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
        gate=[g for g in ('float','cutlass::bfloat16_t') if f'(kda::sm90::kernel::Tag)11, {g}>' in name]
        initial=[i for i in ('false','true') if f'(kda::sm90::kernel::Tag)8, cute::C<{i}>' in name]
        assert len(gate)==len(initial)==1
        key=(gate[0],initial[0]);assert key not in result
        result[key]=re.findall(r'/\*[0-9a-f]+\*/\s*(.*?)\s*;\s*/\*',section)
    assert len(result)==4
    return result

def count(rows):
    return Counter(re.match(r'(?:@!?\w+\s+)?([A-Z][\w.]*)',r)[1] for r in rows)

def check(candidate,parent,aux,loader=24):
    assert len(candidate)==len(parent)==4 and candidate.keys()==parent.keys()
    output={}
    for key,rows in candidate.items():
        a,b=count(rows),count(parent[key])
        for prefix in ('HGMMA','HMMA','WARPGROUP'):
            assert {o:n for o,n in a.items() if o.startswith(prefix)}=={o:n for o,n in b.items() if o.startswith(prefix)},(key,prefix)
        assert all('gsb0, 0x0' in r for r in rows if 'WARPGROUP.DEPBAR' in r)
        regs=[r for r in rows if r.startswith('USETMAXREG.')]
        expected=['USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xc0',
                  f'USETMAXREG.DEALLOC.CTAPOOL {hex(loader)}',
                  ('USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xe8' if aux==232 else
                   'USETMAXREG.DEALLOC.CTAPOOL 0x68')]
        assert sorted(regs)==sorted(expected),(key,regs)
        assert sum(v for k,v in a.items() if k.startswith('UTMA'))==14,'wrong V64 delivery body'
        if loader==32:
            assert aux==232
            assert not any(k.startswith(('LDL','STL')) for k in a),'loader spill remains'
            fixed=('LDSM','STSM','BAR','SYNCS','UTMA')
            assert {o:n for o,n in a.items() if o.startswith(fixed)}=={o:n for o,n in b.items() if o.startswith(fixed)}
        output[str(key)]=dict(sites=len(rows),register_transitions=regs,
            native_matrix_sites=sum(v for k,v in a.items() if k.startswith(('HGMMA','HMMA'))))
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path);p.add_argument('parent',type=Path)
    p.add_argument('--aux',type=int,choices=(104,232),required=True)
    p.add_argument('--loader',type=int,choices=(24,32),default=24)
    args=p.parse_args();c,p=bodies(args.candidate),bodies(args.parent)
    result=check(c,p,args.aux,args.loader)
    def negative(plant):
        try:check(plant,p,args.aux,args.loader)
        except AssertionError:return
        raise AssertionError('negative escaped')
    negative(p);negative(dict(list(c.items())[1:]))
    key=next(iter(c));rows=list(c[key])
    victim=next(i for i,r in enumerate(rows) if 'WARPGROUP.DEPBAR' in r)
    rows[victim]=rows[victim].replace('gsb0, 0x0','gsb0, 0x1')
    negative(c|{key:rows})
    print(json.dumps(dict(status='PASS',scope='STATIC_NOT_DYNAMIC_WORK_OR_SPEED',bodies=result,
        negatives=3,hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (args.candidate,args.parent)}),indent=2))

if __name__=='__main__':main()
