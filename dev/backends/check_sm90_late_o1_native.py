#!/usr/bin/env python3
"""S45 native admission. Increased spills remain visible, not a speed claim."""
import argparse
import hashlib
import json
from pathlib import Path
from check_sm90_value_native import bodies,count


def check(candidate,parent,log):
    assert 'C7512' not in log and 'C7510' not in log,'serialized WGMMA'
    assert len(candidate)==len(parent)==4 and candidate.keys()==parent.keys()
    result={}
    for key,rows in candidate.items():
        a,b=count(rows),count(parent[key]);prefix=('HGMMA','HMMA','WARPGROUP','UTMA')
        assert {o:n for o,n in a.items() if o.startswith(prefix)}=={o:n for o,n in b.items() if o.startswith(prefix)}
        assert all('gsb0, 0x0' in r for r in rows if 'WARPGROUP.DEPBAR' in r)
        regs=[r for r in rows if r.startswith('USETMAXREG.')]
        assert sorted(regs)==sorted(['USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xc0',
            'USETMAXREG.DEALLOC.CTAPOOL 0x18','USETMAXREG.DEALLOC.CTAPOOL 0x68'])
        result[str(key)]=dict(sites=len(rows),parent_sites=len(parent[key]),registers=regs,
            local_opcodes={o:n for o,n in a.items() if o.startswith(('LDL','STL'))})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path);p.add_argument('parent',type=Path)
    p.add_argument('--device-log',type=Path,required=True)
    a=p.parse_args();c,pb=bodies(a.candidate),bodies(a.parent);log=a.device_log.read_text()
    result=check(c,pb,log)
    for plant,text in ((dict(list(c.items())[1:]),log),(c,log+'C7512')):
        try:check(plant,pb,text)
        except AssertionError:pass
        else:raise AssertionError('native negative escaped')
    key=next(iter(c));rows=list(c[key]);idx=next(i for i,r in enumerate(rows) if 'WARPGROUP.DEPBAR' in r)
    rows.pop(idx)
    try:check(c|{key:rows},pb,log)
    except AssertionError:pass
    else:raise AssertionError('removed retirement escaped')
    print(json.dumps(dict(status='PASS_NATIVE_ONLY',bodies=result,negatives=3,
        hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (a.candidate,a.parent,a.device_log)}),indent=2))


if __name__=='__main__':main()
