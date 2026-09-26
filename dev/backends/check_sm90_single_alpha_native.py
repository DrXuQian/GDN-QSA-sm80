#!/usr/bin/env python3
"""Audit the actual removal of the unused scalar alpha-last channel."""
import argparse
from collections import Counter
import json
from pathlib import Path
from check_sm90_aux_final_mask import bodies


def compare(candidate,parent):
    assert len(candidate)==len(parent)==4 and candidate.keys()==parent.keys()
    result={}
    fixed=('HGMMA','HMMA','LDSM','STSM','WARPGROUP','UTMA','BAR','MUFU','FADD','FMUL','FFMA','HADD2','F2FP','USETMAXREG')
    for name,rows in candidate.items():
        c,p=Counter(o for _,o,_ in rows),Counter(o for _,o,_ in parent[name])
        assert Counter({k:v for k,v in c.items() if k.startswith(fixed)}) == \
               Counter({k:v for k,v in p.items() if k.startswith(fixed)}), 'matrix/math/copy/WGMMA protocol changed'
        # Four redundant scalar shared loads/stores and six wait sites:
        # producer self-consumption/duplicate publication and state variants.
        expected={'LDS':4,'STS':4,'SYNCS.PHASECHK.TRANS64':6,
                  'SYNCS.PHASECHK.TRANS64.TRYWAIT':12,
                  'SYNCS.ARRIVE.TRANS64.A1T0':1,
                  'SYNCS.ARRIVE.TRANS64.RED.A1T0':5,
                  'FENCE.VIEW.ASYNC.S':1,'MEMBAR.ALL.CTA':1}
        for op,removed in expected.items():
            assert p[op]-c[op]==removed,(op,p[op],c[op])
        assert c['SYNCS.EXCH.64'] == p['SYNCS.EXCH.64'], 'one-time initialization changed'
        result[name]=dict(sites=len(rows),parent_sites=len(parent[name]),
                          removed=expected,local_ops={k:[p[k],c[k]] for k in sorted(c.keys()|p.keys())
                            if k.startswith(('LDL','STL'))})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path);p.add_argument('parent',type=Path)
    a=p.parse_args();c,b=bodies(a.candidate),bodies(a.parent)
    result=compare(c,b)
    lost=dict(c);n=next(iter(lost))
    lost[n]=[r for r in lost[n] if not r[1].startswith('WARPGROUP.DEPBAR')]
    for name,plant in [('old-channel',b),('lost-completion',lost)]:
        try: compare(plant,b)
        except AssertionError: continue
        raise AssertionError(f'negative escaped: {name}')
    print(json.dumps(dict(status='PASS',scope='NATIVE_MECHANISM_NOT_SPEED',bodies=result,
                          negatives='2/2 EXPECTED_RED'),indent=2))


if __name__=='__main__':main()
