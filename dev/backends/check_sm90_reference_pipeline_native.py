#!/usr/bin/env python3
"""Compare all real gate/initial-state bodies despite added option tags."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from check_sm90_aux_final_mask import bodies


def keyed(path):
    result={}
    for symbol,rows in bodies(path).items():
        name=subprocess.check_output(['c++filt',symbol],text=True)
        gates=[g for g in ('float','cutlass::bfloat16_t')
               if f'(kda::sm90::kernel::Tag)11, {g}>' in name]
        initials=[i for i in ('true','false')
                  if f'(kda::sm90::kernel::Tag)8, cute::C<{i}>' in name]
        assert len(gates)==len(initials)==1,'unbound dtype/state specialization'
        key=(gates[0],initials[0]);assert key not in result
        result[key]=rows
    assert len(result)==4
    return result


def compare(candidate,parent):
    assert candidate.keys()==parent.keys()
    result={}
    fixed=('HGMMA','HMMA','LDSM','STSM','WARPGROUP','UTMA','BAR','MUFU','FADD','FMUL','FFMA','HADD2','F2FP','USETMAXREG')
    for key,rows in candidate.items():
        c,p=Counter(o for _,o,_ in rows),Counter(o for _,o,_ in parent[key])
        assert {o:c[o] for o in c if o.startswith(fixed)} == \
               {o:p[o] for o in p if o.startswith(fixed)},'matrix/math/async-completion changed'
        # Nine rings: Q,K,V,O,QK,KK,alpha,unused-alpha-last,beta; full+empty.
        assert c['SYNCS.EXCH.64']==56 and p['SYNCS.EXCH.64']==32,'stage initialization not lowered'
        assert c['SYNCS.PHASECHK.TRANS64'] < p['SYNCS.PHASECHK.TRANS64'],'unused metadata wait remains'
        result[str(key)]=dict(sites=len(rows),parent_sites=len(parent[key]),
                             current=dict(c),parent=dict(p))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate',type=Path);parser.add_argument('parent',type=Path)
    args=parser.parse_args();c,p=keyed(args.candidate),keyed(args.parent)
    result=compare(c,p)
    missing=dict(c);missing.pop(next(iter(missing)))
    completion=dict(c);key=next(iter(completion))
    completion[key]=[r for r in completion[key] if not r[1].startswith('WARPGROUP.DEPBAR')]
    for label,plant in [('old-stages',p),('missing-type',missing),('lost-completion',completion)]:
        try:compare(plant,p)
        except AssertionError:continue
        raise AssertionError(f'negative escaped: {label}')
    print(json.dumps(dict(status='PASS',scope='ACTUAL_NATIVE_PROFILE_NOT_SPEED',
                          bodies=result,negatives='3/3 EXPECTED_RED'),indent=2))


if __name__=='__main__':main()
