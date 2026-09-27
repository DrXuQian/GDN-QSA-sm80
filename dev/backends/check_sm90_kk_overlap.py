#!/usr/bin/env python3
"""S57 publication CFG on the S50 overlapping-state parent, all four bodies."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
from check_sm90_binary import inspect
from check_sm90_paired_tail import progress

ROOT = Path(__file__).resolve().parents[2]


def split(text):
    inspect(text)
    parts = re.split(r'Function\s*:\s*(\S+)', text)
    return dict(zip(parts[1::2], parts[2::2]))


def parse(body):
    return [(int(m[1],16), m[2] or '', m[3], m[4].strip()) for m in re.finditer(
        r'/\*([0-9a-f]+)\*/\s+(?:(@!?\w+)\s+)?([A-Z][\w.]*)\s*([^;]*);',body)]


def publication(rows):
    starts = [i for i,r in enumerate(rows) if r[2]=='SYNCS.ARRIVE.TRANS64.A1T0' and '+0x29080]' in r[3]]
    qready = [r[0] for r in rows if r[2]=='SYNCS.ARRIVE.TRANS64.A1T0' and '+0x29060]' in r[3]]
    waits = [i for i,r in enumerate(rows) if r[2]=='SYNCS.PHASECHK.TRANS64.TRYWAIT' and '+0x29070]' in r[3]]
    if len(starts)!=len(qready) or len(starts)!=2 or len(waits)!=4:
        raise ValueError('full/tail publication/retry denominator differs')
    entries = []
    retry_pcs = []
    for s,ready in zip(starts,qready):
        if s+1 not in waits or rows[s+2][1:3]!=('@!P0','BRA'):
            raise ValueError('KK-ready does not precede QK-empty acquire')
        entry = rows[s+1][0]
        target = int(rows[s+2][3],16)
        if not entry < ready < target:
            raise ValueError('wrong mainline/tail CFG geometry')
        retry = [i for i in waits if target <= rows[i][0] <= target+0x40]
        if len(retry)!=1:
            raise ValueError('missing unique out-of-line retry')
        r=retry[0]
        if rows[r+3][1:3]!=('@!P0','BRA') or int(rows[r+3][3],16)!=rows[r][0]:
            raise ValueError('retry does not loop on the same QK barrier')
        if rows[r+4][1:3]!=('','BRA') or int(rows[r+4][3],16)!=entry+0x20:
            raise ValueError('retry returns outside admitted acquire continuation')
        entries.append((rows[s][0],entry,ready));retry_pcs.append(rows[r][0])
    if len(set(retry_pcs))!=2:
        raise ValueError('mixed full/tail retry')
    return dict(main_publications=entries,out_of_line_retries=retry_pcs)


def check(control,candidate):
    old,new=split(control),split(candidate)
    if old.keys()!=new.keys():raise ValueError('specialization changed')
    result={}
    for key in new:
        a,b=parse(old[key]),parse(new[key])
        cats=lambda rs: Counter(r[2] for r in rs if r[2].startswith(
            ('HGMMA','HMMA','LDSM','STSM','SYNCS','WARPGROUP','BAR','UTMA')))
        if cats(a)!=cats(b):raise ValueError('math or data lifetime changed')
        result[key]=publication(b)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--device-log',type=Path,required=True)
    a=p.parse_args()
    state='csrc/backends/sm90/scalar_gdn_state.cuh'
    parent_state=subprocess.check_output(['git','show',f'170f34f:{state}'],cwd=ROOT,text=True)
    if (ROOT/state).read_text()!=parent_state:raise ValueError('S50 state lifetime changed')
    old,new=a.parent.read_text(),a.candidate.read_text()
    if 'C7512' in a.device_log.read_text():raise ValueError('serialized WGMMA')
    result=check(old,new)
    wrong_ready=re.sub(r'(SYNCS\.ARRIVE\.TRANS64\.A1T0[^;]*\+)0x29080\]',
                       r'\g<1>0x29060]',new,count=1)
    if wrong_ready==new:raise AssertionError('negative failed to change the publishing instruction')
    plants=[old,wrong_ready,
            new.replace('WARPGROUP.DEPBAR.LE','REMOVED.DEPBAR',1)]
    for x in plants:
        try:check(old,x)
        except ValueError:continue
        raise AssertionError('stale order/wrong ready/lost retirement negative escaped')
    print(json.dumps(dict(native=result,native_negatives=3,
        retained_data_ring_progress={str(n):progress(n) for n in range(1,9)},
        scope='native CFG plus reduced progress, not device memory-order proof'),indent=2))


if __name__=='__main__':main()
