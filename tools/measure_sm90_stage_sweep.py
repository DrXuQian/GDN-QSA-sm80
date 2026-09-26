#!/usr/bin/env python3
"""Sequential numerical admission and screening of the declared32cell space.

Uses the existing14-case/overflow runners unchanged. Compiler and numerical
failures are never environmental SKIPs. Event screening cannot promote code.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from sm90_stage_space import validate,select_finalists

ROOT=Path(__file__).resolve().parents[1]


def verify_cases(directory,parent):
    got=json.loads((directory/'cases.json').read_text())
    want=json.loads((parent/'relative-cases/cases.json').read_text())
    assert got['denominator']==got['completed']==want['denominator']==want['completed']==14
    assert len(got['cases'])==len(want['cases'])==14
    for g,w in zip(got['cases'],want['cases']):
        assert g['case']==w['case'] and g['rc']==w['rc']==0
        for key in ('input_sha256','output_sha256','errors'):
            assert g['result'][key]==w['result'][key],(g['case'],key)
        assert max(g['result']['errors'])<.02


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--builds',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--parent-data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stop-starting-at',required=True,help='UTC ISO time reserving time for final nsys/handoff')
    a=p.parse_args()
    deadline=datetime.fromisoformat(a.stop_starting_at)
    if deadline.tzinfo is None:p.error('deadline must specify UTC timezone')
    if a.out.exists():p.error('preserve existing numerical/screening receipts')
    a.out.mkdir(parents=True)
    index=json.loads((a.builds/'index.json').read_text())
    rows=index['cells'];validate(rows)
    if any(r['state']=='PENDING' for r in rows):raise RuntimeError('wait for complete compilation denominator')
    import torch
    from sys import path
    path.insert(0,str(ROOT/'benchmarks'))
    from bench_sm90_hopper import DeviceWatch
    DeviceWatch(0).sample(idle=True)
    # A separate short-lived query must exit before child DeviceWatch guards.
    # Do not create a persistent CUDA context in this CPU orchestrator.
    props=json.loads(subprocess.check_output([sys.executable,'-c',
        'import json,torch; p=torch.cuda.get_device_properties(0); '
        'print(json.dumps(dict(name=p.name,sms=p.multi_processor_count,'
        'shared=getattr(p,"shared_memory_per_block_optin",None))))'],text=True))
    limit=props['shared']
    if not isinstance(limit,int) or limit<=0:raise RuntimeError('actual shared capacity unavailable; do not guess')
    for row in rows:
        if row['state']=='COMPILED':
            if max(t['shared_bytes'] for t in row['types'])>limit:
                row.update(state='RESOURCE_REJECTED',reason=f'actual optin shared limit {limit}')
            else:row['state']='ADMISSION_PENDING'
    def save():
        validate(rows)
        result=dict(denominator=32,counts=dict(Counter(r['state'] for r in rows)),cells=rows,
                    shared_limit=limit,device=props['name'],sms=props['sms'],
                    source_index_sha256=hashlib.sha256((a.builds/'index.json').read_bytes()).hexdigest(),
                    updated_utc=datetime.now(timezone.utc).isoformat(),
                    finalists=select_finalists(rows),finalist_scope='EVENT_SCREEN_ONLY_NOT_SPEED_ADMISSION')
        (a.out/'index.json').write_text(json.dumps(result,indent=2)+'\n')
    def run(directory,label,cmd):
        with (directory/(label+'.log')).open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=240)
        if result.returncode:raise RuntimeError(f'{label} failed rc={result.returncode}; inspect log')
    order=list(range(1,32));random.Random(0x6A09E667).shuffle(order);order=[0]+order
    save()
    for config in order:
        row=rows[config]
        if row['state']!='ADMISSION_PENDING':continue
        if datetime.now(timezone.utc)>=deadline:break
        directory=a.out/f'cell-{config:02d}';directory.mkdir()
        extension=next(Path(row['build']).glob('_gdn_fused_sm90*.so'))
        assert hashlib.sha256(extension.read_bytes()).hexdigest()==row['binary_sha256']
        try:
            run(directory,'cases',[sys.executable,str(ROOT/'tests/run_sm90_hopper_cases.py'),
                '--extension',str(extension),'--backend','cuda_sm90','--out',str(directory/'cases')])
            verify_cases(directory/'cases',a.parent_data)
            for magnitude in (8,10000):
                destination=directory/f'stress{magnitude}'
                run(directory,f'stress{magnitude}',[sys.executable,str(ROOT/'tests/run_sm90_aux_mask_stress.py'),
                    '--extension',str(extension),'--gate',str(-magnitude),'--out',str(destination)])
                got=torch.load(destination/'captured.pt',weights_only=True,map_location='cpu')
                want=torch.load(a.parent_data/f'relative-stress{magnitude}/captured.pt',weights_only=True,map_location='cpu')
                assert len(got)==len(want)==2
                assert all(torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
                           for x,y in zip(got,want)),f'stress{magnitude} raw mismatch'
            row['numerics']='14CPU+PARENT_FINGERPRINTS+2DIRECT_RAW_STRESSES/PASS'
        except Exception as e:
            row.update(state='NUMERIC_FAIL',reason=str(e));save()
            print(f'[stage admission] config={config} NUMERIC_FAIL {e}',flush=True)
            # A new stage lifetime defect needs isolation, not more device work.
            raise
        try:
            run(directory,'screen',[sys.executable,str(ROOT/'benchmarks/screen_sm90_stage_cell.py'),
                '--candidate',str(extension),'--control',str(a.control),'--out',str(directory/'screen')])
            receipt=json.loads((directory/'screen/result.json').read_text())
            assert receipt['status']=='PASS'
            row.update(state='SCREENED',screen=receipt['screen'],evidence=str(directory.resolve()))
        except Exception as e:
            row.update(state='SCREEN_FAIL',reason=str(e));save();raise
        save()
        print(f"[stage screen] config={config:02d} stages={row['stages']} "
              f"ratios={[round(v['candidate_over_control'],5) for v in row['screen'].values()]} "
              f"counts={dict(Counter(r['state'] for r in rows))} denominator=32",flush=True)
    save()
    print(f'[stage sweep] counts={dict(Counter(r["state"] for r in rows))} denominator=32 '
          f'finalists={select_finalists(rows)} NSYS_ADMISSION_PENDING',flush=True)


if __name__=='__main__':main()
