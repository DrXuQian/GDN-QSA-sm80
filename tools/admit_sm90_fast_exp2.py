#!/usr/bin/env python3
"""S74 explicit math variant: unchanged CPU gate, no fake parent-RAW claim."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'benchmarks'), str(ROOT/'tests')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    from bench_sm90_hopper import DeviceWatch
    from profile_sm90_libraries import build_receipt
    from sm90_process_family import admission_watch
    root=Path('/workspace/gdn-sm90-fast-exp2-20260927')
    build=root/'s74-build-r2'
    binary=next(build.glob('_gdn_fused_sm90*.so'))
    receipt=build_receipt(binary,'cuda_sm90','native')
    manifest=json.loads((build/'build.json').read_text())
    if manifest.get('exp2_mode')!='fastmath-exp2-only' or '-DGDN_SM90_FAST_EXP2=1' not in receipt['flags']:
        raise ValueError('not the authorized exp2-only binary')
    a.out.mkdir()
    row=dict(id='s74',root=str(root),build=str(build),status='RUNNING',
             binary_sha256=receipt['extension_sha256'],math_variant='fastmath-exp2-only')
    result=dict(denominator=1,rows=[row],performance='NOT_RUN',routing='UNCHANGED',
                parent_raw='DIAGNOSTIC_ONLY_USER_AUTHORIZED_ARITHMETIC_VARIANT')
    watch=admission_watch(DeviceWatch)(0)
    def run(label,cmd):
        with (a.out/(label+'.log')).open('x') as log:
            child=subprocess.run([sys.executable,*map(str,cmd)],stdout=log,
                stderr=subprocess.STDOUT,timeout=300)
        if child.returncode: raise RuntimeError(f'{label} failed; see {a.out}')
    try:
        for _ in range(3): watch.sample(idle=True);time.sleep(.2)
        if any(x['telemetry'].split(',')[0]!='GPU-1d5fdef3-4899-79d9-19e6-c9c815b2a59c' for x in watch.records):
            raise ValueError('unexpected GPU UUID')
        watch.thread.start()
        run('cases',[ROOT/'tests/run_sm90_hopper_cases.py','--extension',binary,
                    '--backend','cuda_sm90','--out',a.out/'cases'])
        actual=json.loads((a.out/'cases/cases.json').read_text())
        parent=json.loads(Path('/workspace/gdn-sm90-win-20260926/relative-cases/cases.json').read_text())
        if actual['completed']!=actual['denominator'] or actual['completed']!=14:
            raise ValueError('fixed fourteen-case denominator changed')
        differences=[]
        for got,want in zip(actual['cases'],parent['cases']):
            if got['case']!=want['case'] or got['rc']!=0 or got['result']['input_sha256']!=want['result']['input_sha256']:
                raise ValueError('wrong workload/input or failed CPU oracle')
            differences.append(dict(case=got['case'],errors=got['result']['errors'],
                parent_raw_equal=got['result']['output_sha256']==want['result']['output_sha256']))
        row['cross_parent_diagnostic']=differences
        for magnitude in (8,10000):
            run(f'stress{magnitude}',[ROOT/'tests/run_sm90_aux_mask_stress.py',
                '--extension',binary,'--gate',-magnitude,'--out',a.out/f'stress{magnitude}'])
        row.update(status='PASS',numerical='14CPU+2EXTREME_UNCHANGED_2_PERCENT')
    except Exception as e:
        row.update(status='FAIL',error=repr(e));raise
    finally:
        watch.stop.set()
        if watch.thread.ident is not None: watch.thread.join(timeout=22)
        if watch.thread.is_alive() or watch.errors:
            row.update(status='FAIL',error='incomplete exclusive-device evidence')
        result.update(device_watch=dict(records=watch.records,errors=watch.errors),
                      harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (a.out/'admission.json').write_text(json.dumps(result,indent=2)+'\n')
    if row['status']!='PASS': raise RuntimeError(row)
    print('[S74 admission] PASS: fixed2%CPU gate; performance NOT_RUN',flush=True)


if __name__=='__main__': main()
