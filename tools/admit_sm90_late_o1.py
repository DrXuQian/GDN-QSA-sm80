#!/usr/bin/env python3
"""S45 only: reuse the fixed14-case oracle and two exact overflow stresses."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks'),str(ROOT/'tests')]


def main():
    import torch
    from bench_sm90_hopper import DeviceWatch
    from profile_sm90_libraries import build_receipt
    root=Path('/workspace/gdn-sm90-late-o1-20260926')
    parent=Path('/workspace/gdn-sm90-win-20260926')
    build=root/'s45-build'
    binary=next(build.glob('_gdn_fused_sm90*.so'))
    identity=build_receipt(binary,'cuda_sm90','native')
    out=root/'s45-admission-r2';out.mkdir()
    row=dict(id='s45',root=str(root),build=str(build),status='RUNNING',
             binary_sha256=identity['extension_sha256'])
    result=dict(denominator=1,rows=[row],performance='NOT_RUN',routing='UNCHANGED')
    from sm90_process_family import admission_watch
    watch=admission_watch(DeviceWatch)(0)
    def run(label,command):
        with (out/(label+'.log')).open('x') as log:
            p=subprocess.run([sys.executable,*map(str,command)],stdout=log,stderr=subprocess.STDOUT,timeout=300)
        if p.returncode:raise RuntimeError(f'{label} rc={p.returncode}, see {out}')
    try:
        for _ in range(3):watch.sample(idle=True);time.sleep(.2)
        if any(r['telemetry'].split(',')[0]!='GPU-1d5fdef3-4899-79d9-19e6-c9c815b2a59c' for r in watch.records):
            raise RuntimeError('unexpected device identity')
        watch.thread.start()
        run('native',[root/'source/dev/backends/check_sm90_late_o1_native.py',build/'codegen/image.sass',
            parent/'relative-build/codegen/image.sass','--device-log',build/'device.log'])
        run('progress',[root/'source/dev/backends/check_sm90_late_o1.py'])
        run('cases',[ROOT/'tests/run_sm90_hopper_cases.py','--extension',binary,'--backend','cuda_sm90','--out',out/'cases'])
        expected=json.loads((parent/'relative-cases/cases.json').read_text())
        actual=json.loads((out/'cases/cases.json').read_text())
        assert actual['completed']==actual['denominator']==len(actual['cases'])==14
        for got,want in zip(actual['cases'],expected['cases']):
            assert got['case']==want['case'] and got['rc']==want['rc']==0
            for key in ('input_sha256','output_sha256','errors'):assert got['result'][key]==want['result'][key],(got['case'],key)
        for magnitude in (8,10000):
            directory=out/f'stress{magnitude}'
            run(f'stress{magnitude}',[ROOT/'tests/run_sm90_aux_mask_stress.py','--extension',binary,'--gate',-magnitude,'--out',directory])
            got=torch.load(directory/'captured.pt',weights_only=True,map_location='cpu')
            want=torch.load(parent/f'relative-stress{magnitude}/captured.pt',weights_only=True,map_location='cpu')
            assert len(got)==len(want)==2
            assert all(torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)) for x,y in zip(got,want))
        row.update(status='PASS',numerical='14CPU_PARENT_FINGERPRINTS+2DIRECT_BYTE_STRESSES')
    except Exception as error:
        row.update(status='FAIL',error=repr(error));raise
    finally:
        watch.stop.set()
        if watch.thread.ident is not None:watch.thread.join(timeout=22)
        if watch.thread.is_alive() or watch.errors:row.update(status='FAIL',error='incomplete exclusive-device evidence')
        result.update(device_watch=dict(records=watch.records,errors=watch.errors),
            harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (out/'admission.json').write_text(json.dumps(result,indent=2)+'\n')
    if row['status']!='PASS':raise RuntimeError(row)
    print('[S45 admission] PASS; performance NOT_RUN',flush=True)


if __name__=='__main__':main()
