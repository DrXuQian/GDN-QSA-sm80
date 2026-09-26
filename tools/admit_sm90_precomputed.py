#!/usr/bin/env python3
"""S47 fixed numerical/resource admission; no timing follows a failed cell."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks'),str(ROOT/'tests'),str(ROOT/'tools')]


def main():
    import torch
    from bench_sm90_hopper import DeviceWatch
    from profile_sm90_libraries import build_receipt
    from sm90_process_family import admission_watch
    from sm90_workloads import WORKLOADS, validate_inventory
    from sm90_execution_contract import from_build, PREPARED
    root=Path('/workspace/gdn-sm90-precomputed-aux-20260926')
    parent=Path('/workspace/gdn-sm90-win-20260926')
    build=root/'s47-build'
    binary=next(build.glob('_gdn_fused_sm90*.so'))
    identity=build_receipt(binary,'cuda_sm90','native')
    if from_build(identity)!=PREPARED:raise ValueError('candidate not the prepared build')
    out=root/'s47-admission';out.mkdir()
    row=dict(id='s47',root=str(root),build=str(build),status='RUNNING',
             binary_sha256=identity['extension_sha256'])
    result=dict(denominator=1,rows=[row],performance='NOT_RUN',routing='UNCHANGED')
    watch=admission_watch(DeviceWatch)(0)
    def run(label,command,timeout=300):
        with (out/(label+'.log')).open('x') as log:
            p=subprocess.run(list(map(str,command)),stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if p.returncode:raise RuntimeError(f'{label} rc={p.returncode}, see {out}')
    try:
        for _ in range(3):watch.sample(idle=True);time.sleep(.2)
        if any(r['telemetry'].split(',')[0]!='GPU-1d5fdef3-4899-79d9-19e6-c9c815b2a59c' for r in watch.records):
            raise RuntimeError('unexpected physical device identity')
        watch.thread.start()
        run('native',[sys.executable,root/'source/dev/backends/check_sm90_precomputed_binary.py',
            build/'launch.o','--cuobjdump','/usr/local/cuda-12.8/bin/cuobjdump','--out',out/'native'])
        run('progress',[sys.executable,root/'source/dev/backends/check_sm90_precomputed_protocol.py'])
        include=root/'source/csrc/backends/sm90'
        run('map-build',['/usr/local/cuda-12.8/bin/nvcc','-std=c++17','-O2','--expt-relaxed-constexpr',
            '--extended-lambda','-gencode=arch=compute_90a,code=sm_90a',f'-I{include}',f'-I{include}/cula',
            '-I/workspace/gdn-sm90-h800-20260926/source/third_party/cutlass/include',
            root/'source/dev/backends/sm90_precomputed_map.cu','-o',out/'host-map'])
        validate_inventory(WORKLOADS)
        try:validate_inventory(WORKLOADS[:-1])
        except ValueError:pass
        else:raise AssertionError('lost workload escaped')
        for workload in WORKLOADS:run('map-'+workload.name,[out/'host-map',*workload.shape])
        row['map_workloads']=len(WORKLOADS)
        spec=importlib.util.spec_from_file_location('_gdn_fused_sm90',binary)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if module.execution_structure!='chunk-parallel-aux+V64-state;2-kernels;private-physical-operands':
            raise ValueError('loaded execution structure differs')
        resources={}
        for fp32 in (False,True):
            for initial in (False,True):
                key=f'fp32={fp32},initial={initial}'
                resources[key]=dict(module.resources(fp32,initial))
                if resources[key]['state']['blocks_per_sm']<2:
                    raise ValueError(f'registered state occupancy failed: {resources[key]}')
        row['resources']=resources
        run('cases',[sys.executable,ROOT/'tests/run_sm90_hopper_cases.py','--extension',binary,
                     '--backend','cuda_sm90','--out',out/'cases'])
        expected=json.loads((parent/'relative-cases/cases.json').read_text())
        actual=json.loads((out/'cases/cases.json').read_text())
        if actual['completed']!=14 or actual['denominator']!=14 or len(actual['cases'])!=14:
            raise ValueError('fixed numerical denominator changed')
        for got,want in zip(actual['cases'],expected['cases']):
            if got['case']!=want['case'] or got['rc']!=0 or want['rc']!=0:raise ValueError('case mismatch')
            for key in ('input_sha256','output_sha256','errors'):
                if got['result'][key]!=want['result'][key]:raise ValueError(f'parent raw result differs: {got["case"]}/{key}')
        for magnitude in (8,10000):
            directory=out/f'stress{magnitude}'
            run(f'stress{magnitude}',[sys.executable,ROOT/'tests/run_sm90_aux_mask_stress.py',
                '--extension',binary,'--gate',-magnitude,'--out',directory])
            got=torch.load(directory/'captured.pt',weights_only=True,map_location='cpu')
            want=torch.load(parent/f'relative-stress{magnitude}/captured.pt',weights_only=True,map_location='cpu')
            if len(got)!=2 or len(want)!=2 or not all(torch.equal(x.contiguous().view(torch.uint8),
                        y.contiguous().view(torch.uint8)) for x,y in zip(got,want)):
                raise ValueError('extreme-gate direct parent-byte mismatch')
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
    print('[S47 admission] PASS; performance NOT_RUN',flush=True)


if __name__=='__main__':main()
