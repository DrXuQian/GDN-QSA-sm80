#!/usr/bin/env python3
"""Registered delivery candidates: fixed14-case oracle and two raw stresses."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks'),str(ROOT/'tests')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',choices=('s45','s49','s50','s51','s52','s53','s54','s55','s56','s57','s58','s60','s61','s62','s63','s64','s65','s66','s68','s69'),default='s45')
    choice=parser.parse_args().candidate
    import torch
    from bench_sm90_hopper import DeviceWatch
    from profile_sm90_libraries import build_receipt
    root=Path({'s45':'/workspace/gdn-sm90-late-o1-20260926',
               's49':'/workspace/gdn-sm90-paired-state-20260926',
               's50':'/workspace/gdn-sm90-paired-tail-20260927',
               's51':'/workspace/gdn-sm90-local-inverse-20260927',
               's52':'/workspace/gdn-sm90-v64-paired-tail-20260927',
               's53':'/workspace/gdn-sm90-inverse-tail-20260927',
               's54':'/workspace/gdn-sm90-independent-tail-20260927',
               's55':'/workspace/gdn-sm90-v64-inverse-tail-20260927',
               's56':'/workspace/gdn-sm90-causal-sectors-20260927',
               's57':'/workspace/gdn-sm90-overlap-kk-first-20260927',
               's58':'/workspace/gdn-sm90-paired-retire-20260927',
               's60':'/workspace/gdn-sm90-aux-exp2-index-20260927',
               's61':'/workspace/gdn-sm90-local-diagonal-20260927',
               's62':'/workspace/gdn-sm90-kk-lookahead-20260927',
               's63':'/workspace/gdn-sm90-aux-ring-sweep-20260927',
               's64':'/workspace/gdn-sm90-aux-ring-sweep-20260927',
               's65':'/workspace/gdn-sm90-aux-ring-sweep-20260927',
               's66':'/workspace/gdn-sm90-local-inverse-prefix-20260927',
               's68':'/workspace/gdn-sm90-state-park-20260927',
               's69':'/workspace/gdn-sm90-packed-newv-20260927'}[choice])
    parent=Path('/workspace/gdn-sm90-win-20260926')
    build=root/(choice+'-build')
    binary=next(build.glob('_gdn_fused_sm90*.so'))
    identity=build_receipt(binary,'cuda_sm90','native')
    out=root/('s45-admission-r2' if choice=='s45' else choice+'-admission');out.mkdir()
    row=dict(id=choice,root=str(root),build=str(build),status='RUNNING',
             binary_sha256=identity['extension_sha256'])
    result=dict(denominator=1,rows=[row],performance='NOT_RUN',routing='UNCHANGED')
    from sm90_process_family import admission_watch
    watch=admission_watch(DeviceWatch)(0)
    def run(label,command,interpreter=True):
        with (out/(label+'.log')).open('x') as log:
            argv=([sys.executable] if interpreter else [])+list(map(str,command))
            p=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        if p.returncode:raise RuntimeError(f'{label} rc={p.returncode}, see {out}')
    try:
        for _ in range(3):watch.sample(idle=True);time.sleep(.2)
        if any(r['telemetry'].split(',')[0]!='GPU-1d5fdef3-4899-79d9-19e6-c9c815b2a59c' for r in watch.records):
            raise RuntimeError('unexpected device identity')
        watch.thread.start()
        if choice=='s69':
            run('source-native',[root/'source/dev/backends/check_sm90_packed_newv.py',
                '--candidate',build/'codegen/image.sass',
                '--parent','/workspace/gdn-sm90-paired-tail-20260927/s50-build/codegen/image.sass',
                '--original',parent/'relative-build/codegen/image.sass',
                '--device-log',build/'device.log'])
        elif choice=='s68':
            if '-DGDN_SM90_SHARED_STATE=1' not in identity['flags']:
                raise ValueError('S68 must select the registered one-state shared-H body')
            run('source-lifetime-native',[root/'source/dev/backends/check_sm90_state_park.py',
                '--candidate',build/'codegen/image.sass','--device-log',build/'device.log'])
        elif choice=='s53':
            for label,checker,control in (
                ('inverse','check_sm90_inverse_local.py','/workspace/gdn-sm90-paired-tail-20260927/s50-build'),
                ('overlap','check_sm90_paired_tail.py','/workspace/gdn-sm90-local-inverse-20260927/s51-build')):
                run('native-'+label,[root/'source/dev/backends'/checker,
                    '--candidate',build/'codegen/image.sass','--parent',Path(control)/'codegen/image.sass',
                    '--device-log',build/'device.log'])
        elif choice in ('s54','s56','s57','s58','s60','s61','s62','s63','s64','s65','s66'):
            checker={'s54':'check_sm90_independent_tail.py','s56':'check_sm90_causal_sectors.py',
                     's57':'check_sm90_kk_overlap.py','s58':'check_sm90_paired_retire.py',
                     's60':'check_sm90_exp2_index.py',
                     's61':'check_sm90_local_diagonal.py',
                     's62':'check_sm90_kk_lookahead.py',
                     's63':'check_sm90_aux_rings.py','s64':'check_sm90_aux_rings.py',
                     's65':'check_sm90_aux_rings.py','s66':'check_sm90_inverse_prefix.py'}[choice]
            ring_config={'s63':1,'s64':2,'s65':3}.get(choice)
            if ring_config is not None and f'-DGDN_SM90_AUX_RING_CONFIG={ring_config}' not in identity['flags']:
                raise ValueError('compiled auxiliary ring configuration differs')
            run('native-source',[root/'source/dev/backends'/checker,
                '--candidate',build/'codegen/image.sass',
                '--parent','/workspace/gdn-sm90-paired-tail-20260927/s50-build/codegen/image.sass',
                '--device-log',build/'device.log',
                *(['--ptx',root/'retained-ptx/launch.ptx'] if choice in ('s61','s66') else []),
                *(['--config',ring_config,'--types',root/f'types-{ring_config}.jsonl']
                  if ring_config is not None else [])])
            if choice=='s60':
                run('source-range',[root/'source/tests/test_sm90_aux_range.py'])
                run('exact-exp2-seam',[root/'exp2-probe'],interpreter=False)
        elif choice in ('s50','s51','s52','s55'):
            checker='check_sm90_inverse_local.py' if choice in ('s51','s55') else 'check_sm90_paired_tail.py'
            native_parent=(Path('/workspace/gdn-sm90-value-split-20260926/s38-build')
                           if choice=='s52' else parent/'relative-build')
            if choice=='s55':
                native_parent=Path('/workspace/gdn-sm90-v64-paired-tail-20260927/s52-build')
            if choice in ('s52','s55'):
                flags=json.loads((build/'build.json').read_text())['flags']
                if '-DGDN_SM90_VALUE_SPLIT_AUX_REGS=232' not in flags:
                    raise ValueError('V64 candidate must compile the registered aux232 geometry')
            run('native-source',[root/'source/dev/backends'/checker,
                '--candidate',build/'codegen/image.sass','--parent',native_parent/'codegen/image.sass',
                '--device-log',build/'device.log'])
        else:
            native='check_sm90_late_o1_native.py' if choice=='s45' else 'check_sm90_paired_state_native.py'
            progress='check_sm90_late_o1.py' if choice=='s45' else 'check_sm90_paired_state.py'
            run('native',[root/'source/dev/backends'/native,build/'codegen/image.sass',
                parent/'relative-build/codegen/image.sass','--device-log',build/'device.log'])
            run('progress',[root/'source/dev/backends'/progress])
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
    print(f'[{choice.upper()} admission] PASS; performance NOT_RUN',flush=True)


if __name__=='__main__':main()
