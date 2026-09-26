#!/usr/bin/env python3
"""Recheck V64 admission and the complete registered two-library verdict.

No timing threshold lives here: all verdicts come from analyze_sm90_nsys.
This records evidence only; it never shuts down a machine or changes routing.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'dev/backends'))
from check_sm90_value_native import bodies,check
from analyze_sm90_nsys import extract

UUID='GPU-1d5fdef3-4899-79d9-19e6-c9c815b2a59c'
CAPTURES=['s37-flashinfer-g-0.1',
          's38-flashinfer-g-0.1','s38-flashinfer-g-1.0',
          's38-flashqla-g-0.1','s38-flashqla-g-1.0',
          's38-confirm-flashinfer-g-0.1','s38-confirm-flashinfer-g-1.0']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def admitted(rows):
    if len(rows)!=len(CAPTURES) or {r['capture'] for r in rows}!=set(CAPTURES):return False
    selected=[r for r in rows if r['capture'].startswith('s38-')]
    return (len(selected)==6 and len({r['binary_sha256'] for r in selected})==1
            and all(r['all_reference_paths_win'] and r['vs_parent']=='CANDIDATE-WINS' for r in selected))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--stage-harness',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.root
    sys.path.insert(0,str(a.stage_harness/'tools'))
    from measure_sm90_stage_sweep import verify_cases
    import torch
    parent=bodies(a.parent/'relative-build/codegen/image.sass')
    checks={}
    for name,aux in (('s37',104),('s38',232)):
        build=root/(name+'-build');identity=json.loads((build/'build.json').read_text())
        binary=list(build.glob('_gdn_fused_sm90*.so'));assert len(binary)==1
        assert identity['complete'] and sha(binary[0])==identity['extension_sha256']
        assert f'-DGDN_SM90_VALUE_SPLIT_AUX_REGS={aux}' in identity['flags']
        assert 'C7512' not in (build/'device.log').read_text() and 'C7510' not in (build/'device.log').read_text()
        native=check(bodies(build/'codegen/image.sass'),parent,aux)
        verify_cases(root/(name+'-cases'),a.parent)
        for magnitude in (8,10000):
            x=torch.load(root/f'{name}-stress{magnitude}/captured.pt',weights_only=True,map_location='cpu')
            y=torch.load(a.parent/f'relative-stress{magnitude}/captured.pt',weights_only=True,map_location='cpu')
            assert len(x)==len(y)==2
            assert all(torch.equal(l.contiguous().view(torch.uint8),r.contiguous().view(torch.uint8)) for l,r in zip(x,y))
        screen=json.loads((root/f'{name}-screen/result.json').read_text())
        watch=json.loads((root/f'{name}-screen/device-watch.json').read_text())
        assert screen['status']=='PASS' and watch['errors']==[]
        checks[name]=dict(source=identity['repository_revision'],binary_sha256=identity['extension_sha256'],
            native=native,numerics='14CPU_PARENT_FINGERPRINTS+2DIRECT_BYTE_STRESSES/PASS',
            screen=screen['screen'],build_sha256=sha(build/'build.json'))
    verify_cases(root/'default-cases',a.parent)
    host=(root/'host-tests.log').read_text();assert 'Ran 38 tests' in host and '\nOK\n' in host
    runs=[]
    for name in CAPTURES:
        d=root/name;receipt=json.loads((d/'receipt.json').read_text());result=json.loads((d/'result.json').read_text())
        assert receipt['device_watch']['errors']==[] and receipt['candidate_requires_raw_bit']
        assert all(r['repeat']=='8/8 RAW-BIT' for r in receipt['admission'].values())
        assert all(r['telemetry'].split(',')[0]==UUID for r in receipt['device_watch']['records'])
        assert result['shape']==[1,2048,16,32,128] and result['gate'] in (-.1,-1.)
        assert all(r['calls']==12 for r in result['summary'].values())
        with sqlite3.connect(f'file:{d}/forward.sqlite?mode=ro',uri=True) as conn:
            recovered=json.loads(json.dumps(extract(conn,receipt)))
        assert recovered=={k:v for k,v in result.items() if k!='evidence_sha256'}
        for path,digest in result['evidence_sha256'].items():
            local=ROOT/'tools/analyze_sm90_nsys.py' if Path(path).name=='analyze_sm90_nsys.py' else d/Path(path).name
            assert sha(local)==digest
        candidate=name[:3];build=receipt['incumbent_builds']['ours-candidate']
        assert build['extension_sha256']==checks[candidate]['binary_sha256']
        runs.append(dict(capture=name,gate=result['gate'],source=build['repository_revision'],
            binary_sha256=build['extension_sha256'],forwards=len(result['forwards']),
            timings={k:v['kernel_sum_us'] for k,v in result['summary'].items()},
            vs_parent=result['comparisons']['ours-candidate']['verdict'],
            all_reference_paths_win=result['candidate_beats_all_reference_paths'],
            references=result['candidate_vs_references'],
            sha256={n:sha(d/n) for n in ('receipt.json','result.json','forward.sqlite','forward.nsys-rep')}))
    # Missing a confirmation, a loss, or a mixed binary must not authorize a
    # success claim. These checks consume the same goal function as the report.
    assert not admitted(runs[:-1])
    changed=copy.deepcopy(runs);changed[-1]['all_reference_paths_win']=False;assert not admitted(changed)
    changed=copy.deepcopy(runs);changed[-1]['binary_sha256']='wrong-image';assert not admitted(changed)
    result=dict(scope='H800_PHYSICAL_NOT_NATIVE_PPU17',device_uuid=UUID,
        workload='B1/T2048/Hqk16/Hv32/K128/V128/C64; gates-.1/-1; BF16; FP32finalstate',
        goal_admitted=admitted(runs),native_ppu17='SKIP_SDK_MODEL_UNAVAILABLE',
        incumbent='S24_RETAINED',selected='S38_EXPLICIT_VALUE_SPLIT_232',routing='UNCHANGED',
        captures=len(runs),complete_forwards=sum(r['forwards'] for r in runs),
        reanalysis='EXACT_SQLITE_REEXTRACTION_ALL_CAPTURES',goal_negatives=3,
        default='14CPU_PARENT_BITS_PASS; native_recompile_NONIDENTICAL_27840_vs27856',
        host_tests=38,checks=checks,runs=runs)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('runs','checks')})
    if not result['goal_admitted']:raise RuntimeError('goal NOT MET; retain machine and continue')

if __name__=='__main__':main()
