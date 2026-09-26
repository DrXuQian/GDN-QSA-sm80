#!/usr/bin/env python3
"""Close the registered32cell inventory using raw numeric/native/nsys evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import torch
from sm90_stage_space import validate,select_finalists,admit_types
from build_sm90_stage_sweep import keyed_native,native_gate
from measure_sm90_stage_sweep import verify_cases
from analyze_sm90_nsys import extract

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.root
    index=json.loads((root/'screen/index.json').read_text());rows=index['cells'];validate(rows)
    assert index['counts']=={'SCREENED':32} and all(r['state']=='SCREENED' for r in rows)
    assert index['finalists']==select_finalists(rows)
    parent=keyed_native(a.parent/'relative-build/codegen/image.sass')
    checks=[]
    for row in rows:
        build=root/f"builds/cell-{row['id']:02d}"
        evidence=root/f"screen/cell-{row['id']:02d}"
        identity=json.loads((build/'build.json').read_text());admit_types(row,identity['stage_types'])
        assert identity['complete'] and identity['extension_sha256']==row['binary_sha256']
        binaries=list(build.glob('_gdn_fused_sm90*.so'));assert len(binaries)==1
        assert sha(binaries[0])==row['binary_sha256']
        assert 'C7512' not in (build/'device.log').read_text()
        native=native_gate(build/'codegen/image.sass',parent,row)
        verify_cases(evidence/'cases',a.parent)
        for magnitude in (8,10000):
            got=torch.load(evidence/f'stress{magnitude}/captured.pt',weights_only=True,map_location='cpu')
            want=torch.load(a.parent/f'relative-stress{magnitude}/captured.pt',weights_only=True,map_location='cpu')
            assert len(got)==len(want)==2
            assert all(torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)) for x,y in zip(got,want))
        screen=json.loads((evidence/'screen/result.json').read_text())
        assert screen['status']=='PASS' and screen['screen']==row['screen']
        checks.append(dict(id=row['id'],stages=row['stages'],binary_sha256=row['binary_sha256'],
            source=identity['repository_revision'],native=native,screen=row['screen'],
            numerical='14CPU_PARENT_FINGERPRINTS+2DIRECT_BYTE_STRESSES/PASS'))
    runs=[]
    for cell in index['finalists']:
        for gate in ('-0.1','-1.0'):
            directory=root/f'cell-{cell:02d}-fi-g{gate}'
            receipt=json.loads((directory/'receipt.json').read_text())
            result=json.loads((directory/'result.json').read_text())
            assert receipt['device_watch']['errors']==[] and receipt['candidate_requires_raw_bit']
            assert all(r['repeat']=='8/8 RAW-BIT' for r in receipt['admission'].values())
            assert all(r['calls']==12 for r in result['summary'].values())
            with sqlite3.connect(f'file:{directory}/forward.sqlite?mode=ro',uri=True) as con:
                recovered=json.loads(json.dumps(extract(con,receipt)))
            assert recovered=={k:v for k,v in result.items() if k!='evidence_sha256'}
            for path,digest in result['evidence_sha256'].items():
                local=Path(__file__).with_name('analyze_sm90_nsys.py') if Path(path).name=='analyze_sm90_nsys.py' else directory/Path(path).name
                assert sha(local)==digest
            assert receipt['incumbent_builds']['ours-candidate']['extension_sha256']==rows[cell]['binary_sha256']
            runs.append(dict(cell=cell,gate=float(gate),forwards=len(result['forwards']),
                timings={k:v['kernel_sum_us'] for k,v in result['summary'].items()},
                verdict_vs_parent=result['comparisons']['ours-candidate'],
                references=result['candidate_vs_references'],
                files={name:sha(directory/name) for name in ('receipt.json','result.json','forward.sqlite','forward.nsys-rep')}))
    result=dict(scope='H800_NOT_NATIVE_PPU17',denominator=32,compiled=32,native_pass=32,
        numeric_pass=32,screened=32,pending=0,rejected=0,finalists=index['finalists'],
        captures=len(runs),complete_forwards=sum(r['forwards'] for r in runs),
        reanalysis='EXACT_SQLITE_NATIVE_NUMERIC_REEXTRACTION',routing='UNCHANGED',cells=checks,runs=runs)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('cells','runs')})

if __name__=='__main__':main()
