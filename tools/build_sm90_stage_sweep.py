#!/usr/bin/env python3
"""Build all32 actual stage configurations; fail/reject/pending stay separate."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from sm90_stage_space import space,validate,admit_types

ROOT=Path(__file__).resolve().parents[1]


def keyed_native(path):
    result={}
    for section in path.read_text().split('Function :')[1:]:
        symbol=section.splitlines()[0].strip()
        if 'FlatKernelTmaWarpSpecializedKdaFwd' not in symbol:continue
        name=subprocess.check_output(['c++filt',symbol],text=True)
        gates=[g for g in ('float','cutlass::bfloat16_t') if f'(kda::sm90::kernel::Tag)11, {g}>' in name]
        initials=[i for i in ('false','true') if f'(kda::sm90::kernel::Tag)8, cute::C<{i}>' in name]
        assert len(gates)==len(initials)==1,'unknown actual specialization'
        key=(gates[0],initials[0]);assert key not in result
        result[key]=re.findall(r'/\*[0-9a-f]+\*/\s*(.*?)\s*;\s*/\*',section)
    assert len(result)==4,'missing actual specialization'
    return result


def opcode_count(rows):
    return Counter(re.match(r'(?:@!?\w+\s+)?([A-Z][\w.]*)',r)[1] for r in rows)


def native_gate(path,parent,cell):
    got=keyed_native(path)
    assert got.keys()==parent.keys()
    if cell['id']==0:
        assert got==parent,'default native instructions/operands changed'
    stages=cell['stages']
    rings=stages[0]+stages[1]+stages[2]+stages[3]+4+2*stages[4]+stages[5]
    for key,rows in got.items():
        a,b=opcode_count(rows),opcode_count(parent[key])
        for prefix in ('HGMMA','HMMA','WARPGROUP','UTMA','USETMAXREG'):
            assert {o:n for o,n in a.items() if o.startswith(prefix)} == \
                   {o:n for o,n in b.items() if o.startswith(prefix)},(key,prefix)
        assert a['SYNCS.EXCH.64']==2*rings,'ring initialization not tied to compiled depths'
        assert all('gsb0, 0x0' in r for r in rows if 'WARPGROUP.DEPBAR' in r),'wait-all changed'
    return dict(scope='STATIC_NATIVE_NOT_SPEED',bodies=4,
                sites=sum(len(r) for r in got.values()),default_identity=cell['id']==0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--control-sass',type=Path,required=True)
    p.add_argument('--cutlass-root',type=Path,required=True)
    p.add_argument('--compiler',type=Path,default=Path('/usr/local/cuda-12.8/bin/nvcc'))
    p.add_argument('--jobs',type=int,default=8)
    a=p.parse_args()
    if not 1<=a.jobs<=16:p.error('bounded CPU jobs1..16')
    if a.out.exists():p.error('fresh directory required; do not overwrite compilation evidence')
    a.out.mkdir(parents=True)
    rows=[r|dict(state='PENDING') for r in space()]
    parent=keyed_native(a.control_sass)
    def save():
        validate(rows)
        result=dict(denominator=32,counts=dict(Counter(r['state'] for r in rows)),cells=rows,
                    source=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                    control_sass_sha256=hashlib.sha256(a.control_sass.read_bytes()).hexdigest())
        (a.out/'index.json').write_text(json.dumps(result,indent=2)+'\n')
    def build(cell):
        directory=a.out/f"cell-{cell['id']:02d}"
        cmd=[sys.executable,str(ROOT/'tools/build_gdn_sm90.py'),'--target','cuda_sm90','--release',
             '--stage-config',str(cell['id']),'--compiler',str(a.compiler),'--cutlass-root',str(a.cutlass_root),'--out',str(directory)]
        with (a.out/f"cell-{cell['id']:02d}.log").open('w') as log:
            rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
        item=cell|dict(build=str(directory.resolve()),command=cmd)
        if rc:return item|dict(state='BUILD_FAIL',reason=f'compiler/link/identity rc={rc}; inspect log')
        try:
            manifest=json.loads((directory/'build.json').read_text())
            assert manifest['complete'] and manifest['stage_config']==cell['id']
            admit_types(cell,manifest['stage_types'])
            native_log=(directory/'device.log').read_text()
            if 'C7512' in native_log or 'C7510' in native_log:
                return item|dict(state='NATIVE_REJECTED',reason='compiler serialized WGMMA',types=manifest['stage_types'])
            native=native_gate(directory/'codegen/image.sass',parent,cell)
            return item|dict(state='COMPILED',types=manifest['stage_types'],native=native,
                            binary_sha256=manifest['extension_sha256'])
        except Exception as e:
            return item|dict(state='NATIVE_FAIL',reason=str(e))
    save()
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        tasks={pool.submit(build,r):r['id'] for r in rows}
        for task in as_completed(tasks):
            index=tasks[task]
            try:rows[index]=task.result()
            except Exception as e:rows[index]=rows[index]|dict(state='BUILD_FAIL',reason=str(e))
            save()
            print(f"[stage build] cell={index:02d} state={rows[index]['state']} "
                  f"counts={dict(Counter(r['state'] for r in rows))} denominator=32",flush=True)
    if any(r['state'].endswith('FAIL') for r in rows):return 1
    return 0


if __name__=='__main__':sys.exit(main())
