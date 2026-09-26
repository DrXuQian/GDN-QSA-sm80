#!/usr/bin/env python3
"""Six-body S47 codegen gate, not device correctness/performance admission."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def inspect(sass, device_log):
    if 'C7512' in device_log:
        raise ValueError('serialized WGMMA: ptxas C7512')
    if not re.search(r'arch = sm_90a', sass):
        raise ValueError('not assembled SM90a')
    pieces = re.split(r'Function\s*:\s*(\S+)', sass)
    bodies = dict(zip(pieces[1::2],pieces[2::2]))
    state = {k:v for k,v in bodies.items() if 'FlatKernelTmaWarpSpecializedKdaFwd' in k}
    prepare = {k:v for k,v in bodies.items() if 'prepare_aux_device' in k}
    if len(state)!=4 or len(prepare)!=2 or len(bodies)!=6:
        raise ValueError(f'expected 4 state + 2 prepare bodies, got {len(state)}/{len(prepare)}/{len(bodies)}')
    result = {}
    for role, selected in (('state',state),('prepare',prepare)):
        for name,body in selected.items():
            counts = {op:len(re.findall(r'\b'+op+r'(?:\.|\s)',body)) for op in
                      ('HGMMA','HMMA','UTMALDG','UTMASTG','STG','LDGSTS','LDL','STL','WARPGROUP')}
            required = ('HGMMA','UTMALDG','UTMASTG','STG','LDGSTS') if role=='state' else ('HGMMA','HMMA','UTMALDG','STG')
            if any(not counts[op] for op in required):
                raise ValueError(f'{role}: missing required operation: {counts}')
            if role=='state' and counts['HMMA']:
                raise ValueError('inverse unexpectedly still present in state')
            if role=='state' and (not re.search(r'\bSTG\.E\.U16\s',body) or
                                 not re.search(r'\bSTG\.E(?:\.64|\.128)?\s',body)):
                raise ValueError('state missing BF16 tail or FP32 final publication')
            result[name] = dict(role=role,opcodes=counts)
    return result


def negatives(sass,log):
    pieces=re.split(r'(Function\s*:\s*\S+)',sass)
    plants=[]
    for role in ('prepare_aux_device','FlatKernelTmaWarpSpecializedKdaFwd'):
        index=next(i for i in range(1,len(pieces),2) if role in pieces[i])
        for op in ('HGMMA','UTMALDG','STG'):
            copy=pieces.copy()
            copy[index+1]=re.sub(r'\b'+op+r'(?=\.|\s)','REMOVED',copy[index+1])
            plants.append((''.join(copy),log))
        copy=pieces.copy(); copy[index]=copy[index].replace(role,'OMITTED',1)
        plants.append((''.join(copy),log))
    plants.append((sass,log+'\nwarning C7512'))
    for text,record in plants:
        try:inspect(text,record)
        except ValueError:pass
        else:raise AssertionError('missing image/math/publication or serialization escaped')
    return len(plants)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary',type=Path);p.add_argument('--cuobjdump',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    sass=subprocess.check_output([a.cuobjdump,'--dump-sass',str(a.binary)],text=True)
    resource=subprocess.check_output([a.cuobjdump,'--dump-resource-usage',str(a.binary)],text=True)
    log=(a.binary.parent/'device.log').read_text()
    (a.out/'image.sass').write_text(sass);(a.out/'resources.txt').write_text(resource)
    result=dict(kernels=inspect(sass,log),negative_controls=negatives(sass,log),
                scope='CUDA-SM90-ASSEMBLY-NOT-PPU-EXECUTION')
    (a.out/'codegen.json').write_text(json.dumps(result,indent=2)+'\n')
    print('[S47 native] PASS six bodies; 9 negatives; device/numerics/performance NOT_RUN')


if __name__=='__main__':main()
