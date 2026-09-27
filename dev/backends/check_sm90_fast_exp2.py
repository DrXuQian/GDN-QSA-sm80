#!/usr/bin/env python3
"""Admit the generated exp2-only experiment, not a global fastmath build."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from check_sm90_packed_newv import bodies


def identity(manifest, fast):
    flags = manifest['flags']
    expected = 'fastmath-exp2-only' if fast else 'standard-exp2'
    if manifest.get('exp2_mode') != expected:
        raise ValueError('exp2 mode identity differs')
    if ('-DGDN_SM90_FAST_EXP2=1' in flags) != fast:
        raise ValueError('macro and identity disagree')
    if any(x.startswith(('--use_fast_math', '--ftz', '--prec-div', '--fmad')) for x in flags):
        raise ValueError('global arithmetic flags are not this experiment')


def native(fast, standard, parent, log):
    f, s, p = map(bodies, (fast, standard, parent))
    if len(f) != 4 or f.keys() != s.keys() or s.keys() != p.keys():
        raise ValueError('four-body specialization denominator changed')
    if s != p:
        raise ValueError('default mode changed native instruction/operand/PC stream')
    stacks = list(map(int, re.findall(r'(\d+) bytes stack frame', log)))
    if 'C7512' in log or len(stacks) != 4 or max(stacks) > 104:
        raise ValueError('native WGMMA/resource admission failed')
    result = []
    for symbol in f:
        fc, sc = (Counter(op for _, op, _ in x[symbol]) for x in (f, s))
        for prefix in ('HGMMA', 'HMMA', 'UTMA', 'WARPGROUP', 'USETMAXREG', 'BAR.', 'SYNCS'):
            if {k:v for k,v in fc.items() if k.startswith(prefix)} != {
                    k:v for k,v in sc.items() if k.startswith(prefix)}:
                raise ValueError('matrix/pipeline/roles changed: '+prefix)
        correction = lambda rows: sum(op.startswith('FSETP') and '-126' in args
                                     for _,op,args in rows)
        if correction(s[symbol]) == 0 or correction(f[symbol]) != 0:
            raise ValueError('actual standard-exp2 correction not removed')
        if fc['MUFU.EX2'] != sc['MUFU.EX2']:
            raise ValueError('exp2 useful evaluations changed')
        result.append(dict(symbol=symbol, standard_sites=len(s[symbol]),
            fast_sites=len(f[symbol]), standard_correction=correction(s[symbol]),
            fast_correction=correction(f[symbol]), standard_counts=sc, fast_counts=fc))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('fast', 'standard', 'parent'):
        p.add_argument('--'+key, type=Path, required=True, help='build directory')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    manifest = json.loads((a.fast/'build.json').read_text())
    control = json.loads((a.standard/'build.json').read_text())
    identity(manifest, True); identity(control, False)
    f,s,b = ((d/'codegen/image.sass').read_text() for d in (a.fast,a.standard,a.parent))
    log = (a.fast/'device.log').read_text()
    report = native(f,s,b,log)
    def red(fn):
        try: fn()
        except ValueError: return
        raise AssertionError('negative escaped')
    red(lambda: native(s,s,b,log))
    red(lambda: native(f.replace('Function :', 'Missing :', 1),s,b,log))
    red(lambda: identity(manifest | {'flags': [x for x in manifest['flags'] if 'FAST_EXP2' not in x]}, True))
    red(lambda: identity(manifest | {'flags': manifest['flags']+['--use_fast_math']}, True))
    red(lambda: native(f,s,b,log+' C7512'))
    result = dict(status='PASS', scope='NATIVE_NOT_NUMERIC_OR_SPEED', bodies=report,
                  negatives=5, stacks=list(map(int,re.findall(r'(\d+) bytes stack frame',log))),
                  hashes={str(d):hashlib.sha256((d/'launch.o').read_bytes()).hexdigest()
                          for d in (a.fast,a.standard,a.parent)})
    text = json.dumps(result, indent=2)+'\n'
    if a.out:
        with a.out.open('x') as out: out.write(text)
    print(text, end='')


if __name__ == '__main__': main()
