#!/usr/bin/env python3
"""S69: one exact conversion seam; paired-tail native lifetime retained."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from check_sm90_paired_tail import native as paired_native

ROOT = Path(__file__).resolve().parents[2]
FILE = 'csrc/backends/sm90/scalar_gdn_state.cuh'
OLD = 'auto operand_delta = kda::sm90::collective::make_acc_into_op<Element>(acc_delta,typename Base::TiledMmaKV::LayoutA_TV{});'
NEW = 'auto delta_bf16 = convert_fragment<Element>(acc_delta);\n            auto operand_delta = kda::sm90::collective::make_acc_into_op<Element>(delta_bf16,typename Base::TiledMmaKV::LayoutA_TV{});'


def bodies(text):
    result = {}
    for part in text.split('Function :')[1:]:
        symbol = part.splitlines()[0].strip()
        if 'FlatKernelTmaWarpSpecializedKdaFwd' not in symbol:
            continue
        if symbol in result:
            raise ValueError('duplicate body')
        result[symbol] = re.findall(r'/\*([0-9a-f]+)\*/\s*(?:@!?\w+\s+)?([A-Z][\w.]*)\s*(.*?)\s*;\s*/\*',part)
    return result


def source_check(text):
    old = subprocess.check_output(['git','show','170f34f:'+FILE], cwd=ROOT, text=True)
    if text != old.replace('#include "scalar_gdn_aux.cuh"', '#include "scalar_gdn_aux.cuh"\n#include "fragment_convert.cuh"').replace(OLD, NEW):
        raise ValueError('not the sole registered packed conversion')
    helper = ROOT/'csrc/backends/sm90/fragment_convert.cuh'
    # Exact S35 helper bytes, bound without requiring its unrelated branch
    # object to exist in a minimal S50->S69 source bundle.
    proven = '7efe6f6b05377e1214335523720efa822ced68a3e4ec64ce32ddeb7dec6d0d51'
    if hashlib.sha256(helper.read_bytes()).hexdigest() != proven:
        raise ValueError('changed RNE helper rather than reusing it')


def check(candidate, parent, original, log):
    result = paired_native(candidate, original, log)
    c, p = bodies(candidate), bodies(parent)
    if c.keys() != p.keys() or len(c) != 4:
        raise ValueError('four-body denominator changed')
    stacks = list(map(int,re.findall(r'(\d+) bytes stack frame',log)))
    if len(stacks) != 4 or max(stacks) > 104:
        raise ValueError('stack ceiling')
    for key in c:
        cc, pc = Counter(op for _,op,_ in c[key]), Counter(op for _,op,_ in p[key])
        if cc['F2F.BF16.F32'] != 0 or pc['F2F.BF16.F32'] != 128:
            raise ValueError('scalar conversion not removed')
        for prefix in ('HGMMA','HMMA','UTMA','WARPGROUP','USETMAXREG'):
            if {k:v for k,v in cc.items() if k.startswith(prefix)} != {k:v for k,v in pc.items() if k.startswith(prefix)}:
                raise ValueError('changed useful math or async batch: '+prefix)
        result[key]['packed_delta'] = dict(scalar_before=128,scalar_after=0,
                                          parent_sites=len(p[key]), candidate_sites=len(c[key]))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('candidate','parent','original','device-log'):
        p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args(); text=(ROOT/FILE).read_text();source_check(text)
    c,p,o,l=a.candidate.read_text(),a.parent.read_text(),a.original.read_text(),a.device_log.read_text()
    result=check(c,p,o,l)
    def red(fn):
        try:fn()
        except (ValueError,AssertionError):return
        raise AssertionError('negative escaped')
    red(lambda:source_check(text.replace(NEW,OLD)))
    red(lambda:source_check(text.replace('operand_scaled(i) =','operand_delta(i) =',1)))
    red(lambda:check(p,p,o,l))
    red(lambda:check(c.replace('gsb0, 0x0','gsb0, 0x1',1),p,o,l))
    red(lambda:check(c,p,o,l+' C7512'))
    print(json.dumps(dict(status='PASS',scope='EXACT_SOURCE_NATIVE_NOT_SPEED',bodies=result,negatives=5,
        hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (a.candidate,a.parent,a.original,a.device_log)}),indent=2))


if __name__=='__main__':main()
