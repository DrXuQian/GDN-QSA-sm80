#!/usr/bin/env python3
"""Bind the last-level inverse change, its two rounded partials and native work."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
from check_sm90_binary import inspect

ROOT=Path(__file__).resolve().parents[2]
WIRE='csrc/backends/sm90/cula/kerutils/device/sm80/collective_inverse.hpp'
HELPER='csrc/backends/sm90/inverse_local_reduction.cuh'


def source(old,wire,helper):
    call='''gdn::sm90::inverse64_local_reduction<Element>(sT,wg_sync_named_barrier_id_,
        [](auto const& acc,auto const& mma) __attribute__((always_inline)) {
          return detail::SM80::make_acc_into_op<Element>(acc,mma);
        });'''
    expected=old.replace('#include "kerutils/common/cute_ext.hpp"',
        '#include "kerutils/common/cute_ext.hpp"\n#include "inverse_local_reduction.cuh"',1).replace(
        'blockwise_diagonal_inversed_32x32_to_64x64(sT);',call,1)
    if wire!=expected:raise ValueError('anything besides exact FP16 last-level call changed')
    for token in ('if(warp<2)', 'gemm(first,d,c,dc);', 'transform(dc,[](auto x){return -x;});',
                  'gemm(second,dc_half(_,_,_0{}),a(_,_,_0{}),low);',
                  'gemm(second,dc_half(_,_,_1{}),a(_,_,_1{}),high);',
                  'output(i)=Element(high(i))+Element(low(i));'):
        if token not in helper:raise ValueError('wrong participant/dot/rounding: '+token)
    barrier='cutlass::arch::NamedBarrier::arrive_and_wait(128,barrier_id);'
    if helper.count(barrier)!=1:raise ValueError('input alias barrier count wrong')
    if not helper.index('gemm(second,dc_half(_,_,_1{})')<helper.index(barrier)<helper.index('copy(copy_o,'):
        raise ValueError('aliased C overwritten before all readers finish')


def native(cand,parent,log):
    if 'C7512' in log or 'C7510' in log:raise ValueError('serialized state WGMMA')
    c,p=inspect(cand),inspect(parent)
    if c.keys()!=p.keys():raise ValueError('specialization inventory changed')
    def bodies(s):
        pieces=re.split(r'Function\s*:\s*(\S+)',s)
        return {k:Counter(re.findall(r'/\*[0-9a-f]+\*/\s*(?:@!?\w+\s+)?([A-Z][\w.]*)',v))
                for k,v in zip(pieces[1::2],pieces[2::2])}
    cb,pb=bodies(cand),bodies(parent);results={}
    for key in c:
        a,b=cb[key],pb[key]
        for op in ('HGMMA','UTMALDG','UTMASTG','STG','state_stores','tail_stores'):
            if c[key][op]!=p[key][op]:raise ValueError('unchanged matrix/delivery path differs')
        for prefix in ('WARPGROUP','UTMA'):
            if {o:n for o,n in a.items() if o.startswith(prefix)}!={o:n for o,n in b.items() if o.startswith(prefix)}:
                raise ValueError('state async/delivery protocol differs')
        if a['HMMA.16816.F32']-b['HMMA.16816.F32']!=16 or a['HMMA.1688.F32']!=b['HMMA.1688.F32']:
            raise ValueError('two-warp/full-row last-level work not emitted')
        if b['BAR.SYNC.DEFER_BLOCKING']-a['BAR.SYNC.DEFER_BLOCKING']!=2:
            raise ValueError('full/tail partial-publication barriers remain')
        if b['STSM.16.M88.4']-a['STSM.16.M88.4']!=4:
            raise ValueError('partial shared publication remains')
        if a['HADD2']!=b['HADD2']:raise ValueError('FP16 partial addition changed')
        results[key]=dict(lastlevel_warps=(4,2),lastlevel_hmma_per_warp=(8,16),
            useful_lastlevel_hmma_per_CTA=32,native_counts=c[key],
            barrier_delta=-2,partial_store_delta=-4)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path);parser.add_argument('--parent',type=Path)
    parser.add_argument('--device-log',type=Path);args=parser.parse_args()
    old=subprocess.check_output(['git','show',f'06b7471:{WIRE}'],cwd=ROOT,text=True)
    wire=(ROOT/WIRE).read_text();helper=(ROOT/HELPER).read_text();source(old,wire,helper)
    for w,h in ((old,helper),(wire,helper.replace('if(warp<2)','if(warp<1)')),
        (wire,helper.replace('dc_half(_,_,_1{}),a(_,_,_1{})','dc_half(_,_,_0{}),a(_,_,_1{})')),
        (wire,helper.replace('Element(high(i))+Element(low(i))','Element(high(i)+low(i))')),
        (wire,helper.replace('cutlass::arch::NamedBarrier::arrive_and_wait(128,barrier_id);',''))):
        try:source(old,w,h)
        except ValueError:pass
        else:raise AssertionError('source inverse negative escaped')
    result=dict(source='PASS',source_negatives=5,
                helper_sha256=hashlib.sha256(helper.encode()).hexdigest(),scope='NATIVE_NOT_SPEED')
    if args.candidate:
        c,p,l=args.candidate.read_text(),args.parent.read_text(),args.device_log.read_text()
        result['native']=native(c,p,l)
        for image,log in ((p,l),(c,l+' C7512'),(c.replace('HMMA.16816','REMOVED.16816',1),l)):
            try:native(image,p,log)
            except ValueError:pass
            else:raise AssertionError('native inverse negative escaped')
        result['native_negatives']=3
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
