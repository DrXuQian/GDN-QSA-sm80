#!/usr/bin/env python3
"""Bind warp-local inverse prefix, preserving every helper's arithmetic."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
from check_sm90_binary import inspect

ROOT=Path(__file__).resolve().parents[2]
FILE='csrc/backends/sm90/cula/kerutils/device/sm80/collective_inverse.hpp'


def source(old,new):
    prefix,half=old.split('struct CollectiveInverse {',1)
    prefix=prefix.replace('#include "kerutils/common/cute_ext.hpp"',
        '#include "kerutils/common/cute_ext.hpp"\n#include "inverse_prefix_owner.cuh"')
    start=half.index('    if (thread_idx < 64) {  // compute 8x8 inverse')
    end=half.index('\n    cutlass::arch::NamedBarrier::arrive_and_wait',
                   half.index('      blockwise_diagonal_inversed_16x16_to_32x32',start))
    replacement='''    if (thread_idx < 64) {
      compute_diagonal_inverse_NxN<8>(t8X8sT(_, _, thread_idx / 8, thread_idx / 8), thread_idx % 8);
      __syncwarp();
      auto t16X16sT = flat_divide(sT, Shape<_16, _16>{});
      CUTE_UNROLL
      for (int half=0; half<2; ++half) {
        int tile=gdn::sm90::inverse_prefix_tile16(thread_idx / 32, half);
        blockwise_diagonal_inversed_8x8_to_16x16(t16X16sT(_, _, tile, tile));
      }
      __syncwarp();
      auto t32X32sT = flat_divide(sT, Shape<_32, _32>{});
      blockwise_diagonal_inversed_16x16_to_32x32(t32X32sT(_, _, thread_idx / 32, thread_idx /32));
    }'''
    expected=prefix+'struct CollectiveInverse {'+half[:start]+replacement+half[end:]
    def plain(s):return re.sub(r'\s+','',re.sub(r'//[^\n]*','',s))
    if plain(expected)!=plain(new):raise ValueError('unregistered inverse arithmetic/ownership/fence change')
    for file in ('scalar_gdn_aux.cuh','scalar_gdn_state.cuh'):
        rel='csrc/backends/sm90/'+file
        if (ROOT/rel).read_bytes()!=subprocess.check_output(['git','show','170f34f:'+rel],cwd=ROOT):
            raise ValueError('unrelated auxiliary/state change')


def native(old,new,log):
    if 'C7512' in log or 'C7510' in log:raise ValueError('serialized WGMMA')
    a,b=inspect(old),inspect(new)
    if a.keys()!=b.keys():raise ValueError('specialization denominator changed')
    def bodies(text):
        p=re.split(r'Function\s*:\s*(\S+)',text)
        return {key:re.findall(r'/\*[0-9a-f]+\*/\s*(.*?)\s*;\s*/\*',part)
                for key,part in zip(p[1::2],p[2::2])}
    def fixed(rows):
        ops=[re.sub(r'^@!?\w+\s+','',r).split()[0] for r in rows]
        return Counter(op for op in ops if op.startswith(('HMMA','HGMMA','UTMA','SYNCS','BAR.','WARPGROUP')))
    parent,candidate=bodies(old),bodies(new);result={}
    for key,rows in candidate.items():
        expected=fixed(parent[key]);expected['BAR.SYNC.DEFER_BLOCKING']-=4;expected['HMMA.1688.F32']+=4
        if fixed(rows)!=expected:raise ValueError('unexpected matrix/data/barrier delta')
        result[key]=dict(sites=len(rows),counts=dict(fixed(rows)))
    return result


def fences(ptx,text):
    marker=re.search(r'\.file\s+(\d+)\s+"[^"]*/collective_inverse.hpp"',ptx)
    if not marker:raise ValueError('missing PTX source identity')
    lines=[i for i,s in enumerate(text.splitlines(),1) if s.strip()=='__syncwarp();']
    if len(lines)!=2:raise ValueError('expected two prefix warp fences')
    parts=re.split(r'(?m)^\s*(?:\.visible\s+)?\.entry\s+([^\s(]+)\s*\(',ptx)
    entries=dict(zip(parts[1::2],parts[2::2]));patterns=[]
    for line in lines:
        pattern=r'inlined_at\s+'+marker[1]+r' '+str(line)+r' \d+\s*\n\s*bar.warp.sync\s+-1;'
        if len(entries)!=4 or any(len(re.findall(pattern,body))!=2 for body in entries.values()):
            raise ValueError('missing full/tail source-bound warp fence')
        patterns.append(pattern)
    return patterns


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('candidate','parent','device-log','ptx'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--out',type=Path);a=p.parse_args()
    old=subprocess.check_output(['git','show','170f34f:'+FILE],cwd=ROOT,text=True)
    text=(ROOT/FILE).read_text();source(old,text)
    for plant in (old,text.replace('__syncwarp();','',1),text.replace('half<2','half<1',1)):
        try:source(old,plant)
        except ValueError:pass
        else:raise AssertionError('source negative escaped')
    before,after,log=a.parent.read_text(),a.candidate.read_text(),a.device_log.read_text()
    result=native(before,after,log)
    for plant,plog in ((before,log),(after,log+' C7512'),(after.replace('HMMA.1688','REMOVED.1688',1),log)):
        try:native(before,plant,plog)
        except ValueError:pass
        else:raise AssertionError('native negative escaped')
    ptx=a.ptx.read_text();patterns=fences(ptx,text)
    for pattern in patterns:
        plant,n=re.subn(pattern,lambda m:m[0].replace('bar.warp.sync','REMOVED.warp.sync'),ptx,count=1)
        if n!=1:raise AssertionError('fence negative not planted')
        try:fences(plant,text)
        except ValueError:pass
        else:raise AssertionError('PTX exact-fence negative escaped')
    output=json.dumps(dict(status='PASS',bodies=result,source_negatives=3,native_negatives=3,
        ptx_negatives=2,warp_fence='CUDA/PTX retained; no one-for-one native WARPSYNC claim'),indent=2)+'\n'
    if a.out:a.out.write_text(output)
    print(output,end='')


if __name__=='__main__':main()
