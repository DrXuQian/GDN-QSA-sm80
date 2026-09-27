#!/usr/bin/env python3
"""S61 exact source change and all-body inverse publication postconditions."""
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
        '#include "kerutils/common/cute_ext.hpp"\n#include "inverse_diagonal_owner.cuh"')
    old_first='''    if (thread_idx < 64) {  // compute 8x8 inverse on diagnal directly
      compute_diagonal_inverse_NxN<8>(t8X8sT(_, _, thread_idx / 8, thread_idx / 8), thread_idx % 8);
    }

    cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup, wg_sync_named_barrier_id_);'''
    new_first='''    auto owner = gdn::sm90::inverse_diagonal_owner(thread_idx);
    compute_diagonal_inverse_NxN<8>(
        t8X8sT(_, _, owner.tile, owner.tile), owner.row, owner.publishes);
    __syncwarp();'''
    half=half.replace(old_first,new_first).replace(
        'compute_diagonal_inverse_NxN(TensorT&& mat, int tid_in_group) {',
        'compute_diagonal_inverse_NxN(TensorT&& mat, int tid_in_group, bool publishes) {').replace(
        '    store_row(tid_in_group, row);','    if (publishes) store_row(tid_in_group, row);')
    def clean(s):return re.sub(r'\s+','',re.sub(r'//[^\n]*','',s))
    if clean(new)!=clean(prefix+'struct CollectiveInverse {'+half):
        raise ValueError('not the exact ownership/fence change')


def bodies(sass):
    inspect(sass);s=re.split(r'Function\s*:\s*(\S+)',sass)
    return {name:re.findall(r'/\*([0-9a-f]+)\*/\s*(.*?)\s*;\s*/\*',body)
            for name,body in zip(s[1::2],s[2::2])}


def native(old,new):
    a,b=bodies(old),bodies(new)
    if a.keys()!=b.keys():raise ValueError('kernel denominator changed')
    def opcode(r):return re.sub(r'^@!?\w+\s+','',r).split()[0]
    def fixed(rows):return Counter(opcode(r) for _,r in rows
        if opcode(r).startswith(('HMMA','HGMMA','UTMA','SYNCS','BAR.','WARPGROUP')))
    result={}
    for name,rows in b.items():
        expected=fixed(a[name]);expected['BAR.SYNC.DEFER_BLOCKING']-=2
        if fixed(rows)!=expected:raise ValueError('only two first-level CTA barriers may disappear')
        first=next(i for i,(_,r) in enumerate(rows) if 'BAR.SYNC' in r and '0xd, 0x80' in r)
        stop=next(i for i in range(first+1,len(rows)) if 'HMMA.' in rows[i][1])
        diag=rows[first:stop]
        counts=Counter(opcode(r) for _,r in diag)
        if counts['FFMA']!=21 or counts['SHFL.IDX']!=21 or counts['BAR.SYNC.DEFER_BLOCKING']!=1:
            raise ValueError('first diagonal math/order or CTA barrier changed')
        stores=[i for i,(_,r) in enumerate(diag) if 'STS.128' in r]
        loads=[i for i,(_,r) in enumerate(diag) if 'LDSM.' in r]
        if len(stores)!=1 or not loads or stores[0]>=min(loads):
            raise ValueError('diagonal publication must precede its merge reads')
        result[name]=dict(first_diagonal_sites=len(diag),first_diagonal=dict(counts),
            full_math_data=dict(fixed(rows)),
            explicit_native_warpsync_in_first_diagonal=counts['WARPSYNC.ALL'])
    return result


def warp_fence(ptx,source_text):
    marker=re.search(r'\.file\s+(\d+)\s+"[^"]*/collective_inverse.hpp"',ptx)
    if not marker:raise ValueError('PTX source identity missing')
    line=next(i for i,s in enumerate(source_text.splitlines(),1) if s.strip()=='__syncwarp();')
    pattern=(r'inlined_at\s+'+marker[1]+r' '+str(line)+
             r' \d+\s*\n\s*bar.warp.sync\s+-1;')
    parts=re.split(r'(?m)^\s*(?:\.visible\s+)?\.entry\s+([^\s(]+)\s*\(',ptx)
    entries=dict(zip(parts[1::2],parts[2::2]))
    if len(entries)!=4 or any(len(re.findall(pattern,body))!=2 for body in entries.values()):
        raise ValueError('need full/tail first-inverse fence in each of four PTX entries')
    return pattern


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--device-log',type=Path,required=True);p.add_argument('--ptx',type=Path,required=True)
    a=p.parse_args()
    old=subprocess.check_output(['git','show',f'170f34f:{FILE}'],cwd=ROOT,text=True)
    new=(ROOT/FILE).read_text();source(old,new)
    for bad in (old,new.replace('__syncwarp();','',1),new.replace('if (publishes) store_row','store_row',1)):
        try:source(old,bad)
        except ValueError:continue
        raise AssertionError('source ownership/fence negative escaped')
    text=a.ptx.read_text();pattern=warp_fence(text,new)
    planted,n=re.subn(pattern,lambda m:m[0].replace('bar.warp.sync','REMOVED.warp.sync'),text,count=1)
    if n!=1:raise AssertionError('PTX fence negative not planted')
    try:warp_fence(planted,new)
    except ValueError:pass
    else:raise AssertionError('deleted exact PTX fence escaped')
    if 'C7512' in a.device_log.read_text():raise ValueError('WGMMA serialization')
    old,new=a.parent.read_text(),a.candidate.read_text();result=native(old,new)
    for bad in (old,new.replace('HMMA.','REMOVED.',1),new.replace('SYNCS.','REMOVED.',1)):
        try:native(old,bad)
        except ValueError:continue
        raise AssertionError('native negative escaped')
    print(json.dumps(dict(status='PASS',bodies=result,source_negatives=3,native_negatives=3,ptx_negatives=1,
        warp_fence='CUDA/PTX retained; compiler may elide redundant native WARPSYNC; no explicit-opcode claim'),indent=2))


if __name__=='__main__':main()
