#!/usr/bin/env python3
"""Reject S59 if the explicit PTX branch is if-converted in any aux clone."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from check_sm90_binary import inspect


def bodies(sass):
    inspect(sass)
    p = re.split(r'Function\s*:\s*(\S+)', sass)
    return {name: [(int(pc, 16), row) for pc, row in
            re.findall(r'/\*([0-9a-f]+)\*/\s*(.*?)\s*;\s*/\*', body)]
            for name, body in zip(p[1::2], p[2::2])}


def inspect_batches(sass):
    result = {}
    for name, rows in bodies(sass).items():
        lookup = {pc: i for i, (pc, _) in enumerate(rows)}
        batches = []
        for i, (pc, row) in enumerate(rows):
            branch = re.fullmatch(r'@!?P\d+ BRA (0x[0-9a-f]+)', row)
            if not branch or int(branch[1], 16) not in lookup:
                continue
            j = lookup[int(branch[1], 16)]
            slow = [r for _, r in rows[i+1:j]]
            count = sum('MUFU.EX2' in r for r in slow)
            if count not in (6, 8) or sum('-126' in r for r in slow) != count:
                continue
            merge = re.fullmatch(r'BRA (0x[0-9a-f]+)', rows[j-1][1])
            if not merge or int(merge[1], 16) not in lookup:
                continue
            k = lookup[int(merge[1], 16)]
            fast = [r for _, r in rows[j:k]]
            if (sum('MUFU.EX2' in r for r in fast) != count or
                    any('-126' in r or r.startswith(('LDL', 'STL')) for r in fast)):
                raise ValueError('fast path still compares or uses local tuple')
            batches.append(dict(branch=hex(pc), fast=hex(rows[j][0]),
                                merge=hex(rows[k][0]), values=count))
        # Two 32-element clones, four tuple calls each. Constant diagonals may
        # eliminate two EX2 values, not an entire dynamic tuple branch.
        result[name] = batches
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--device-log',type=Path,required=True)
    a=p.parse_args()
    c,b=a.candidate.read_text(),a.parent.read_text()
    got=inspect_batches(c)
    print(json.dumps(dict(bodies=got,expected_batches_per_body=8),indent=2))
    if 'C7512' in a.device_log.read_text(): raise ValueError('WGMMA serialization')
    if any(len(v)!=8 for v in got.values()):
        raise ValueError('actual native batch branches incomplete; if-conversion NOT admitted')
    def counts(s):
        return {name: Counter(re.sub(r'^@!?\w+\s+', '', r).split()[0]
            for _,r in rows if re.sub(r'^@!?\w+\s+', '', r).startswith(
                ('HMMA','HGMMA','UTMA','SYNCS','BAR.','WARPGROUP')))
            for name,rows in bodies(s).items()}
    if counts(c)!=counts(b): raise ValueError('matrix or data protocol changed')


if __name__=='__main__': main()
