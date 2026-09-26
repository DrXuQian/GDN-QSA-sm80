#!/usr/bin/env python3
"""S48 must not change the native preparation instructions while varying state."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def bodies(path):
    pieces=re.split(r'Function\s*:\s*(\S+)',Path(path).read_text())
    selected={}
    for name,body in zip(pieces[1::2],pieces[2::2]):
        if 'prepare_aux_device' not in name:continue
        lines=re.findall(r'/\*[0-9a-f]+\*/\s+([^\n]*?;)',body)
        selected[name]='\n'.join(re.sub(r'\s+',' ',line.strip()) for line in lines)
    if len(selected)!=2:raise ValueError('missing gate variant in preparation')
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate');p.add_argument('control')
    a=p.parse_args();got=bodies(a.candidate);want=bodies(a.control)
    if got!=want:raise ValueError('prepared native sequence changed')
    print(json.dumps(dict(status='PASS',prepare_bodies=2,
        sequence_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in got.items()}),indent=2))


if __name__=='__main__':main()
