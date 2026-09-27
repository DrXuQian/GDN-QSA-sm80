#!/usr/bin/env python3
"""S18 native postcondition: issue preference removed, data machinery retained."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from check_sm90_binary import inspect


def bodies(sass):
    inspect(sass)  # Real target, all four actual bodies, live state/output.
    fields = re.split(r'Function\s*:\s*(\S+)', sass)
    return dict(zip(fields[1::2], fields[2::2]))


def counts(body):
    ops = re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?(\w+(?:\.\w+)*)', body)
    return Counter(op for op in ops if op.startswith(('HGMMA', 'HMMA', 'WARPGROUP', 'UTMA', 'SYNCS')))


def check(control, candidate):
    old, new = bodies(control), bodies(candidate)
    if old.keys() != new.keys():
        raise ValueError('different actual specialization denominator')
    results = {}
    for name in old:
        pattern = r'\bBAR\.(?:SYNC(?:\.DEFER_BLOCKING)?|ARV)\s+[^;]*,\s*0x100\s*;'
        old_issue = re.findall(pattern, old[name])
        new_issue = re.findall(pattern, new[name])
        if not old_issue or new_issue:
            raise ValueError('two-WG issue protocol was not removed')
        if counts(old[name]) != counts(new[name]):
            raise ValueError('matrix/data-completion/stage-reuse machinery changed')
        results[name] = {'old_issue_sites': len(old_issue), 'new_issue_sites': 0,
                         'fixed_native_families': dict(counts(new[name]))}
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('control', type=Path)
    p.add_argument('candidate', type=Path)
    a = p.parse_args()
    old, new = a.control.read_text(), a.candidate.read_text()
    result = check(old, new)
    for plant in (old, new.replace('WARPGROUP.DEPBAR.LE', 'REMOVED.DEPBAR', 1),
                  new.replace('SYNCS.ARRIVE.TRANS64.RED.A1T0', 'REMOVED.ARRIVE', 1)):
        try:
            check(old, plant)
        except ValueError:
            continue
        raise AssertionError('old protocol or removed data dependency escaped')
    print(json.dumps({'bodies': result, 'negative_controls': 3,
                      'scope': 'static CUDA native; not device-race proof'}, indent=2))


if __name__ == '__main__':
    main()
