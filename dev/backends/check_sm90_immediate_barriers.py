#!/usr/bin/env python3
"""S14 codegen postcondition, not a GPU synchronization/correctness proof."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from check_sm90_binary import inspect as inspect_image


def inspect(sass):
    admitted = inspect_image(sass)  # All four real, nonempty specializations.
    parts = re.split(r"Function\s*:\s*(\S+)", sass)
    bodies = dict(zip(parts[1::2], parts[2::2]))
    result = {}
    for symbol in admitted:
        calls = re.findall(
            r"\bBAR\.(SYNC(?:\.DEFER_BLOCKING)?|ARV)\s+([^;]+);", bodies[symbol])
        state = [(op, [v.strip() for v in args.split(',')]) for op, args in calls
                 if args.strip().endswith(', 0x100')]
        counts = Counter()
        for op, args in state:
            if len(args) != 2 or args[0] not in ('0x4', '0x5'):
                raise ValueError('state barrier has a runtime/noncanonical ID operand')
            counts[(op.split('.')[0], args[0])] += 1
        required = {(op, ident) for op in ('SYNC', 'ARV') for ident in ('0x4', '0x5')}
        if set(counts) != required:
            raise ValueError('missing state wait/notify barrier ID')
        result[symbol] = {f'{op}:{ident}': n for (op, ident), n in sorted(counts.items())}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sass', type=Path)
    p.add_argument('--old-sass', type=Path, required=True)
    args = p.parse_args()
    current = args.sass.read_text()
    result = inspect(current)
    try:
        inspect(args.old_sass.read_text())
    except ValueError as exc:
        if 'runtime/noncanonical ID' not in str(exc):
            raise
    else:
        raise AssertionError('old native image escaped the S14 mechanism gate')
    plant = re.sub(r'(BAR\.SYNC\.DEFER_BLOCKING\s+)0x[45](, 0x100)',
                   r'\g<1>R200\2', current, count=1)
    assert plant != current
    try:
        inspect(plant)
    except ValueError as exc:
        assert 'runtime/noncanonical ID' in str(exc)
    else:
        raise AssertionError('a runtime-ID regression escaped')
    print(json.dumps(dict(kernels=result, negative_controls=2,
        sass_sha256=hashlib.sha256(current.encode()).hexdigest(),
        scope='STATIC-NATIVE-OPERANDS-NOT-TIMING-OR-RACE-PROOF'), indent=2))


if __name__ == '__main__':
    main()
