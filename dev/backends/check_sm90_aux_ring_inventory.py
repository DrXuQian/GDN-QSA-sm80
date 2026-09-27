#!/usr/bin/env python3
"""Close the registered four-tuple denominator, not just each individual build."""
import argparse
import copy
import json
from pathlib import Path

from check_sm90_aux_rings import admit_types


def admit_inventory(receipts):
    if len(receipts) != 4 or {r['config'] for r in receipts} != set(range(4)):
        raise ValueError('missing/duplicate registered QK/KK configuration')
    for row in receipts:
        config = row['config']
        admit_types(row['types'], config)
        if row['status'] != 'PASS' or row['native']['bodies'] != 4:
            raise ValueError('individual native/type gate failed')
        if config == 0 and not row['native']['default_native_identical']:
            raise ValueError('default native control differs')
        if set(row['progress']) != {str(n) for n in range(1, 9)}:
            raise ValueError('progress denominator changed')
    return {'configurations': 4, 'actual_types': 16, 'native_bodies': 16}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipts', nargs='+', type=Path)
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in args.receipts]
    counts = admit_inventory(rows)
    missing_type = copy.deepcopy(rows)
    missing_type[0]['types'].pop()
    plants = [rows[:-1], rows[:-1] + rows[:1], missing_type]
    for plant in plants:
        try:
            admit_inventory(plant)
        except ValueError:
            continue
        raise AssertionError('inventory negative escaped')
    print(json.dumps({'status': 'PASS', **counts, 'negatives': 3,
                      'scope': 'LOCAL_INVENTORY_NOT_PERFORMANCE'}))


if __name__ == '__main__':
    main()
