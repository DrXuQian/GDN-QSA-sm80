"""Declared denominator and actual-type admission for a stage-only search."""
from itertools import product

DENOMINATOR = 32


def space():
    rows = []
    for index in range(DENOMINATOR):
        rows.append(dict(id=index, stages=[2, 2+(index&1), 1+((index>>1)&1),
                    1+((index>>2)&1), 2+3*((index>>3)&1), 2+3*((index>>4)&1)]))
    validate(rows)
    return rows


def validate(rows):
    expected = set(product((2,), (2,3), (1,2), (1,2), (2,5), (2,5)))
    if (len(rows) != DENOMINATOR or {r['id'] for r in rows} != set(range(DENOMINATOR))
            or {tuple(r['stages']) for r in rows} != expected):
        raise ValueError('missing, duplicate or unsupported configuration in 32-cell denominator')


def admit_types(cell, actual):
    if len(actual) != 4 or {(r['gate_fp32'],r['initial']) for r in actual} != set(product((0,1),repeat=2)):
        raise ValueError('missing gate/initial-state specialization')
    if any(r['config'] != cell['id'] or r['stages'] != cell['stages'] for r in actual):
        raise ValueError('requested configuration does not match actual compiled stages')
    if any(not isinstance(r['shared_bytes'], int) or r['shared_bytes'] <= 0 for r in actual):
        raise ValueError('missing actual shared resource size')


def select_finalists(rows):
    """Top two measured event ratios per gate; NOT a speed verdict."""
    selected = set()
    for gate in ('-0.1','-1.0'):
        measured = [r for r in rows if r['state'] == 'SCREENED' and r['id'] != 0]
        ranked = sorted(measured, key=lambda r:r['screen'][gate]['candidate_over_control'])
        selected.update(r['id'] for r in ranked[:2])
    return sorted(selected)
