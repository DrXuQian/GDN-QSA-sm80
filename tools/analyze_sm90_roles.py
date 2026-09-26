"""Diagnostic leader timelines, not a decomposition of uninstrumented time."""
from statistics import median

CAPACITY = (64, 64, 3, 16)
POINTS = (9, 13, 13)
LABELS = (
    ("K-wait", "KK-issue+Q-wait", "QK-issue+both-retire", "alpha-beta-wait",
     "scalar+slot-acquire", "shared-publish+sync", "inverse+sync", "cast+commit+release"),
    ("alpha-Q-wait", "O1+Q-release", "K-wait+SK", "V-wait", "residual",
     "inverse-beta-wait", "newV+release", "delta-cast+QK-wait", "O2+release",
     "output-slot+store", "alpha-last-wait", "decay+update+release"),
)


def index(cta, chunk, role, point):
    return ((cta*64+chunk)*3+role)*16+point


def analyze(data, ctas, chunks):
    assert len(data) == 64*64*3*16, "trace capacity denominator"
    assert 0 < ctas <= 64 and 0 < chunks <= 64
    expected = ctas*chunks*sum(POINTS)
    assert sum(value != 0 for value in data) == expected, "nonzero trace denominator"
    for cta in range(64):
        for chunk in range(64):
            for role, points in enumerate(POINTS):
                row = data[index(cta,chunk,role,0):index(cta,chunk,role,0)+16]
                valid = cta < ctas and chunk < chunks
                assert all((v != 0) == (valid and p < points) for p,v in enumerate(row)), "trace owner/sentinel"
                if valid:
                    assert row[:points] == sorted(row[:points]), "non-monotonic role timestamps"
                    if chunk:
                        assert row[0] >= data[index(cta,chunk-1,role,points-1)], "chunk order"
    roles = {}
    for role, points in enumerate(POINTS):
        sums = [[] for _ in range(points-1)]
        spans, cadence = [], []
        for cta in range(ctas):
            spans.append((data[index(cta,chunks-1,role,points-1)]-data[index(cta,0,role,0)])/1000)
            for p in range(points-1):
                sums[p].append(sum(data[index(cta,c,role,p+1)]-data[index(cta,c,role,p)] for c in range(chunks))/1000)
            for c in range(2, chunks-1):
                cadence.append((data[index(cta,c,role,0)]-data[index(cta,c-1,role,0)])/1000)
        roles[str(role)] = dict(span_us=dict(median=median(spans),range=[min(spans),max(spans)]),
            steady_chunk_cadence_us=median(cadence) if cadence else None,
            interval_sum_us={label:median(samples) for label,samples in zip(LABELS[0 if role==0 else 1],sums)})
    # Same CTA/chunk only. Aux point7 is inverse retired, not complete publication.
    lag = [(data[index(cta,c,role,5)]-data[index(cta,c,0,7)])/1000
           for cta in range(ctas) for c in range(1,chunks-1) for role in (1,2)]
    return dict(expected_stamps=expected,roles=roles,
        state_inverse_wait_entry_minus_aux_inverse_done_us=dict(median=median(lag),range=[min(lag),max(lag)]) if lag else None,
        scope="INSTRUMENTED_LEADER_INTERVALS_INCLUDE_PROBES_AND_CONTENTION_NOT_ADDITIVE_ACROSS_ROLES")
