#!/usr/bin/env python3
"""S54: preserve complete S50 data/lifetime graph, remove only issue preference."""
import argparse
import json
from pathlib import Path
import subprocess
from check_sm90_independent_issue import check

ROOT=Path(__file__).resolve().parents[2]
PREFIX='csrc/backends/sm90/'
AUX=PREFIX+'scalar_gdn_aux.cuh'
ORDER=PREFIX+'ordered_pair.cuh'
INSERT='''// Only the aux-owned inverse has independent state-WG V-column owners.
// Stage retirement and joint output publication remain in their pipelines.
// This removes a tensor-issue preference, never a data dependency.
struct IndependentStateIssue {
    CUTE_DEVICE void init(int) {}
    CUTE_DEVICE void ordered_or_wait(int) {}
    CUTE_DEVICE void notify_next_blocked(int) {}
};

'''


def source(old,new):
    expected=dict(old)
    expected[ORDER]=old[ORDER].replace('namespace gdn::sm90 {\n\n',
                                     'namespace gdn::sm90 {\n\n'+INSERT,1)
    expected[AUX]=old[AUX].replace(
        'using OrderedMathBarriers = OrderedPair<Base::OrderedBarrierId0, Base::OrderedBarrierId1>;',
        'using OrderedMathBarriers = std::conditional_t<AuxInverse, IndependentStateIssue,\n'
        '        OrderedPair<Base::OrderedBarrierId0, Base::OrderedBarrierId1>>;',1)
    if new!=expected:raise ValueError('S50 data/lifetime source or exact policy boundary changed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path);p.add_argument('--candidate',type=Path)
    p.add_argument('--device-log',type=Path);a=p.parse_args()
    files=subprocess.check_output(['git','ls-tree','-r','--name-only','170f34f','--',PREFIX],cwd=ROOT,text=True).splitlines()
    old={f:subprocess.check_output(['git','show',f'170f34f:{f}'],cwd=ROOT,text=True) for f in files}
    new={str(f.relative_to(ROOT)):f.read_text() for f in (ROOT/PREFIX).rglob('*') if f.is_file()}
    source(old,new)
    for token in ('kp.consumer_release(kr);','warpgroup_wait<0>();','op.producer_commit(ow);','ap.consumer_wait(ar);'):
        plant=dict(new);key=PREFIX+'scalar_gdn_state.cuh'
        assert token in plant[key];plant[key]=plant[key].replace(token,'',1)
        try:source(old,plant)
        except ValueError:pass
        else:raise AssertionError('removed data dependency escaped')
    try:source(old,old)
    except ValueError:pass
    else:raise AssertionError('old policy escaped')
    result=dict(source='EXACT_S50_DATA_GRAPH',source_negatives=5)
    if a.candidate:
        log=a.device_log.read_text()
        if 'C7512' in log or 'C7510' in log:raise ValueError('async WGMMA serialized')
        parent,candidate=a.parent.read_text(),a.candidate.read_text()
        result['native']=check(parent,candidate)
        for plant in (parent,candidate.replace('WARPGROUP.DEPBAR.LE','REMOVED.DEPBAR',1),
                      candidate.replace('SYNCS.ARRIVE.TRANS64.RED.A1T0','REMOVED.ARRIVE',1)):
            try:check(parent,plant)
            except ValueError:pass
            else:raise AssertionError('native data dependency negative escaped')
        result['native_negatives']=3
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
