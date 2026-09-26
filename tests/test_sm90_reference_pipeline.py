"""Reference delivery's ordering/lifetimes, separate from speed admission."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]/'csrc/backends/sm90'


def admit(aux,state,kernel):
    loader=aux.split('void load_qkv(',1)[1].split('void load_scalar_alpha(',1)[0]
    ops=['kload.step(', 'qload.step(', 'vload.step(']
    assert [loader.index(x) for x in ops] == sorted(loader.index(x) for x in ops)
    body=aux.split('void compute_aux_safe(',1)[1]
    ops=['kkp.producer_acquire(kw)', 'solve.compute(', 'kkp.producer_commit(kw)',
         'qkp.producer_acquire(qw)', 'qkp.producer_commit(qw)', 'bp.consumer_release(br)']
    assert [body.index(x) for x in ops] == sorted(body.index(x) for x in ops)
    assert 'StateUsesBeta = !AuxInverse' in state
    assert 'if constexpr (StateUsesBeta) bp.consumer_wait(br)' in state
    assert 'if constexpr (StateUsesBeta) { bp.consumer_release(br); ++br; }' in state
    assert 'beta_pipeline_params.consumer_arv_count = BetaConsumerThreads;' in kernel
    assert '(CollectiveMainloop::StateUsesBeta ? NumStateMmaWarpGroups : 0)' in kernel
    assert 'if constexpr (NeedsBeta && CollectiveMainloop::StateUsesBeta)' in kernel


class ReferencePipeline(unittest.TestCase):
    def test_source_and_negative_lifetimes(self):
        aux=(ROOT/'scalar_gdn_aux.cuh').read_text()
        state=(ROOT/'scalar_gdn_state.cuh').read_text()
        kernel=(ROOT/'cula/kda/sm90/kernel/kernel_kda_fwd.hpp').read_text()
        admit(aux,state,kernel)
        plants=[
            (aux.replace('kkp.producer_commit(kw);',''),state,kernel),
            (aux,state.replace('if constexpr (StateUsesBeta) bp.consumer_wait(br)',
                               'bp.consumer_wait(br)'),kernel),
            (aux,state,kernel.replace('beta_pipeline_params.consumer_arv_count = BetaConsumerThreads;',
                                    'beta_pipeline_params.consumer_arv_count = 384;')),
        ]
        for a,s,k in plants:
            with self.assertRaises((AssertionError,ValueError)):admit(a,s,k)

    def test_five_stage_ring_all_interleavings(self):
        # Complete alpha consumers each own128 threads. Both the3-consumer
        # alpha ring and1-consumer beta ring wrap through all reachable orders.
        for consumers in (1,3):
            for chunks in range(1,17):
                seen=set();todo=[(0,)*(consumers+1)]
                while todo:
                    s=todo.pop()
                    if s in seen:continue
                    seen.add(s)
                    if all(x==chunks for x in s):continue
                    enabled=False;p=s[0]
                    if p<chunks and (p<5 or all(c>p-5 for c in s[1:])):
                        todo.append((p+1,*s[1:]));enabled=True
                    for i,c in enumerate(s[1:],1):
                        if c<p:
                            self.assertLessEqual(p-c,5)
                            nxt=list(s);nxt[i]+=1;todo.append(tuple(nxt));enabled=True
                    self.assertTrue(enabled,'nonterminal ring deadlock')


if __name__=='__main__':unittest.main()
