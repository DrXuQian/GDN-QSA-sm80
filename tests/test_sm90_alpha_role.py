"""Producer-role source contracts; actual device tests still prove execution."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]/'csrc/backends/sm90'
AUX=(ROOT/'scalar_gdn_aux.cuh').read_text()
KERNEL=(ROOT/'cula/kda/sm90/kernel/kernel_kda_fwd.hpp').read_text()


def admit(aux,kernel):
    if 'SeparateScalarGateProducer = true' not in aux:
        raise ValueError('separate producer not selected')
    qkv=aux.split('CUTE_DEVICE void load_qkv(',1)[1].split('CUTE_DEVICE void load_alpha_and_last(',1)[0]
    if 'load_scalar_gate<' in qkv:
        raise ValueError('scalar prefix remains on QKV warp')
    alpha=aux.split('CUTE_DEVICE void load_alpha_and_last(',1)[1].split('CUTE_DEVICE void compute_aux_safe(',1)[0]
    events=['load_scalar_gate<64,128>','ap.consumer_wait(ar)',
            'lp.producer_acquire(lw)','lp.producer_commit(lw)',
            'ap.consumer_release(ar)']
    places=[alpha.index(item) for item in events]
    if places!=sorted(places):raise ValueError('prefix/last lifetime reordered')
    if 'NeedsAlpha && !CollectiveMainloop::SeparateScalarGateProducer' not in kernel:
        raise ValueError('QKV warp still claims alpha producer')
    if 'alpha_pipeline_params.role = MainloopAlphaPipeline::ThreadCategory::ProducerConsumer;' not in kernel:
        raise ValueError('alpha warp must produce and consume')
    if 'alpha_pipeline_params.consumer_arv_count = NumStateMathThreads + NumAuxMathThreads + cutlass::NumThreadsPerWarp;' not in kernel:
        raise ValueError('416-thread alpha consumer denominator changed')


class AlphaRole(unittest.TestCase):
    def test_real_role_contract(self):admit(AUX,KERNEL)
    def test_four_lifetime_and_role_plants(self):
        plants=[
            (AUX.replace('qload.step(qsd,block,qw,leader);','load_scalar_gate<64,128>();'),KERNEL),
            (AUX,KERNEL.replace('ThreadCategory::ProducerConsumer','ThreadCategory::Consumer')),
            (AUX.replace('ap.consumer_release(ar);',''),KERNEL),
            (AUX,KERNEL.replace('NumAuxMathThreads + cutlass::NumThreadsPerWarp;','NumAuxMathThreads;')),
        ]
        for aux,kernel in plants:
            with self.assertRaises(ValueError):admit(aux,kernel)

if __name__=='__main__':unittest.main()
