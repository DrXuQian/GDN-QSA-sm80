"""S16 role/channel lifetime contracts; not substitutes for device admission."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'csrc/backends/sm90'
AUX = (ROOT / 'scalar_gdn_aux.cuh').read_text()
GATE = (ROOT / 'scalar_gate.cuh').read_text()
KERNEL = (ROOT / 'cula/kda/sm90/kernel/kernel_kda_fwd.hpp').read_text()


def admit(aux, gate, kernel):
    if 'SeparateScalarGateProducer = true' not in aux:
        raise ValueError('producer separation is not selected')
    qkv = aux.split('CUTE_DEVICE void load_qkv(', 1)[1].split(
        'CUTE_DEVICE void load_alpha_and_last(', 1)[0]
    if 'load_scalar_gate<' in qkv or 'gate_factors[' in qkv:
        raise ValueError('prefix/factors still execute on TMA warp')
    alpha = aux.split('CUTE_DEVICE void load_alpha_and_last(', 1)[1].split(
        'CUTE_DEVICE void compute_aux_safe(', 1)[0]
    channels = {
        'stage*128+lane': 'elo', 'stage*128+lane+32': 'ehi',
        'stage*128+64+lane': 'elo * params.scale',
        'stage*128+64+lane+32': 'ehi * params.scale',
    }
    for index, value in channels.items():
        if f'smem.gate_factors[{index}] = {value};' not in alpha:
            raise ValueError('missing or changed factor publication')
    if 'block,ap,aw,alpha,factors);' not in alpha:
        raise ValueError('factor callback is not bound to prefix publication')
    for text, events in (
        (gate, ['pipeline.producer_acquire(stage);', 'publish_factors(lane, lo, hi, stage.index());',
                'cutlass::arch::fence_view_async_shared();', 'pipeline.producer_commit(stage);']),
        (alpha, ['load_scalar_gate<64,128>', 'ap.consumer_wait(ar);',
                 'lp.producer_acquire(lw);', 'lp.producer_commit(lw);', 'ap.consumer_release(ar);']),
    ):
        positions = [text.index(item) for item in events]
        if positions != sorted(positions):
            raise ValueError('stage publication/release order changed')
    if 'NeedsAlpha && !CollectiveMainloop::SeparateScalarGateProducer' not in kernel:
        raise ValueError('TMA warp still claims alpha producer')
    if 'alpha_pipeline_params.role = MainloopAlphaPipeline::ThreadCategory::ProducerConsumer;' not in kernel:
        raise ValueError('prefix warp must both produce and consume')
    if ('alpha_pipeline_params.consumer_arv_count = AlphaConsumers;' not in kernel or
            'AlphaConsumers = StateThreads+AuxThreads+32;' not in kernel):
        raise ValueError('alpha actual-state+aux+producer participant contract changed')


class FactorOwner(unittest.TestCase):
    def test_actual_source(self):
        admit(AUX, GATE, KERNEL)

    def test_missing_channel_callback_owner_and_lifetime_are_red(self):
        plants = [
            (AUX.replace('smem.gate_factors[stage*128+64+lane+32] = ehi * params.scale;', ''), GATE, KERNEL),
            (AUX.replace('block,ap,aw,alpha,factors);', 'block,ap,aw,alpha);'), GATE, KERNEL),
            (AUX, GATE, KERNEL.replace('ThreadCategory::ProducerConsumer', 'ThreadCategory::Consumer')),
            (AUX, GATE.replace('publish_factors(lane, lo, hi, stage.index());', ''), KERNEL),
            (AUX.replace('ap.consumer_release(ar);', ''), GATE, KERNEL),
            (AUX, GATE, KERNEL.replace('StateThreads+AuxThreads+32;', '256+AuxThreads+32;')),
        ]
        for args in plants:
            with self.assertRaises(ValueError):
                admit(*args)


if __name__ == '__main__':
    unittest.main()
