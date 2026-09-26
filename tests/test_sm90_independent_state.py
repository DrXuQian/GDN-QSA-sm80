"""S18 removes issue ordering only. Data/lifetime code is hash-bound to S16.

This is an experiment-scope guard, not a substitute for CPU owner enumeration
or asynchronous device admission. Changing these files requires a new proof.
"""
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'csrc/backends/sm90'
FIXED = {
    'scalar_gdn_state.cuh': '3aca64676ecff563cff720005c323aa15f85254c814ed7f935c8e96d60a2bc6e',
    'scalar_gate.cuh': 'd4a88dd7cff2ab73d9ca67ceb40eafa52dd652f5173449e307f4e35a610df8e1',
    'cula/kda/sm90/kernel/kernel_kda_fwd.hpp': '24ee5e5327ff58b8f46c4061e737009af73c5847284b09ff238ee982a7a443ab',
    'cula/kda/sm90/collective/mainloop_kda_fwd.hpp': '531d0f3f315095e5b2142bf0eb96a218dd5adcf85bfd22c5bd95c108935b1374',
    'cula/kda/sm90/collective/load_tma.hpp': 'b2b9d9e06486690a286ec9d391294b89bbeedb4560f31a186883604958fe044a',
    'cula/kda/sm90/collective/store_tma.hpp': 'bf7eae1a1b1afc380249aa7c40042be1984a2290eaec090fb586d4cd845c5873',
}


def admit(texts):
    for name, digest in FIXED.items():
        if hashlib.sha256(texts[name]).hexdigest() != digest:
            raise ValueError(f'S16 data/lifetime contract changed: {name}')


class IndependentState(unittest.TestCase):
    def test_data_protocols_and_arithmetic_unchanged(self):
        admit({name: (ROOT/name).read_bytes() for name in FIXED})

    def test_removed_real_completion_or_release_is_red(self):
        for operation in (b'kp.consumer_release(kr);', b'warpgroup_wait<0>();',
                          b'op.producer_commit(ow);', b'ap.consumer_wait(ar);'):
            texts = {name: (ROOT/name).read_bytes() for name in FIXED}
            name = 'scalar_gdn_state.cuh'
            assert operation in texts[name]
            texts[name] = texts[name].replace(operation, b'', 1)
            with self.assertRaises(ValueError):
                admit(texts)


if __name__ == '__main__':
    unittest.main()
