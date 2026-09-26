"""Selected loader contract; actual device coverage remains required."""
from pathlib import Path
import re
import unittest

SOURCE = (Path(__file__).resolve().parents[1] /
          'csrc/backends/sm90/cula/kda/sm90/collective/mainloop_kda_fwd.hpp').read_text()


def admit(text):
    body = text.split('    load_qkv(', 1)[1].split('    load_beta(', 1)[0]
    calls = re.findall(r'([qkv])_collective_load\.step\(([^;]+)\);', body)
    if [name for name, _ in calls] != ['k', 'q', 'v']:
        raise ValueError('TMA issue order is not K,Q,V exactly once')
    for name, args in calls:
        if args != f'{name}_src_dst, blk, {name}_smem_pipe_write, lane_predicate':
            raise ValueError('source/stage binding changed')
        expected = (f'Load{name.upper()}(params.tma_load_{name}, '
                    f'{name}_pipeline, storage.smem_{name})')
        if expected not in body:
            raise ValueError('TMA descriptor/pipeline/destination binding changed')


class TmaOrder(unittest.TestCase):
    def test_selected_order_and_bindings(self):
        admit(SOURCE)

    def test_old_order_and_wrong_descriptor_are_red(self):
        k = 'k_collective_load.step(k_src_dst, blk, k_smem_pipe_write, lane_predicate);'
        q = 'q_collective_load.step(q_src_dst, blk, q_smem_pipe_write, lane_predicate);'
        old = SOURCE.replace(k, 'ORDER_SWAP').replace(q, k).replace('ORDER_SWAP', q)
        for text in (old, SOURCE.replace('LoadK(params.tma_load_k,', 'LoadK(params.tma_load_q,')):
            with self.assertRaises(ValueError):
                admit(text)


if __name__ == '__main__':
    unittest.main()
