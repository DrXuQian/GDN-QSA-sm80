"""CPU-only admission for the real workload authority and ABI adapters."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "benchmarks"), str(ROOT / "tests"), str(ROOT)]
from sm90_workloads import (WORKLOADS, BY_NAME, FAMILIES, GATES, Workload,
                            validate_inventory, validate_offsets)


class Inventory(unittest.TestCase):
    def test_registered_denominator_and_axes(self):
        validate_inventory(WORKLOADS)
        self.assertEqual(len(WORKLOADS) * len(GATES) * len(FAMILIES), 56)
        self.assertEqual(BY_NAME["heads64-gva4"].shape, (1, 2048, 16, 64))
        self.assertEqual(BY_NAME["seq2048"].shape, (1, 2048, 16, 32))
        self.assertEqual({w.head_ratio for w in WORKLOADS}, {1, 2, 4})
        self.assertEqual({w.batch for w in WORKLOADS}, {1, 2, 4})
        self.assertEqual(len({w.shape for w in WORKLOADS}), 12)

    def test_missing_row_is_red(self):
        with self.assertRaisesRegex(ValueError, "exactly 14"):
            validate_inventory(WORKLOADS[:-1])

    def test_duplicate_row_is_red(self):
        with self.assertRaisesRegex(ValueError, "exactly 14"):
            validate_inventory(WORKLOADS[:-1] + (WORKLOADS[0],))

    def test_real_batch_boundaries_and_negative(self):
        w = BY_NAME["batch2"]
        validate_offsets(w, [0, 2048, 4096])
        with self.assertRaisesRegex(ValueError, "never merge"):
            validate_offsets(w, [0, 4096])

    def test_bad_head_ratio_is_red(self):
        with self.assertRaises(ValueError):
            Workload("bad", 1, 65, 3, 4)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU Torch unavailable; actual tensor adapters NOT tested here")
class TensorAdapters(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_default_fixture_is_previous_input_byte_for_byte(self):
        from test_ppu_gdn_backend import fixture, digest
        from sm90_library_inputs import make_inputs
        w = BY_NAME["seq2048"]
        for gate in GATES:
            cpu, initial = make_inputs(w, gate, fixture)
            self.assertIsNone(initial)
            self.assertEqual(digest(cpu), digest(fixture(1, 2048, 16, 32, gate)))

    def test_batch_gva_and_vk_adapters_and_negatives(self):
        from sm90_library_inputs import (flatten_tokens, flatten_gate, expand_reference_heads,
            validate_expanded_heads, flashinfer_initial, flashinfer_outputs)
        w = Workload("test", 2, 65, 1, 4)
        q = torch.arange(2 * 65 * 128).reshape(2, 65, 1, 128)
        v = torch.arange(2 * 65 * 4 * 128).reshape(2, 65, 4, 128)
        g = torch.arange(2 * 65 * 4).reshape(2, 65, 4)
        state = torch.arange(2 * 4 * 128 * 128).reshape(2, 4, 128, 128)
        q_expand, k_expand = expand_reference_heads((q, q), w)
        self.assertTrue(torch.equal(q_expand[:, :, 3], q[:, :, 0]))
        self.assertTrue(torch.equal(k_expand, q_expand))
        with self.assertRaisesRegex(ValueError, "GVA"):
            validate_expanded_heads((q.repeat_interleave(2, 2),) * 2, w)
        flat = flatten_tokens(v, w, 4)
        self.assertTrue(torch.equal(flat[65], v[1, 0]))
        self.assertTrue(torch.equal(flatten_gate(g, w)[65], g[1, 0]))
        vk = flashinfer_initial(state, w)
        self.assertEqual(vk[1, 3, 7, 19], state[1, 3, 19, 7])
        output, final = flashinfer_outputs((flat, vk), w)
        self.assertTrue(torch.equal(output, v))
        self.assertTrue(torch.equal(final, state))
        # Square matrices hide a transpose in shape checks, but not in values.
        bad = flashinfer_outputs((flat, state), w)[1]
        with self.assertRaises(AssertionError):
            self.assertTrue(torch.equal(bad, state), "wrong VK transpose must be red")

    def test_omitted_batch_boundary_changes_the_actual_oracle(self):
        from test_ppu_gdn_backend import fixture, assert_pair
        from gdn_qsa_sm80.reference.gdn_chunk_ref import torch_recurrent_gated_delta_rule as ref
        q, k, v, g, beta = fixture(2, 65, 1, 1, -.1)
        want = ref(q, k, v, g, beta, output_final_state=True)
        joined = [x.reshape(1, 130, *x.shape[2:]) for x in (q, k, v, g, beta)]
        bad = ref(*joined, output_final_state=True)[0].reshape_as(want[0])
        with self.assertRaises(AssertionError):
            assert_pair((bad,), (want[0],))

    def test_distinct_fp32_gate_and_initial_are_measured_inputs(self):
        from test_ppu_gdn_backend import fixture, digest
        from sm90_library_inputs import make_inputs
        w = BY_NAME["initial-vary"]
        cpu, initial = make_inputs(w, -.1, fixture)
        self.assertEqual(cpu[3].dtype, torch.float32)
        self.assertGreater(cpu[3].float().std().item(), 0)
        self.assertEqual(initial.dtype, torch.float32)
        self.assertGreater(initial.count_nonzero().item(), 0)
        self.assertFalse(torch.equal(initial, initial.transpose(-2, -1)))
        self.assertNotEqual(digest(cpu), digest(fixture(*w.shape, -.1)))


if __name__ == "__main__":
    unittest.main()
