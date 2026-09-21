#!/usr/bin/env python3
"""CPU tests for opt-in loading, timing order and paired admission."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks")]
from gdn_qsa_sm80 import gdn_wy_interface as api
from bench_ppu_wy_fla import order
from bench_ppu_gdn_fla import checked_pair, verdict


class Contracts(unittest.TestCase):
    def tearDown(self):
        api._backend.cache_clear()

    def test_opt_in_required(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GDN_QSA_WY_EXTENSION"):
                api._backend()

    def test_empty_path_not_directory_fallback(self):
        with patch.dict("os.environ", {"GDN_QSA_WY_EXTENSION": ""}, clear=True):
            with self.assertRaises(RuntimeError):
                api._backend()

    def test_six_orders_balance_positions(self):
        rows = [order(i) for i in range(6)]
        self.assertEqual(len(set(rows)), 6)
        for column in zip(*rows):
            for role in ("original", "wy", "fla"):
                self.assertEqual(column.count(role), 2)

    def test_wrong_values_are_failures(self):
        want = (torch.ones(3), torch.ones(4))
        checked_pair(want, want)
        for wrong in ((torch.zeros(3), want[1]), (want[0], torch.zeros(4)), (want[0], None),
                      (torch.full((3,), float("nan")), want[1])):
            with self.assertRaises(AssertionError):
                checked_pair(wrong, want)

    def test_both_verdict_directions_and_overlap(self):
        self.assertEqual(verdict([1, 2], [3, 4]), "OURS-WINS")
        self.assertEqual(verdict([3, 4], [1, 2]), "FLA-WINS")
        self.assertEqual(verdict([1, 3], [2, 4]), "UNRESOLVED")

    def test_forwarding_and_no_state(self):
        class Fake:
            def forward(self, *args):
                self.args = args
                return torch.ones(1), torch.ones(1)
        fake = Fake()
        x = torch.zeros(3, 2).T
        with patch.object(api, "_backend", return_value=fake):
            out, state = api.gdn_chunk_wy(x, x, x, x, x, output_final_state=False)
        self.assertIsNone(state)
        self.assertTrue(all(t.is_contiguous() for t in fake.args[:5]))
        self.assertIsNone(fake.args[5])
        self.assertFalse(fake.args[6])


if __name__ == "__main__":
    unittest.main()
