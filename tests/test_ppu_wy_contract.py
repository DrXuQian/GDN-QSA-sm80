#!/usr/bin/env python3
"""CPU tests for opt-in loading, timing order and paired admission."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks")]
from gdn_qsa_sm80 import gdn_wy_interface as api
import bench_ppu_wy_fla as benchmark
from bench_ppu_wy_fla import (comparison_summary, delivery_comparisons, order,
                              DELIVERY_ROLES, TILE_ROLES, experiment, resolve_samples)
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

    def test_delivery_orders_are_complete_and_balanced(self):
        rows = [order(i, DELIVERY_ROLES) for i in range(14)]
        self.assertEqual(len(set(rows)), 14)
        for row in rows:
            self.assertEqual(set(row), set(DELIVERY_ROLES))
        for column in zip(*rows):
            for role in DELIVERY_ROLES:
                self.assertEqual(column.count(role), 2)

    def test_tiled_orders_and_masks_are_a_distinct_balanced_family(self):
        names, roles = experiment(tile_ab=True)
        self.assertEqual(roles, TILE_ROLES)
        self.assertEqual(tuple(api.DELIVERIES[name] for name in names), (8, 16, 32, 48, 56))
        self.assertEqual(len(roles), 8)
        rows = [order(i, roles) for i in range(resolve_samples(None, roles))]
        self.assertEqual(len(set(rows)), 16)
        for row in rows:
            self.assertEqual(set(row), set(roles))
        for column in zip(*rows):
            for role in roles:
                self.assertEqual(column.count(role), 2)
        with self.assertRaises(ValueError):
            experiment(delivery_ab=True, tile_ab=True)

    def test_missing_state_output_cell_cannot_shrink_the_denominator(self):
        def check_inventory():
            _, roles = experiment(tile_ab=True)
            self.assertEqual(set(roles), {"original", "wy", "fla", "wy-tiled-prepare",
                "wy-tiled-state", "wy-tiled-output", "wy-tiled-state-output", "wy-tiled-all"})
            self.assertEqual(len(roles), 8)
        check_inventory()
        missing = tuple(name for name in api.TILED_DELIVERIES if name != "tiled-state-output")
        with patch.object(benchmark, "TILED_DELIVERIES", missing):
            with self.assertRaises(AssertionError):
                check_inventory()

    def test_sample_count_tracks_actual_family_and_rejects_old_fourteen(self):
        self.assertEqual(resolve_samples(None, experiment()[1]), 12)
        self.assertEqual(resolve_samples(None, DELIVERY_ROLES), 14)
        self.assertEqual(resolve_samples(None, TILE_ROLES), 16)
        self.assertEqual(resolve_samples(32, TILE_ROLES), 32)
        for samples in (0, 7, 14, 15, 17, 24):
            with self.subTest(samples=samples), self.assertRaisesRegex(ValueError, "multiple of 16"):
                resolve_samples(samples, TILE_ROLES)

    def test_state_output_keeps_scalar_prepare_at_the_real_python_abi(self):
        class Fake:
            def forward(self, *args):
                self.args = args
                return torch.ones(1), torch.ones(1)
        x, fake = torch.zeros(1), Fake()
        def check_selected_mask():
            with patch.object(api, "_backend", return_value=fake):
                api.gdn_chunk_wy(x, x, x, x, x, delivery="tiled-state-output")
            self.assertEqual(len(fake.args), 8)
            self.assertEqual(fake.args[-1], 48)
        check_selected_mask()
        for wrong_mask in (0, 16, 32, 56):
            with self.subTest(mask=wrong_mask), patch.dict(api.DELIVERIES, {"tiled-state-output": wrong_mask}):
                with self.assertRaises(AssertionError):
                    check_selected_mask()

    def test_state_output_verdict_compares_measured_pair_to_each_incumbent(self):
        arms = {role: dict(samples_us=times) for role, times in (
            ("wy", [700., 710.]), ("fla", [480., 490.]),
            ("original", [420., 430.]), ("wy-tiled-state", [450., 460.]),
            ("wy-tiled-all", [440., 446.]), ("wy-tiled-state-output", [432., 438.]))}
        result = delivery_comparisons(arms, "tiled-state-output")
        self.assertEqual(set(result), {"wy", "fla", "original", "wy-tiled-state", "wy-tiled-all"})
        self.assertEqual(result["original"]["verdict"], "CONTROL-WINS")
        self.assertEqual(result["wy-tiled-all"]["verdict"], "CANDIDATE-WINS")
        self.assertAlmostEqual(result["wy-tiled-all"]["descriptive_speedup"], 443. / 435.)
        # A low median cannot erase the observed overlap or a losing arm.
        arms["wy-tiled-state-output"]["samples_us"] = [432., 435., 480.]
        self.assertEqual(delivery_comparisons(arms, "tiled-state-output")["wy-tiled-all"]["verdict"],
                         "UNRESOLVED")
        arms["wy-tiled-state-output"]["samples_us"] = [760., 770.]
        self.assertTrue(all(x["verdict"] == "CONTROL-WINS"
                            for x in delivery_comparisons(arms, "tiled-state-output").values()))
        del arms["wy-tiled-all"]
        with self.assertRaises(KeyError):
            delivery_comparisons(arms, "tiled-state-output")

    def test_complete_comparison_runs_pair_and_rejects_raw_bit_drift(self):
        # Exercise actual admission/call binding/timing loops on CPU. Events
        # are synthetic: this tests plumbing, not any device timing claim.
        class Event:
            def __init__(self, **kwargs):
                pass
            def record(self):
                pass
            def synchronize(self):
                pass
            def elapsed_time(self, other):
                return 1.0
        cpu = tuple(torch.ones(1) for _ in range(5))
        want = (torch.ones(1), torch.ones(1))
        args = SimpleNamespace(delivery_ab=False, tile_ab=True, warmup=5, samples=16, launches=10)
        seen = []
        def run(plant=False):
            def wy(*inputs, delivery="scalar"):
                seen.append(delivery)
                # Within the unchanged 2% gate, but not scalar raw equality.
                return (want[0] + .001, want[1]) if plant and delivery == "tiled-state-output" else want
            with patch.object(benchmark.admission, "fixture", return_value=cpu), \
                    patch.object(benchmark.admission, "reference", return_value=want), \
                    patch.object(benchmark.admission, "gdn_chunk", return_value=want), \
                    patch.object(benchmark, "gdn_chunk_wy", side_effect=wy), \
                    patch.object(benchmark, "fla_call", return_value=lambda: want), \
                    patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "Event", Event), \
                    redirect_stdout(io.StringIO()):
                return benchmark.compare(None, -.1, args, torch.device("cpu"))
        result = run()
        self.assertEqual(set(result["arms"]), set(TILE_ROLES))
        for name in ("scalar", *api.TILED_DELIVERIES):
            self.assertEqual(seen.count(name), 8 + 5 + 16 * 10)
        pair = result["arms"]["wy-tiled-state-output"]
        self.assertEqual(pair["delivery_mask"], 48)
        self.assertEqual(len(pair["samples_us"]), 16)
        self.assertEqual(len(pair["versus"]), 5)
        with self.assertRaisesRegex(AssertionError, "wy-tiled-state-output output/state bits differ"):
            run(plant=True)

    def test_delivery_mask_is_consumed_not_silently_ignored(self):
        class Fake:
            def forward(self, *args):
                self.args = args
                return torch.ones(1), torch.ones(1)
        x = torch.zeros(1)
        fake = Fake()
        for name, mask in api.DELIVERIES.items():
            with patch.object(api, "_backend", return_value=fake):
                api.gdn_chunk_wy(x, x, x, x, x, delivery=name)
            self.assertEqual(len(fake.args), 7 if name == "scalar" else 8)
            if name != "scalar":
                self.assertEqual(fake.args[-1], mask)
        with patch.object(api, "_backend", side_effect=AssertionError("should not load")):
            with self.assertRaisesRegex(ValueError, "unknown WY delivery"):
                api.gdn_chunk_wy(x, x, x, x, x, delivery="typo")

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

    def test_fast_median_with_slow_samples_is_still_unresolved(self):
        arms = {role: dict(samples_us=times) for role, times in (
            ("original", [910.096, 915.720, 960.612]),
            ("wy", [713.744, 715.232, 726.492]),
            ("fla", [474.344, 480.806, 852.296]))}
        result = comparison_summary(arms)
        self.assertEqual(result["wy_vs_original"], "WY-WINS")
        self.assertEqual(result["wy_vs_fla"], "UNRESOLVED")
        self.assertAlmostEqual(result["wy_over_fla"], 715.232 / 480.806)
        self.assertEqual(result["ratio_scope"], "DESCRIPTIVE_MEDIANS_NOT_ADMISSION")
        # Removing the slow sample changes the answer: that is exactly why the
        # capture must not silently trim it to produce a desired winner.
        arms["fla"]["samples_us"] = [474.344, 480.806]
        self.assertEqual(comparison_summary(arms)["wy_vs_fla"], "FLA-WINS")

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
