"""Repair cannot alter arithmetic, inputs, capture or numerical thresholds."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from sm90_measurement_epochs import prove_identity_only


class Epochs(unittest.TestCase):
    def test_actual_repair_only_changes_collector(self):
        old = subprocess.check_output(["git", "show", "71534d6:benchmarks/profile_sm90_libraries.py"], cwd=ROOT)
        # That completed matrix's metadata-only repair is immutable. S47 is
        # a new execution-contract epoch, not eligible to resume old captures.
        new = subprocess.check_output(["git", "show", "5f5905b:benchmarks/profile_sm90_libraries.py"], cwd=ROOT)
        prove_identity_only(old, new)

    def test_two_launch_contract_cannot_reuse_identity_only_resume(self):
        old = subprocess.check_output(["git", "show", "5f5905b:benchmarks/profile_sm90_libraries.py"], cwd=ROOT)
        new = (ROOT / "benchmarks/profile_sm90_libraries.py").read_bytes()
        with self.assertRaisesRegex(ValueError, "measurement changed"):
            prove_identity_only(old,new)

    def test_input_timing_or_threshold_change_is_red(self):
        old = 'def reference_binaries():\n return 1\ndef run():\n x=0.02\n return x\n'
        new = old.replace('return 1', 'return 2')
        prove_identity_only(old, new)
        for bad in (new.replace('0.02', '0.2'), new + 'samples=1\n', new.replace('return x', 'return 0')):
            with self.assertRaisesRegex(ValueError, "measurement changed"):
                prove_identity_only(old, bad)


if __name__ == "__main__":
    unittest.main()
