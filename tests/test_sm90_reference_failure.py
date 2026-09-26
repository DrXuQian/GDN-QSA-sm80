"""Reference failure never admits a failed candidate or a different criterion."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from admit_sm90_followups import admit_reference_diagnosis


class ReferenceFailure(unittest.TestCase):
    def fixture(self):
        good = dict(verdict="PASS", errors=[.001, .002], repeat="8/8 RAW-BIT", fingerprint="same")
        return dict(status="DIAGNOSIS_COMPLETE_NOT_PERFORMANCE", criterion=.02,
                    device_watch=dict(errors=[]), roles={"ours-cuda": dict(good), "ours-candidate": dict(good),
                     "flashqla-auto": dict(errors=[.03, .01]), "flashqla-no-cp": dict(errors=[.03, .01])})

    def test_reference_failure_remains_distinct(self):
        admit_reference_diagnosis(self.fixture(), .02)

    def test_our_error_raw_change_and_missing_role_are_red(self):
        for plant in ("error", "nan", "raw", "missing", "threshold", "interference"):
            data = self.fixture()
            if plant == "error": data["roles"]["ours-candidate"]["errors"] = [.03, .01]
            elif plant == "nan": data["roles"]["ours-candidate"]["errors"] = [float("nan"), .01]
            elif plant == "raw": data["roles"]["ours-candidate"]["fingerprint"] = "different"
            elif plant == "missing": del data["roles"]["ours-candidate"]
            elif plant == "threshold": data["criterion"] = .04
            else: data["device_watch"]["errors"] = ["foreign process"]
            with self.assertRaises(ValueError):
                admit_reference_diagnosis(data, .02)


if __name__ == "__main__":
    unittest.main()
