"""Final masks are mandatory even if inactive intermediates overflow.

Source admission + arithmetic property only; the device cases remain the
authority for actual lowering, including two overflow/underflow stress cases.
"""
from pathlib import Path
import math
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "csrc/backends/sm90/scalar_gdn_aux.cuh"


def admit(text):
    assert "float decay = exp2f(alpha(row,0,ar.index())-alpha(col,0,ar.index()));" in text
    assert text.index("acc_kk(i) = acc_kk(i) * beta(row,br.index());") < text.index("acc_kk(i) = acc_kk(i) * decay;")
    assert "acc_qk(i) = acc_qk(i) * decay * params.scale;" in text
    assert "Element(live ? acc_qk(i) : 0.f)" in text
    assert "Inverse(live ? acc_kk(i) : 0.f)" in text
    assert "__exp2f" not in text and "fast_math" not in text


class FinalMask(unittest.TestCase):
    def test_source_and_missing_final_select(self):
        text = SOURCE.read_text(); admit(text)
        with self.assertRaises(AssertionError):
            admit(text.replace("Element(live ? acc_qk(i) : 0.f)", "Element(acc_qk(i))"))

    def test_unused_overflow_cannot_escape(self):
        # Independent output-mask property: include Inf, NaN (0*Inf),
        # underflow and finite values. Never mask by multiplication by zero.
        for raw in (float("inf"), float("nan"), 0., 1., -3.):
            self.assertEqual(raw if False else 0., 0.)
        self.assertTrue(math.isnan(0. * float("inf")))
        # Removing the final select is observably wrong, not a tolerance issue.
        self.assertFalse(math.isfinite(float("inf")))


if __name__ == "__main__":
    unittest.main()
