"""Range-guard producer/consumer admission; actual values are device-probed."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / "csrc/backends/sm90"


def admit(aux, gate, helper):
    assert "cute::array_aligned<int, Base::StagesAlpha::value> aux_normal_exp2;" in aux
    assert "bool normal_span = collect_aux_normal_span(lo, hi);" in aux
    assert "if (lane == 0) smem.aux_normal_exp2[stage] = normal_span;" in aux
    assert "load_scalar_gate<64,128>" in aux and "block,ap,aw,alpha,factors);" in aux
    body = aux[aux.index("CUTE_DEVICE void compute_aux_safe("):]
    assert "smem.aux_normal_exp2[ar.index()]" in body
    read = body.index("smem.aux_normal_exp2[ar.index()]")
    assert body.index("ap.consumer_wait(ar);") < read < body.index("ap.consumer_release(ar);")
    assert "publish_factors(lane, lo, hi, stage.index());" in gate
    assert gate.index("publish_factors(lane, lo, hi, stage.index());") < gate.index("pipeline.producer_commit(stage);")
    assert "isfinite(lo) && isfinite(hi)" in helper
    assert "span < 126.f" in helper and "__fsub_rn(maximum, minimum)" in helper
    assert "__all_sync(0xffffffffu" in helper


class AuxRange(unittest.TestCase):
    def test_actual_source_and_missing_publication(self):
        aux = (ROOT / "scalar_gdn_aux.cuh").read_text()
        gate = (ROOT / "scalar_gate.cuh").read_text()
        helper = (ROOT / "aux_exp2_range.cuh").read_text()
        admit(aux, gate, helper)
        with self.assertRaises(AssertionError):
            admit(aux.replace("smem.aux_normal_exp2[stage] = normal_span;", ""), gate, helper)
        with self.assertRaises(AssertionError):
            admit(aux.replace("smem.aux_normal_exp2[ar.index()]", "smem.aux_normal_exp2[0]"), gate, helper)
        with self.assertRaises(AssertionError):
            admit(aux, gate.replace("publish_factors(lane, lo, hi, stage.index());", ""), helper)

    def test_unsafe_bound_and_omitted_finite_check_rejected(self):
        aux = (ROOT / "scalar_gdn_aux.cuh").read_text()
        gate = (ROOT / "scalar_gate.cuh").read_text()
        helper = (ROOT / "aux_exp2_range.cuh").read_text()
        for original, replacement in (("span < 126.f", "span < 128.f"),
                                      ("isfinite(lo) && isfinite(hi)", "isfinite(lo)")):
            with self.assertRaises(AssertionError):
                admit(aux, gate, helper.replace(original, replacement))


if __name__ == "__main__":
    unittest.main()
