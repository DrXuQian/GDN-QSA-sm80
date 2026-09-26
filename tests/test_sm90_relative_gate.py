"""Relative channel ownership/publication contracts, before device admission."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]/"csrc/backends/sm90"


def admit(aux,state,gate):
    for token in (
        "cute::array_aligned<float, 64 * Base::StagesAlpha::value> relative_gate;",
        "relative_gate_last_is_hi(valid) ? hi : lo, relative_gate_last_lane(valid)",
        "exp2f(__fsub_rn(last_prefix,lo))", "exp2f(__fsub_rn(last_prefix,hi))",
        "smem.relative_gate[relative_gate_index(stage,lane)] = rlo;",
        "smem.relative_gate[relative_gate_index(stage,lane+32)] = rhi;",
        "lane < valid ?", "lane+32 < valid ?",
    ):
        assert token in aux,token
    assert "float gain = smem.relative_gate[relative_gate_index(ar.index(),t)];" in state
    assert "exp2f(" not in state
    assert "publish_factors(lane, lo, hi, stage.index());" in gate
    assert gate.index("publish_factors(lane, lo, hi, stage.index());") < gate.index("pipeline.producer_commit(stage);")
    assert state.index("smem.relative_gate[") < state.index("ap.consumer_release(ar);")


class RelativeGate(unittest.TestCase):
    def test_actual_source_and_missing_channel_negative(self):
        aux=(ROOT/"scalar_gdn_aux.cuh").read_text()
        state=(ROOT/"scalar_gdn_state.cuh").read_text()
        gate=(ROOT/"scalar_gate.cuh").read_text()
        admit(aux,state,gate)
        for old,new in (("smem.relative_gate[relative_gate_index(stage,lane+32)] = rhi;",""),
                        ("relative_gate_last_lane(valid)","0")):
            with self.assertRaises(AssertionError):admit(aux.replace(old,new),state,gate)
        with self.assertRaises(AssertionError):admit(aux,state,gate.replace("publish_factors(lane, lo, hi, stage.index());",""))


if __name__=="__main__":unittest.main()
