"""Publication order/liveness, not a model of GPU arithmetic or memory ordering.

Native mbarrier inspection and the unchanged device numerical/replay gate are
separate requirements. The old joint-publication source must fail this gate.
"""
from collections import deque
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
REL = "csrc/backends/sm90/scalar_gdn_aux.cuh"


def source_order(text):
    text = re.sub(r"//[^\n]*", "", text)
    tokens = ["kkp.producer_acquire(kw)", "copy(store_kk,",
              "solve.compute(kk(_,_,kw.index()))",
              "copy(store_qk,tq.retile_S(operand)",
              "fence_view_async_shared()", "kkp.producer_commit(kw)",
              "qkp.producer_acquire(qw)", "copy(store_qk, tq.retile_S(out_qk)",
              "fence_view_async_shared()", "qkp.producer_commit(qw)",
              "ap.consumer_release(ar)", "bp.consumer_release(br)"]
    cursor = text.index("void compute_aux_safe")
    for token in tokens:
        cursor = text.index(token, cursor) + len(token)
    # Matching state consumers retire KK only after NewV, then consume QK.
    state = (ROOT / "csrc/backends/sm90/scalar_gdn_state.cuh").read_text()
    assert state.index("kkp.consumer_release(kkr)") < state.index("qkp.consumer_wait(qkr)")


def enumerate_interleavings(chunks, omit_kk=False):
    # Aux publishes KK, then QK; both state WGs retire KK, then QK. Each
    # two-slot ring can overwrite a slot only after BOTH consumers retire it.
    # Tuple: aux events completed, state0 events completed, state1 completed.
    todo = deque([(0, 0, 0)]); seen = set(todo); terminal = (2*chunks,)*3
    while todo:
        p, a, b = state = todo.popleft()
        if state == terminal:
            continue
        nexts = []
        if p < 2*chunks:
            phase = p % 2; chunk = p // 2
            old_event = 2*(chunk-2)+phase
            if chunk < 2 or (a > old_event and b > old_event):
                nexts.append((p+1, a, b))
        for consumer, at in enumerate((a, b)):
            if at < p and not (omit_kk and at % 2 == 0):
                nexts.append((p, a+1, b) if consumer == 0 else (p, a, b+1))
        if not nexts:
            raise AssertionError(f"deadlock: {state}")
        for successor in nexts:
            if successor not in seen:
                seen.add(successor); todo.append(successor)
    assert terminal in seen
    return len(seen)


class AuxPublication(unittest.TestCase):
    def test_actual_order_and_old_source_negative(self):
        source_order((ROOT / REL).read_text())
        old = subprocess.check_output(["git", "show", f"64691d1:{REL}"], cwd=ROOT, text=True)
        with self.assertRaises(ValueError):
            source_order(old)

    def test_all_role_interleavings_and_missing_publish_negative(self):
        self.assertGreater(sum(enumerate_interleavings(n) for n in range(1, 33)), 1000)
        with self.assertRaises(AssertionError):
            enumerate_interleavings(4, omit_kk=True)


if __name__ == "__main__":
    unittest.main()
