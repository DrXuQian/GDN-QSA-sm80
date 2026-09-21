"""Explicit experimental WY backend. Does not change original auto routing."""
from functools import lru_cache
from importlib import util
import os
from pathlib import Path


@lru_cache(maxsize=1)
def _backend():
    path = os.environ.get("GDN_QSA_WY_EXTENSION")
    if not path or not Path(path).is_file():
        raise RuntimeError("set GDN_QSA_WY_EXTENSION to the built _gdn_wy_ppu*.so")
    spec = util.spec_from_file_location("_gdn_wy_ppu", Path(path).resolve())
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gdn_chunk_wy(q, k, v, g, beta, initial_state=None, output_final_state=True):
    """C64 WY forward, BF16 Q/K/V/beta, BF16/FP32 log gate, FP32 state.

    Native GVA, D=128, scale=1/sqrt(128). No QK normalization, gate transform,
    reset approximation, decay-based host synchronization or fallback.
    """
    tensors = tuple(x.contiguous() for x in (q, k, v, g, beta))
    state = initial_state.contiguous() if initial_state is not None else None
    output, final = _backend().forward(*tensors, state, output_final_state)
    return output, final if output_final_state else None
