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


PACKED_DELIVERIES = ("prepare", "state", "output", "all")
TILED_DELIVERIES = ("tiled-prepare", "tiled-state", "tiled-output",
                    "tiled-state-output", "tiled-all")
STATE_DELIVERIES = ("tiled-state-output", "tiled-all", "tiled-state-output-address",
                    "tiled-state-output-gates", "tiled-state-output-both")
STAGE_DELIVERIES = ("tiled-state-output", "tiled-state-output-both", "stage-address-prepare",
                    "stage-address-output", "stage-address-both")
PREPARE_ROWS_DELIVERIES = ("tiled-state-output", "tiled-state-output-both", "stage-address-prepare",
                           "prepare-rows-shared", "prepare-rows-warp")
AIU_DELIVERIES = ("prepare-rows-shared", "aiu-state", "aiu-output", "aiu-state-output")
DELIVERIES = {"scalar": 0, "prepare": 1, "state": 2, "output": 4, "all": 7,
              "tiled-prepare": 8, "tiled-state": 16, "tiled-output": 32,
              "tiled-state-output": 48, "tiled-all": 56,
              "tiled-state-output-address": 112, "tiled-state-output-gates": 176,
              "tiled-state-output-both": 240, "stage-address-prepare": 496,
              "stage-address-output": 752, "stage-address-both": 1008,
              "prepare-rows-shared": 1520, "prepare-rows-warp": 2544,
              "aiu-state": 5616, "aiu-output": 9712, "aiu-state-output": 13808}


def gdn_chunk_wy(q, k, v, g, beta, initial_state=None, output_final_state=True, *, delivery="scalar"):
    """C64 WY forward, BF16 Q/K/V/beta, BF16/FP32 log gate, FP32 state.

    Native GVA, D=128, scale=1/sqrt(128). No QK normalization, gate transform,
    reset approximation, decay-based host synchronization or fallback.
    Delivery variants are opt-in; scalar preserves the admitted control.
    """
    tensors = tuple(x.contiguous() for x in (q, k, v, g, beta))
    state = initial_state.contiguous() if initial_state is not None else None
    if delivery not in DELIVERIES:
        raise ValueError(f"unknown WY delivery {delivery!r}; choose {tuple(DELIVERIES)}")
    # Keep the old 7-argument call for archived scalar bindings.
    args = (*tensors, state, output_final_state)
    if delivery != "scalar":
        args += (DELIVERIES[delivery],)
    output, final = _backend().forward(*args)
    return output, final if output_final_state else None
