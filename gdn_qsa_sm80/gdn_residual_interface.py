"""Explicit non-CP reassociation; not a bit-preserving WY delivery or auto route."""
from .gdn_wy_interface import _backend, DELIVERIES

MATH_CONTRACT = "gated-inverse-residual-bf16-v1"
WY_MATH_CONTRACT = "materialized-wu-bf16-v1"
# Profiling may select a new algorithm; the public WY delivery inventory
# deliberately does not. None is NOT a secretly reused delivery mask.
PROFILE_VARIANTS = {**DELIVERIES, "residual": None}


def math_contract(variant):
    if variant not in PROFILE_VARIANTS:
        raise ValueError(f"unknown algorithm/delivery: {variant}")
    return MATH_CONTRACT if variant == "residual" else WY_MATH_CONTRACT


def gdn_chunk_residual(q, k, v, g, beta, initial_state=None, output_final_state=True):
    """P[beta*(V-exp(prefix)*K@H)] with separate BF16 residual rounding.

    Same shape/type/scale contract as WY, FP32 state, no reset/truncation/CP.
    Do not use old-WY RAW-BIT equality to admit this different association.
    """
    inputs = tuple(x.contiguous() for x in (q, k, v, g, beta))
    state = initial_state.contiguous() if initial_state is not None else None
    output, final = _backend().residual(*inputs, state, output_final_state)
    return output, final if output_final_state else None
