// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/config.hpp>

#ifndef GDN_SM90_FAST_EXP2
#define GDN_SM90_FAST_EXP2 0
#endif
#if GDN_SM90_FAST_EXP2 != 0 && GDN_SM90_FAST_EXP2 != 1
#error "GDN_SM90_FAST_EXP2 must be 0 (standard) or 1 (fastmath)"
#endif

namespace gdn::sm90 {
// Opt-in arithmetic variant, matching FlashInfer's exp2(fastmath=True).
// EX2's approximate/FTZ form is NOT a RAW-equivalent replacement for
// standard exp2f. Keep this confined to gate exponentiation, not global
// --use_fast_math (which would also change division, contraction and FTZ).
CUTE_DEVICE float gate_exp2(float value) {
#if GDN_SM90_FAST_EXP2
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
#else
    return exp2f(value);
#endif
}
} // namespace gdn::sm90
