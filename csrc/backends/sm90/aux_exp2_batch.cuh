// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/config.hpp>

namespace gdn::sm90 {

// A strict, rounded finite range proves every pairwise difference > -126.
CUTE_HOST_DEVICE constexpr bool aux_normal_span(bool finite, float span) {
    return finite && span >= 0.f && span < 126.f;
}

CUTE_DEVICE bool collect_aux_normal_span(float lo, float hi) {
    bool finite = __all_sync(0xffffffffu, isfinite(lo) && isfinite(hi));
    float minimum = fminf(lo, hi), maximum = fmaxf(lo, hi);
    CUTE_UNROLL
    for (int distance = 16; distance; distance /= 2) {
        minimum = fminf(minimum, __shfl_xor_sync(0xffffffffu, minimum, distance));
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, distance));
    }
    return aux_normal_span(finite, __fsub_rn(maximum, minimum));
}

// CUDA12.8/SM90 standard exp2f lowering, not a general fastmath contract.
// normal_span MUST be warp-uniform and prove all eight input values normal.
// One branch surrounds only EX2, not duplicated matrix epilogues. Register
// tuple constraints prohibit a runtime-indexed local array in this helper.
#define GDN_EX2_CORRECTED_SLOT(i) \
    "setp.lt.f32 scale, %" #i ", 0fC2FC0000;\n" \
    "@scale mul.rn.f32 %" #i ", %" #i ", 0f3F000000;\n" \
    "ex2.approx.ftz.f32 %" #i ", %" #i ";\n" \
    "@scale mul.rn.f32 %" #i ", %" #i ", %" #i ";\n"
#define GDN_EX2_DIRECT_SLOT(i) "ex2.approx.ftz.f32 %" #i ", %" #i ";\n"
CUTE_DEVICE void auxiliary_exp2_batch8(float (&values)[8], bool normal_span) {
    asm volatile(
        "{ .reg .pred scale;\n"
        "targets: .branchtargets corrected, direct;\n"
        "brx.idx.uni %8, targets;\n"
        "corrected:\n"
        GDN_EX2_CORRECTED_SLOT(0) GDN_EX2_CORRECTED_SLOT(1)
        GDN_EX2_CORRECTED_SLOT(2) GDN_EX2_CORRECTED_SLOT(3)
        GDN_EX2_CORRECTED_SLOT(4) GDN_EX2_CORRECTED_SLOT(5)
        GDN_EX2_CORRECTED_SLOT(6) GDN_EX2_CORRECTED_SLOT(7)
        "bra.uni done;\n"
        "direct:\n"
        GDN_EX2_DIRECT_SLOT(0) GDN_EX2_DIRECT_SLOT(1)
        GDN_EX2_DIRECT_SLOT(2) GDN_EX2_DIRECT_SLOT(3)
        GDN_EX2_DIRECT_SLOT(4) GDN_EX2_DIRECT_SLOT(5)
        GDN_EX2_DIRECT_SLOT(6) GDN_EX2_DIRECT_SLOT(7)
        "done:\n}"
        : "+f"(values[0]), "+f"(values[1]), "+f"(values[2]), "+f"(values[3]),
          "+f"(values[4]), "+f"(values[5]), "+f"(values[6]), "+f"(values[7])
        : "r"(int(normal_span)));
}
#undef GDN_EX2_CORRECTED_SLOT
#undef GDN_EX2_DIRECT_SLOT

} // namespace gdn::sm90
