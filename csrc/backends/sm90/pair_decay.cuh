// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/config.hpp>

namespace gdn::sm90 {

// Bijection inside each64-column row. Row bits0..4 permute onto column
// bits0,3,4,1,2: both one-row-per-producer-lane and the actual GMMA C
// coordinates have distinct32x4B banks for each scalar request.
CUTE_HOST_DEVICE constexpr int pair_decay_index(int stage, int row, int col) {
    int swizzle = (row & 1) | ((row & 6) << 2) | ((row & 24) >> 2);
    return stage*4096 + row*64 + (col ^ swizzle);
}

CUTE_DEVICE inline void publish_pair_decay(float lo, float hi, int stage, float* target) {
    int lane = int(threadIdx.x) & 31;
    CUTE_NO_UNROLL
    for (int col=0; col<32; ++col) {
        float left = __shfl_sync(0xffffffffu,lo,col);
        float right = __shfl_sync(0xffffffffu,hi,col);
        // Match the rounded shared FP32 prefixes and standard exp2f used by
        // the parent auxiliary role; keep even masked/off-diagonal entries.
        target[pair_decay_index(stage,lane,col)] = exp2f(__fsub_rn(lo,left));
        target[pair_decay_index(stage,lane,col+32)] = exp2f(__fsub_rn(lo,right));
        target[pair_decay_index(stage,lane+32,col)] = exp2f(__fsub_rn(hi,left));
        target[pair_decay_index(stage,lane+32,col+32)] = exp2f(__fsub_rn(hi,right));
    }
}

} // namespace gdn::sm90
