// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/tensor.hpp>

namespace gdn::sm90 {

// Each thread owns 128 FP32 words. Four adjacent words per lane give
// contiguous 512-byte warp transfers; no other thread consumes these slots.
CUTE_HOST_DEVICE constexpr int state_park_index(int thread, int component) {
    return (component / 4) * (128 * 4) + thread * 4 + component % 4;
}

template<bool Store, class Fragment>
CUTE_DEVICE void state_park_copy(float* memory, int thread, Fragment& fragment) {
    static_assert(cute::size(Fragment{}) == 128, "one-WG whole-V128 state");
    CUTE_UNROLL
    for (int i = 0; i < 128; i += 4) {
        uint32_t address = cute::cast_smem_ptr_to_uint(memory + state_park_index(thread, i));
        if constexpr (Store) {
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(address),
                "r"(__float_as_uint(fragment(i))), "r"(__float_as_uint(fragment(i+1))),
                "r"(__float_as_uint(fragment(i+2))), "r"(__float_as_uint(fragment(i+3))) : "memory");
        } else {
            uint32_t a, b, c, d;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" :
                "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(address) : "memory");
            fragment(i) = __uint_as_float(a); fragment(i+1) = __uint_as_float(b);
            fragment(i+2) = __uint_as_float(c); fragment(i+3) = __uint_as_float(d);
        }
    }
}
} // namespace gdn::sm90
