// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/tensor.hpp>

namespace gdn::sm90 {

// Eight groups of256 owner-private float4 slots. A warp's32 v4 transfers
// span512 contiguous bytes (four conflict-free128B phases). No thread reads
// another thread's slots, so no new inter-thread publication is required.
CUTE_HOST_DEVICE constexpr int output_stash_index(int thread, int component) {
    return (component / 4) * (256 * 4) + thread * 4 + component % 4;
}

template<bool Store, class Fragment>
CUTE_DEVICE void output_stash_copy(float* storage, int thread, Fragment& fragment) {
    static_assert(cute::size(Fragment{}) == 32, "exact state O1 fragment required");
    CUTE_UNROLL
    for (int i = 0; i < 32; i += 4) {
        uint32_t address = cute::cast_smem_ptr_to_uint(storage + output_stash_index(thread, i));
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
