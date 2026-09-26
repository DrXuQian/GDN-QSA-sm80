#pragma once
#include <cute/config.hpp>

namespace gdn::sm90 {

// Strict bound: every rounded pairwise difference of finite prefixes is
// greater than -126. No assumption about gate sign, monotonicity or tails.
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

template<bool NormalSpan>
CUTE_DEVICE float auxiliary_exp2(float exponent) {
    // CUDA12.8's standard exp2f has this identical MUFU path for x >= -126.
    // The alternative preserves standard subnormal handling. Native target
    // equivalence is a compiled postcondition, not inferred from these names.
    if constexpr (NormalSpan) {
        float result;
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(exponent));
        return result;
    } else return exp2f(exponent);
}

} // namespace gdn::sm90
