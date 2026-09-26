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

// CUDA12.8/SM90 native exp2f identity: if x < -126, evaluate
// MUFU(x/2)^2 with RN multiplication; otherwise evaluate MUFU(x).
// A producer-proved normal span makes the exceptional predicate false.
// Keep one data path, so the compiler need not merge two full epilogues.
CUTE_DEVICE float auxiliary_exp2_guarded(float exponent, bool normal_span) {
    float result;
    asm("{ .reg .pred scale; .reg .f32 x;\n"
        "  setp.eq.u32 scale, %2, 0;\n"
        "  @scale setp.lt.f32 scale, %1, 0fC2FC0000;\n"
        "  mov.f32 x, %1;\n"
        "  @scale mul.rn.f32 x, %1, 0f3F000000;\n"
        "  ex2.approx.ftz.f32 %0, x;\n"
        "  @scale mul.rn.f32 %0, %0, %0;\n"
        "}" : "=f"(result) : "f"(exponent), "r"(int(normal_span)));
    return result;
}

} // namespace gdn::sm90
