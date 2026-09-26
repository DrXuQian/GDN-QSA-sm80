// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

namespace gdn::sm90 {

// Convert in logical fragment order. CUTLASS selects native paired RNE casts;
// no aliasing reinterpret_cast or assumption that equal shapes imply strides.
template<class To, class Fragment>
CUTE_DEVICE auto convert_fragment(Fragment const& input) {
    using From = typename Fragment::value_type;
    constexpr int Count = decltype(cute::size(input))::value;
    cutlass::Array<From,Count> source;
    CUTE_UNROLL
    for (int i=0; i<Count; ++i) source[i] = input(i);
    auto values = cutlass::NumericArrayConverter<To,From,Count,
        cutlass::FloatRoundStyle::round_to_nearest>{}(source);
    auto output = cute::make_fragment_like<To>(input);
    CUTE_UNROLL
    for (int i=0; i<Count; ++i) output(i) = values[i];
    return output;
}

} // namespace gdn::sm90
