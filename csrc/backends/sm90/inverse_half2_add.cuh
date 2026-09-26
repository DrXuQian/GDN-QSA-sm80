#pragma once

#include <cute/tensor.hpp>
#include <cutlass/array.h>

namespace gdn_sm90 {

// These are two independent FP16 additions, not a reordered reduction. Recast
// the actual accumulator fragment so the compiler sees both lanes together.
template <class Dst, class Src>
CUTE_HOST_DEVICE void inverse_add_half_pairs(Dst& dst, Src const& src) {
  using Element = typename Dst::value_type;
  static_assert(cute::is_same_v<Element, cutlass::half_t>);
  static_assert(cute::is_same_v<typename Src::value_type, Element>);
  using Pair = cutlass::Array<Element, 2>;
  auto dst_pairs = cute::recast<Pair>(dst);
  auto src_pairs = cute::recast<Pair>(src);
  static_assert(cute::size(dst_pairs) * 2 == cute::size(dst));
  static_assert(cute::size(src_pairs) == cute::size(dst_pairs));
  cute::transform(dst_pairs, src_pairs, dst_pairs, cutlass::plus<Pair>{});
}

}  // namespace gdn_sm90
