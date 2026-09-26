#pragma once
#include <cute/tensor.hpp>

namespace gdn::sm90 {

// Match the reference's three bodies: one first (possibly partial), the
// rolled full middle, and one final tail. T is positive by public admission.
template<bool HasInitial, class Body>
CUTE_HOST_DEVICE void for_each_state_chunk(int tokens, Body body) {
    int chunks=(tokens+63)/64;
    body(0,cute::bool_constant<!HasInitial>{},cute::true_type{});
    CUTE_NO_UNROLL
    for (int chunk=1;chunk<chunks-1;++chunk)
        body(chunk,cute::false_type{},cute::false_type{});
    if (chunks>1) body(chunks-1,cute::false_type{},cute::true_type{});
}

CUTE_HOST_DEVICE constexpr int state_chunk_valid(int tokens,int chunk) {
    int remaining=tokens-chunk*64;
    return remaining<64 ? remaining : 64;
}

} // namespace gdn::sm90
