// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/tensor.hpp>
#include <cstdint>

namespace gdn::sm90 {

// Private scratch contains PHYSICAL BF16 halfwords of the two shared-memory
// operands. No logical row-major interpretation or second swizzle on reload.
// Public inputs/state/output and their rounding are unchanged.
struct PreparedAuxLayout {
    static constexpr int Rows = 64;
    static constexpr int PlaneElements = Rows * Rows;
    static constexpr int Planes = 2;  // QK, then inv(I+KK)*diag(beta)
    static constexpr int ChunkElements = Planes * PlaneElements;
    static constexpr int ChunkBytes = ChunkElements * 2;
    static constexpr int VectorElements = 8; // one128-bit transfer
    static constexpr int PlaneVectors = PlaneElements / VectorElements;

    CUTE_HOST_DEVICE static constexpr int chunks(int length) {
        return length / Rows + (length % Rows != 0);
    }
    CUTE_HOST_DEVICE static constexpr int64_t element_offset(
        int batch, int head, int chunk, int heads, int chunk_count, int plane = 0) {
        return ((int64_t(batch)*heads+head)*chunk_count+chunk)*ChunkElements
               + int64_t(plane)*PlaneElements;
    }
    CUTE_HOST_DEVICE static constexpr int64_t elements(int batches, int heads, int length) {
        return int64_t(batches)*heads*chunks(length)*ChunkElements;
    }
};

template<class Layout>
using SingleStage = decltype(cute::composition(Layout{},
    cute::make_tuple(cute::_,cute::_,cute::make_layout(cute::_1{}))));

} // namespace gdn::sm90
