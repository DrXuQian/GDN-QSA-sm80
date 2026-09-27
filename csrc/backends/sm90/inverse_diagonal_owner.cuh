// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/config.hpp>

namespace gdn::sm90 {
struct InverseDiagonalOwner {
    int tile;
    int row;
    bool publishes;
};

// Match the next8->16 merge's warp to its two8x8 diagonal inputs. Every
// lane executes the full-mask shuffle sequence; only lower16 publish.
CUTE_HOST_DEVICE constexpr InverseDiagonalOwner inverse_diagonal_owner(int tid) {
    int lane=tid&31, warp=(tid>>5)&3;
    return {2*warp+(lane&15)/8,lane&7,lane<16};
}
} // namespace gdn::sm90
