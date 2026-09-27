// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/config.hpp>
namespace gdn::sm90 {
CUTE_HOST_DEVICE constexpr int inverse_prefix_tile16(int warp, int half) {
    return 2*warp+half;
}
} // namespace gdn::sm90
