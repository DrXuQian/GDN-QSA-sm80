// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/config.hpp>

namespace gdn::sm90 {

// The caller has already masked the upper triangle and padded rows/columns.
// Padded diagonal entries must still be one: inversion covers the full64x64.
CUTE_HOST_DEVICE constexpr float inverse_unit_input(
    int row, int column, float masked_lower) {
    return row == column ? 1.0f : masked_lower;
}

}  // namespace gdn::sm90
