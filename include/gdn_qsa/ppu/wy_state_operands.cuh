#pragma once

#include <cute/config.hpp>

namespace gdn_qsa::wy {

struct StateOperandTiles {
  CUTE_HOST_DEVICE static constexpr unsigned column(unsigned warp) { return warp & 1u; }
  CUTE_HOST_DEVICE static constexpr unsigned key(unsigned warp, unsigned fragment) {
    return (warp >> 1) * 4u + fragment;
  }
  CUTE_HOST_DEVICE static constexpr unsigned value(unsigned warp, unsigned fragment) {
    return (warp >> 1) * 2u + fragment;
  }
};

// Half-element offsets, NOT bytes. Arguments identify 16x16 cubes, never
// arbitrary element coordinates. The cube swizzle changes only low 8 bits.
template <unsigned Rows>
CUTE_HOST_DEVICE constexpr unsigned operand_cube(unsigned row_tile, unsigned col_tile) {
  static_assert(Rows % 16 == 0);
  return (row_tile + col_tile * (Rows / 16)) * 256u;
}

// Native accumulator lane/slot -> physical position inside one shared cube.
// l013 anchors these coordinates to the actual MMA CLayout, independently
// of the state producer and of the CuTe shared RowLayout.
CUTE_HOST_DEVICE constexpr unsigned operand_fragment(unsigned lane, unsigned slot) {
  unsigned const row = (lane >> 2) + (slot >> 2) * 8u;
  unsigned const col = (lane & 3u) + (slot & 3u) * 4u;
  unsigned const linear = row * 16u + col;
  return linear ^ ((linear >> 3) & 8u);
}

template <unsigned Rows>
CUTE_HOST_DEVICE constexpr unsigned operand_store(unsigned row_tile, unsigned col_tile,
                                                   unsigned lane, unsigned slot) {
  return operand_cube<Rows>(row_tile, col_tile) + operand_fragment(lane, slot);
}

}  // namespace gdn_qsa::wy
