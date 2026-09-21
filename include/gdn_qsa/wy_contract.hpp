#pragma once

#include <cstdint>

namespace gdn_qsa::wy {
constexpr int Chunk = 64;
constexpr int Dim = 128;
constexpr int ValueTile = 32;
constexpr int StateThreads = 64;
constexpr int ParallelThreads = 128;

struct Shape {
  int batch, sequence, q_heads, value_heads;
  constexpr int chunks() const { return (sequence - 1) / Chunk + 1; }
  constexpr int q_head(int h) const { return h / (value_heads / q_heads); }
  constexpr int64_t groups() const { return int64_t(batch) * value_heads * chunks(); }
  constexpr int64_t group(int b, int h, int t) const {
    return (int64_t(b) * value_heads + h) * chunks() + t;
  }
  constexpr int64_t input(int b, int t, int h, int heads) const {
    return ((int64_t(b) * sequence + t) * heads + h) * Dim;
  }
};

// One warp owns all K rows for 16 independent V columns. K is never split
// across CTAs: the update requires no inter-CTA reduction or publication.
constexpr int state_v_start(int slice, int warp) { return slice * ValueTile + warp * 16; }
constexpr int state_k_start(int tile) { return tile * 16; }
constexpr int64_t tile_offset(int64_t group) { return group * Chunk * Dim; }
constexpr int64_t state_offset(int64_t group) { return group * Dim * Dim; }
}  // namespace gdn_qsa::wy
