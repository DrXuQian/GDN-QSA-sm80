#pragma once

#include <cstdint>

#include "gdn_qsa/ppu/warp_mma.cuh"

namespace gdn_qsa::ppu {

// Cooperative CTA matrix product for compile-time multiples of 16.
// C = epilogue(C_accum, row, column). `B` is an ordinary row-major [K,N]
// matrix; the warp primitive receives its transposed logical view [N,K].
template <int M, int N, int K, int NumWarps, class Epilogue>
CUTLASS_DEVICE void cta_gemm_bf16(
    cutlass::bfloat16_t const* a,
    std::int64_t a_row_stride,
    std::int64_t a_k_stride,
    cutlass::bfloat16_t const* b,
    std::int64_t b_k_stride,
    std::int64_t b_column_stride,
    int thread_idx,
    Epilogue&& epilogue) {
  static_assert(M % 16 == 0 && N % 16 == 0 && K % 16 == 0);
  using Mma = WarpMmaBf16M16N16K16;
  int const warp = thread_idx / 32;
  int const lane = thread_idx % 32;
  constexpr int kNBlocks = N / 16;
  constexpr int kTiles = (M / 16) * kNBlocks;
  for (int tile = warp; tile < kTiles; tile += NumWarps) {
    int const m_block = tile / kNBlocks;
    int const n_block = tile - m_block * kNBlocks;
    auto accumulator = Mma::make_accumulator();
    Mma::clear(accumulator);
#pragma unroll
    for (int k_block = 0; k_block < K / 16; ++k_block) {
      auto const* a_tile =
          a + std::int64_t(m_block * 16) * a_row_stride +
          std::int64_t(k_block * 16) * a_k_stride;
      auto const* b_tile =
          b + std::int64_t(k_block * 16) * b_k_stride +
          std::int64_t(n_block * 16) * b_column_stride;
      Mma::accumulate(
          accumulator,
          a_tile, a_row_stride, a_k_stride,
          b_tile, b_column_stride, b_k_stride,
          lane);
    }
    Mma::visit(accumulator, lane, [&](int row, int column, float value) {
      epilogue(m_block * 16 + row, n_block * 16 + column, value);
    });
  }
}

template <int M, int N, int K, int NumWarps>
CUTLASS_DEVICE void cta_gemm_bf16_store(
    cutlass::bfloat16_t const* a,
    cutlass::bfloat16_t const* b,
    cutlass::bfloat16_t* c,
    int thread_idx) {
  cta_gemm_bf16<M, N, K, NumWarps>(
      a, K, 1, b, N, 1, thread_idx,
      [=] __device__(int row, int column, float value) {
        c[std::int64_t(row) * N + column] = cutlass::bfloat16_t(value);
      });
}

}  // namespace gdn_qsa::ppu
