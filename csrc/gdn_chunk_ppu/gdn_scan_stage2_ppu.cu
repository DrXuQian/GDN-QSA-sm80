#include <cstdint>

#include "cutlass/bfloat16.h"
#include "gdn_qsa/ppu/backend.h"
#include "gdn_qsa/ppu/warp_mma.cuh"

#if defined(__HGGCCC__)
#include <hggc_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace {

using Element = cutlass::bfloat16_t;
using WarpMma = gdn_qsa::ppu::WarpMmaBf16M16N16K16;

constexpr int kD = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMBlocks = kD / 16;
constexpr int kPackedNBlocks = (2 * kD) / 16;
constexpr int kKBlocks = kD / 16;

__global__ __launch_bounds__(kThreads) void scan_round_kernel(
    Element const* __restrict__ src_a,
    Element const* __restrict__ src_b,
    Element* __restrict__ dst_a,
    Element* __restrict__ dst_b,
    int groups,
    int offset) {
  int const batch_head = int(blockIdx.x);
  int const group = int(blockIdx.y);
  int const tid = int(threadIdx.x);
  int const warp = tid / 32;
  int const lane = tid % 32;
  std::int64_t const own_cell =
      (std::int64_t(batch_head) * groups + group) * kD * kD;

  if (group < offset) {
    for (int i = tid; i < kD * kD; i += kThreads) {
      dst_a[own_cell + i] = src_a[own_cell + i];
      dst_b[own_cell + i] = src_b[own_cell + i];
    }
    return;
  }

  std::int64_t const prior_cell =
      (std::int64_t(batch_head) * groups + group - offset) * kD * kD;
  Element const* const a2 = src_a + own_cell;
  Element const* const b2 = src_b + own_cell;
  Element const* const a1 = src_a + prior_cell;
  Element const* const b1 = src_b + prior_cell;

  // Packed affine composition:
  //   [A_out | B_tmp] = A2 @ [A1 | B1]
  //   B_out = B_tmp + B2
  // Eight warps cover all 8*16 output tiles with a fixed stride. Generated
  // operands are gathered into the PPU fragment map by WarpMma::accumulate.
  for (int tile = warp; tile < kMBlocks * kPackedNBlocks; tile += kWarps) {
    int const m_block = tile / kPackedNBlocks;
    int const n_block = tile - m_block * kPackedNBlocks;
    bool const is_b = n_block >= kMBlocks;
    int const output_n_block = is_b ? n_block - kMBlocks : n_block;
    Element const* const x = is_b ? b1 : a1;

    auto accumulator = WarpMma::make_accumulator();
    WarpMma::clear(accumulator);
#pragma unroll
    for (int k_block = 0; k_block < kKBlocks; ++k_block) {
      Element const* const a_tile =
          a2 + (m_block * 16) * kD + k_block * 16;
      // WarpMma computes A @ B^T. Viewing a row-major source X[K,N] as
      // B[N,K] means swapping its two logical strides.
      Element const* const b_tile =
          x + (k_block * 16) * kD + output_n_block * 16;
      WarpMma::accumulate(
          accumulator,
          a_tile, kD, 1,
          b_tile, 1, kD,
          lane);
    }

    Element* const destination =
        (is_b ? dst_b : dst_a) + own_cell +
        (m_block * 16) * kD + output_n_block * 16;
    Element const* const addend =
        b2 + (m_block * 16) * kD + output_n_block * 16;
    WarpMma::visit(accumulator, lane, [&](int row, int column, float value) {
      if (is_b) value += float(addend[row * kD + column]);
      destination[row * kD + column] = Element(value);
    });
  }
}

bool valid_arguments(
    void const* src_a, void const* src_b, void* dst_a, void* dst_b,
    int batch_heads, int groups, int offset) {
  return src_a != nullptr && src_b != nullptr && dst_a != nullptr &&
         dst_b != nullptr && batch_heads > 0 && groups > 0 && offset > 0 &&
         offset < groups;
}

}  // namespace

extern "C" int gdn_qsa_ppu_scan_round_bf16(
    std::uint16_t const* src_a,
    std::uint16_t const* src_b,
    std::uint16_t* dst_a,
    std::uint16_t* dst_b,
    int batch_heads,
    int groups,
    int offset,
    void* stream) {
  if (src_a == nullptr || src_b == nullptr || dst_a == nullptr ||
      dst_b == nullptr) {
    return GDN_QSA_PPU_NULL_POINTER;
  }
  if (!valid_arguments(
          src_a, src_b, dst_a, dst_b, batch_heads, groups, offset)) {
    return GDN_QSA_PPU_INVALID_PROBLEM;
  }

  auto const* a = reinterpret_cast<Element const*>(src_a);
  auto const* b = reinterpret_cast<Element const*>(src_b);
  auto* out_a = reinterpret_cast<Element*>(dst_a);
  auto* out_b = reinterpret_cast<Element*>(dst_b);
#if defined(__HGGCCC__)
  hggcStream_t const launch_stream = static_cast<hggcStream_t>(stream);
  (void)hggcGetLastError();
#else
  cudaStream_t const launch_stream = static_cast<cudaStream_t>(stream);
  (void)cudaGetLastError();
#endif
  scan_round_kernel<<<dim3(batch_heads, groups, 1), dim3(kThreads, 1, 1), 0,
                      launch_stream>>>(a, b, out_a, out_b, groups, offset);
#if defined(__HGGCCC__)
  return hggcPeekAtLastError() == hggcSuccess
      ? GDN_QSA_PPU_SUCCESS
      : GDN_QSA_PPU_RUNTIME_ERROR;
#else
  return cudaPeekAtLastError() == cudaSuccess
      ? GDN_QSA_PPU_SUCCESS
      : GDN_QSA_PPU_RUNTIME_ERROR;
#endif
}
