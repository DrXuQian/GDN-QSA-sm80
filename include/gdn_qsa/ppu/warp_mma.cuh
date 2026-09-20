#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/tfloat32.h"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/arch/mma_ppu0010.hpp"
#include "cute/tensor.hpp"
#include "cute/algorithm/gemm.hpp"

namespace gdn_qsa::ppu {

// One logical 16x16x16 BF16 tile. Generated operands are gathered directly
// into the production PPU fragment map; this deliberately avoids assuming a
// CUDA ldmatrix-compatible shared-memory byte layout on PPU0010.
struct WarpMmaBf16M16N16K16 {
  using Element = cutlass::bfloat16_t;
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::MMA_Atom<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>{},
      cute::Layout<cute::Shape<cute::_1, cute::_1>>{},
      cute::Tile<cute::_16, cute::_16, cute::_16>{}));
  using OperandShape = cute::Shape<cute::_16, cute::_16>;
  using OutputShape = cute::Shape<cute::_16, cute::_16>;
  using ThreadMma = decltype(TiledMma{}.get_thread_slice(0));
  using OperandTensor = decltype(cute::make_tensor(
      cute::make_gmem_ptr(static_cast<Element const*>(nullptr)),
      cute::make_layout(OperandShape{},
                        cute::Stride<cute::_16, cute::_1>{})));
  using FragmentA = decltype(std::declval<ThreadMma&>().partition_fragment_A(
      std::declval<OperandTensor&>()));
  using FragmentB = decltype(std::declval<ThreadMma&>().partition_fragment_B(
      std::declval<OperandTensor&>()));
  using Accumulator = decltype(
      cute::partition_fragment_C(TiledMma{}, OutputShape{}));

  static constexpr int kThreads = int(cute::size(TiledMma{}));
  static constexpr int kOperandValuesPerThread =
      int(cute::size(FragmentA{}));
  static constexpr int kAccumulatorValuesPerThread =
      int(cute::size(Accumulator{}));

  static_assert(kThreads == 32);
  static_assert(kOperandValuesPerThread == 8);
  static_assert(kAccumulatorValuesPerThread == 8);

  template <class Visitor>
  CUTLASS_HOST_DEVICE static void for_each_a_coordinate(
      int lane, Visitor&& visitor) {
    auto identity = cute::make_identity_tensor(OperandShape{});
    auto coordinates = TiledMma{}.get_thread_slice(lane).partition_A(identity);
#pragma unroll
    for (int slot = 0; slot < int(cute::size(coordinates)); ++slot) {
      auto const domain = cute::idx2crd(slot, cute::shape(coordinates));
      auto const logical = coordinates(domain);
      visitor(slot, int(cute::get<0>(logical)), int(cute::get<1>(logical)));
    }
  }

  template <class Visitor>
  CUTLASS_HOST_DEVICE static void for_each_b_coordinate(
      int lane, Visitor&& visitor) {
    auto identity = cute::make_identity_tensor(OperandShape{});
    auto coordinates = TiledMma{}.get_thread_slice(lane).partition_B(identity);
#pragma unroll
    for (int slot = 0; slot < int(cute::size(coordinates)); ++slot) {
      auto const domain = cute::idx2crd(slot, cute::shape(coordinates));
      auto const logical = coordinates(domain);
      visitor(slot, int(cute::get<0>(logical)), int(cute::get<1>(logical)));
    }
  }

  template <class Visitor>
  CUTLASS_HOST_DEVICE static void for_each_c_coordinate(
      int lane, Visitor&& visitor) {
    auto identity = cute::make_identity_tensor(OutputShape{});
    auto coordinates = TiledMma{}.get_thread_slice(lane).partition_C(identity);
#pragma unroll
    for (int slot = 0; slot < int(cute::size(coordinates)); ++slot) {
      auto const domain = cute::idx2crd(slot, cute::shape(coordinates));
      auto const logical = coordinates(domain);
      visitor(slot, int(cute::get<0>(logical)), int(cute::get<1>(logical)));
    }
  }

  CUTLASS_HOST_DEVICE static Accumulator make_accumulator() {
    return cute::partition_fragment_C(TiledMma{}, OutputShape{});
  }

  CUTLASS_HOST_DEVICE static void clear(Accumulator& accumulator) {
    cute::clear(accumulator);
  }

  // C += A @ B^T. The two operands are logical row-major matrices expressed
  // by explicit row/reduction strides. They may reside in global or shared
  // storage; the fragment map, rather than a parallel lane formula, owns the
  // delivery ABI.
  CUTLASS_DEVICE static void accumulate(
      Accumulator& accumulator,
      Element const* a, std::int64_t a_row_stride,
      std::int64_t a_reduction_stride,
      Element const* b, std::int64_t b_row_stride,
      std::int64_t b_reduction_stride,
      int lane, int valid_m = 16, int valid_n = 16, int valid_k = 16) {
    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(lane);
    auto donor_a = cute::make_tensor(
        cute::make_gmem_ptr(a),
        cute::make_layout(OperandShape{}, cute::Stride<cute::_16, cute::_1>{}));
    auto donor_b = cute::make_tensor(
        cute::make_gmem_ptr(b),
        cute::make_layout(OperandShape{}, cute::Stride<cute::_16, cute::_1>{}));
    auto fragment_a = thread_mma.partition_fragment_A(donor_a);
    auto fragment_b = thread_mma.partition_fragment_B(donor_b);

    for_each_a_coordinate(lane, [&](int slot, int row, int reduction) {
      auto const domain = cute::idx2crd(slot, cute::shape(fragment_a));
      fragment_a(domain) = row < valid_m && reduction < valid_k
          ? a[std::int64_t(row) * a_row_stride +
              std::int64_t(reduction) * a_reduction_stride]
          : Element{};
    });
    for_each_b_coordinate(lane, [&](int slot, int row, int reduction) {
      auto const domain = cute::idx2crd(slot, cute::shape(fragment_b));
      fragment_b(domain) = row < valid_n && reduction < valid_k
          ? b[std::int64_t(row) * b_row_stride +
              std::int64_t(reduction) * b_reduction_stride]
          : Element{};
    });
    cute::gemm(tiled_mma, fragment_a, fragment_b, accumulator);
  }

  template <class Visitor>
  CUTLASS_DEVICE static void visit(
      Accumulator const& accumulator, int lane, Visitor&& visitor,
      int valid_m = 16, int valid_n = 16) {
    for_each_c_coordinate(lane, [&](int slot, int row, int column) {
      if (row < valid_m && column < valid_n) {
        auto const domain = cute::idx2crd(slot, cute::shape(accumulator));
        visitor(row, column, accumulator(domain));
      }
    });
  }
};

}  // namespace gdn_qsa::ppu
