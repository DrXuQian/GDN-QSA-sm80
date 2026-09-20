#pragma once

#include <cstddef>
#include <cstdio>

// Target primitives only. All scheduling, reset admission, register-state
// lifetimes, double buffering and Neumann expansion live in the original TUs.
#if defined(GDN_QSA_PPU)
#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <cute/atom/mma_traits_ppu0010.hpp>
#include <cute/arch/copy_ppu.hpp>
#include "gdn_qsa/ppu/shared_copy.cuh"
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#endif
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>

namespace gdn_arch {

#if defined(GDN_QSA_PPU)
using Stream = hggcStream_t;
using MmaBf16 = cute::PPU0010_16x16x16_F32BF16BF16F32_TN;
using MmaF16 = cute::PPU0010_16x16x16_F16F16F16F16_TN;
template <int R, int C> using RowLayout = gdn_qsa::ppu::RowLayout<R, C>;
template <int R, int C> using ColumnLayout = gdn_qsa::ppu::ColumnLayout<R, C>;
template <class Kernel>
inline void set_smem(Kernel kernel, std::size_t bytes) {
    auto status = hggcFuncSetAttribute(
        kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
    if (status != hggcSuccess) {
        std::fprintf(stderr, "[PPU GDN] dynamic shared-memory admission failed: %s\n",
                     hggcGetErrorString(status));
    }
}
inline void copy_device(void* dst, void const* src, std::size_t bytes, Stream s) {
    hggcMemcpyAsync(dst, src, bytes, hggcMemcpyDeviceToDevice, s);
}
#else
using Stream = cudaStream_t;
using MmaBf16 = cute::SM80_16x8x16_F32BF16BF16F32_TN;
using MmaF16 = cute::SM80_16x8x16_F16F16F16F16_TN;
template <int R, int C>
using RowLayout = decltype(cute::tile_to_shape(
    cute::GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
    cute::make_shape(cute::Int<R>{}, cute::Int<C>{}), cute::LayoutLeft{}));
template <int R, int C>
using ColumnLayout = decltype(cute::tile_to_shape(
    cute::GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
    cute::make_shape(cute::Int<R>{}, cute::Int<C>{}), cute::LayoutRight{}));
template <class Kernel>
inline void set_smem(Kernel kernel, std::size_t bytes) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
}
inline void copy_device(void* dst, void const* src, std::size_t bytes, Stream s) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
}
#endif

CUTE_DEVICE float bf16_to_float(cutlass::bfloat16_t x) {
    return static_cast<float>(x);
}
CUTE_DEVICE void async_copy16(void* dst, void const* src, bool pred) {
#if defined(GDN_QSA_PPU)
    cute::PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
        *reinterpret_cast<cute::uint128_t const*>(src),
        *reinterpret_cast<cute::uint128_t*>(dst), pred);
#else
    uint32_t address = cute::cast_smem_ptr_to_uint(dst);
    int const bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(address), "l"(src), "r"(bytes));
#endif
}
template <int Threads>
CUTE_DEVICE void compute_barrier() {
#if defined(GDN_QSA_PPU)
    asm volatile("ppu.bar.sync 8, %0;" :: "n"(Threads) : "memory");
#else
    asm volatile("bar.sync 8, %0;" :: "n"(Threads) : "memory");
#endif
}

// Every native PPU accumulator has four columns on each of its two rows.
// Keep all original beta/state update loops; only their coordinate ABI varies.
CUTE_HOST_DEVICE auto row_coordinate(int a, int d, int upper_row) {
#if defined(GDN_QSA_PPU)
    return a + 2 * d + 4 * upper_row;
#else
    return cute::make_coord(cute::make_coord(a, upper_row), 0, d);
#endif
}

template <class Element, class Mma>
CUTE_HOST_DEVICE auto copy_a(Mma const& mma) {
#if defined(GDN_QSA_PPU)
    return gdn_qsa::ppu::SharedCopy<gdn_qsa::ppu::CopyRole::A>{};
#else
    return cute::make_tiled_copy_A(cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>{}, mma);
#endif
}
template <class Element, class Mma>
CUTE_HOST_DEVICE auto copy_at(Mma const& mma) {
#if defined(GDN_QSA_PPU)
    return gdn_qsa::ppu::SharedCopy<gdn_qsa::ppu::CopyRole::AT>{};
#else
    return cute::make_tiled_copy_A(cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, Element>{}, mma);
#endif
}
template <class Element, class Mma>
CUTE_HOST_DEVICE auto copy_b(Mma const& mma) {
#if defined(GDN_QSA_PPU)
    return gdn_qsa::ppu::SharedCopy<gdn_qsa::ppu::CopyRole::B>{};
#else
    return cute::make_tiled_copy_B(cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>{}, mma);
#endif
}
template <class Element, class Mma>
CUTE_HOST_DEVICE auto load_c(Mma const& mma) {
#if defined(GDN_QSA_PPU)
    return gdn_qsa::ppu::SharedCopy<gdn_qsa::ppu::CopyRole::C>{};
#else
    return cute::make_tiled_copy_C(cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>{}, mma);
#endif
}
template <class Element, class Mma>
CUTE_HOST_DEVICE auto load_ct(Mma const& mma) {
#if defined(GDN_QSA_PPU)
    return gdn_qsa::ppu::SharedCopy<gdn_qsa::ppu::CopyRole::CT>{};
#else
    return cute::make_tiled_copy_C(cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, Element>{}, mma);
#endif
}
template <class Element, class Mma>
CUTE_HOST_DEVICE auto store_c(Mma const& mma) {
#if defined(GDN_QSA_PPU)
    return gdn_qsa::ppu::SharedCopy<gdn_qsa::ppu::CopyRole::StoreC>{};
#else
    return cute::make_tiled_copy_C(cute::Copy_Atom<cute::AutoVectorizingCopy, Element>{}, mma);
#endif
}

CUTE_DEVICE void result_to_b_words(uint32_t const* src, uint32_t* dst) {
#if defined(GDN_QSA_PPU)
    using gdn_qsa::ppu::FragmentMap;
    gdn_qsa::ppu::remap<FragmentMap::Result, FragmentMap::Operand, true>(src, dst);
#else
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) cute::SM75_U32x1_MOVM_T::copy(src[i], dst[i]);
#endif
}
CUTE_DEVICE void transpose_a_words(uint32_t const* src, uint32_t* dst) {
#if defined(GDN_QSA_PPU)
    using gdn_qsa::ppu::FragmentMap;
    gdn_qsa::ppu::remap<FragmentMap::Operand, FragmentMap::Operand, true>(src, dst);
#else
    result_to_b_words(src, dst);
#endif
}
CUTE_DEVICE void operand_to_result_words(uint32_t const* src, uint32_t* dst) {
#if defined(GDN_QSA_PPU)
    using gdn_qsa::ppu::FragmentMap;
    gdn_qsa::ppu::remap<FragmentMap::Operand, FragmentMap::Result, false>(src, dst);
#else
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) dst[i] = src[i];
#endif
}

// The original Neumann expansion stores all powers in A-register order.
// Preserve that invariant around the native PPU F16 MMA's C-register order.
CUTE_DEVICE void mma_f16_16x16(
    uint32_t* d, uint32_t const* a, uint32_t const* b, uint32_t const* c) {
#if defined(GDN_QSA_PPU)
    uint32_t acc[4], result[4];
    operand_to_result_words(c, acc);
    MmaF16::fma(result[0], result[1], result[2], result[3],
               a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
               acc[0], acc[1], acc[2], acc[3]);
    using gdn_qsa::ppu::FragmentMap;
    gdn_qsa::ppu::remap<FragmentMap::Result, FragmentMap::Operand, false>(result, d);
#else
    MmaF16::fma(d[0], d[1], a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1]);
    MmaF16::fma(d[2], d[3], a[0], a[1], a[2], a[3], b[2], b[3], c[2], c[3]);
#endif
}

template <class A, class B, class C, class Store>
CUTE_DEVICE void dot_16x16(A const& a, B const& b, C& c, int lane, Store store) {
    using namespace cute;
    auto mma = make_tiled_mma(MmaBf16{}, Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});
#if defined(GDN_QSA_PPU)
    auto thr = mma.get_slice(lane);
    auto acc = partition_fragment_C(mma, Shape<_16, _16>{});
    clear(acc);
    auto copyA = copy_a<cutlass::bfloat16_t>(mma);
    auto copyB = copy_b<cutlass::bfloat16_t>(mma);
    auto thrA = copyA.get_slice(lane);
    auto thrB = copyB.get_slice(lane);
    auto at = local_tile(a, Shape<_16, _16>{}, make_coord(0, 0));
    auto bt = local_tile(b, Shape<_16, _16>{}, make_coord(0, 0));
    auto ra = thr.partition_fragment_A(at);
    auto rb = thr.partition_fragment_B(bt);
    CUTE_UNROLL
    for (int k = 0; k < size<1>(a) / 16; ++k) {
        copy(copyA, thrA.partition_S(local_tile(a, Shape<_16, _16>{}, make_coord(0, k))), ra);
        copy(copyB, thrB.partition_S(local_tile(b, Shape<_16, _16>{}, make_coord(0, k))), rb);
        gemm(thr, ra(_, _, Int<0>{}), rb(_, _, Int<0>{}), acc);
    }
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
        auto rc = gdn_qsa::ppu::result_coord(lane, i);
        c(rc.row, rc.col) = store(acc(i));
    }
#else
    cooperative_gemm(lane, mma, 1.0f, a, b, 0.0f, c,
        identity{}, identity{}, identity{}, store,
        SM75_U32x4_LDSM_N{}, SM75_U32x4_LDSM_N{},
        SM75_U32x4_LDSM_N{}, AutoVectorizingCopy{});
#endif
}

}  // namespace gdn_arch
