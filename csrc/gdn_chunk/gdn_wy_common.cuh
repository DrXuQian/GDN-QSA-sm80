#pragma once

#include "gdn_target.cuh"
#include "gdn_qsa/wy_contract.hpp"
#include "gdn_qsa/ppu/wy_mma.cuh"
#include "gdn_qsa/ppu/wy_delivery.cuh"

namespace gdn_qsa::wy {
struct Inputs {
  BF16 const *q, *k, *v, *beta;
  void const* g;
  float const* initial;
  bool gate_fp32;
  Shape shape;
};
struct Workspace {
  BF16 *w, *u, *snapshots, *vnew;
  float* gates;
};

template <int Rows, int Cols, int Threads>
__device__ void stage(BF16* dst, BF16 const* src, int64_t stride, int valid_rows) {
  // Each transaction is eight contiguous BF16 values, including under the
  // hardware swizzle. Tail transactions use zfill without an invalid pointer.
  for (int i = int(threadIdx.x); i < Rows * Cols / 8; i += Threads) {
    int const r = i / (Cols / 8), c = (i % (Cols / 8)) * 8;
    auto const* p = src + (r < valid_rows ? r * stride + c : 0);
    gdn_arch::async_copy16(dst + swizzle<Rows, Cols>(r, c), p, r < valid_rows);
  }
}
__device__ __forceinline__ void commit_wait() {
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();
}
__device__ __forceinline__ float gate(Inputs const& p, int64_t i) {
  return p.gate_fp32 ? static_cast<float const*>(p.g)[i]
                     : float(static_cast<BF16 const*>(p.g)[i]);
}


int configure_tiled(unsigned delivery);
int configure_state_ab(unsigned options);
int launch_state_ab(Inputs p, Workspace ws, float* final, gdn_arch::Stream stream, unsigned options);
int configure_stage_address(unsigned options);
int launch_address_prepare(Inputs p, Workspace ws, gdn_arch::Stream stream);
int launch_address_output(Inputs p, Workspace ws, BF16* output, gdn_arch::Stream stream);
int launch_tiled_prepare(Inputs p, Workspace ws, gdn_arch::Stream stream);
int launch_tiled_state(Inputs p, Workspace ws, float* final, gdn_arch::Stream stream, unsigned state_options);
int launch_tiled_output(Inputs p, Workspace ws, BF16* output, gdn_arch::Stream stream);
}  // namespace gdn_qsa::wy
