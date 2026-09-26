// Actual inverse fragment + scalar/packed arithmetic seam; no timing claim.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include "inverse_half2_add.cuh"

using namespace cute;
using Half = cutlass::half_t;
using Mma = decltype(make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                  Layout<Shape<_1, _1>>{}, Shape<_16, _32, _16>{}));
constexpr int edge_count = 32;
__device__ __constant__ uint16_t edges[edge_count] = {
  0x0000,0x8000,0x0001,0x8001,0x0002,0x8002,0x03ff,0x83ff,
  0x0400,0x8400,0x0401,0x8401,0x1000,0x9000,0x3555,0xb555,
  0x3bff,0xbbff,0x3c00,0xbc00,0x3c01,0xbc01,0x7bfe,0xfbfe,
  0x7bff,0xfbff,0x7c00,0xfc00,0x7c01,0xfc01,0x7e00,0xfe00};

__global__ void seam(unsigned long long* bad) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= 65536u * edge_count) return;
  auto acc = partition_fragment_C(Mma{}, Shape<_16, _32>{});
  auto lhs = make_tensor_like<Half>(acc);
  auto rhs = make_tensor_like<Half>(acc);
  auto want = make_tensor_like<Half>(acc);
  auto swapped = make_tensor_like<Half>(acc);
  auto dropped = make_tensor_like<Half>(acc);
  auto wrong_rhs = make_tensor_like<Half>(acc);
  static_assert(size(lhs) == 16);
  unsigned bits = id & 65535, edge = id >> 16;
  CUTE_UNROLL
  for (int i = 0; i < size(lhs); ++i) {
    unsigned code = (i & 1) ? (bits * 40503u + 17u) & 65535u : bits;
    lhs(i) = Half::bitcast(code);
    rhs(i) = Half::bitcast(edges[(edge + i) % edge_count]);
    want(i) = lhs(i) + rhs(i);
    swapped(i) = dropped(i) = lhs(i);
  }
  CUTE_UNROLL
  for (int i = 0; i < size(lhs); ++i) wrong_rhs(i) = rhs(i ^ 1);
  gdn_sm90::inverse_add_half_pairs(lhs, rhs);
  gdn_sm90::inverse_add_half_pairs(swapped, wrong_rhs);
  gdn_sm90::inverse_add_half_pairs(dropped, rhs);
  // A genuinely omitted pair, not a post-hoc count adjustment.
  dropped(size(dropped) - 2) = Half::bitcast(bits);
  dropped(size(dropped) - 1) = Half::bitcast((bits * 40503u + 17u) & 65535u);
  unsigned counts[3] = {};
  CUTE_UNROLL
  for (int i = 0; i < size(lhs); ++i) {
    counts[0] += lhs(i).raw() != want(i).raw();
    counts[1] += swapped(i).raw() != want(i).raw();
    counts[2] += dropped(i).raw() != want(i).raw();
  }
  for (int k = 0; k < 3; ++k) if (counts[k]) atomicAdd(bad + k, counts[k]);
}

void check(cudaError_t rc) {
  if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}

int main() {
  // The helper recasts this exact production C fragment, not an invented flat
  // layout. Give every scalar a distinct raw tag and inspect both packed lanes.
  auto acc = partition_fragment_C(Mma{}, Shape<_16, _32>{});
  auto frag = make_tensor_like<Half>(acc);
  for (int i = 0; i < size(frag); ++i) frag(i) = Half::bitcast(0x1000 + i);
  auto pairs = recast<cutlass::Array<Half,2>>(frag);
  for (int i = 0; i < size(pairs); ++i)
    for (int j = 0; j < 2; ++j)
      if (Half(pairs(i)[j]).raw() != frag(2*i+j).raw())
        throw std::runtime_error("actual fragment pair map changed");
  unsigned long long *device = nullptr, counts[3] = {};
  check(cudaMalloc(&device, sizeof(counts)));
  check(cudaMemset(device, 0, sizeof(counts)));
  seam<<<65536u * edge_count / 128, 128>>>(device);
  check(cudaGetLastError());
  check(cudaMemcpy(counts, device, sizeof(counts), cudaMemcpyDeviceToHost));
  check(cudaFree(device));
  printf("[inverse half2 seam] actual_fragment=16 pairs=8 scalar_checks=%u "
         "raw_bad=%llu swapped_negative=%llu dropped_negative=%llu\n",
         65536u*edge_count*16, counts[0], counts[1], counts[2]);
  if (counts[0] || !counts[1] || !counts[2]) return 1;
  puts("[inverse half2 seam] PASS: all half bit patterns x 32 boundary/special operands; two negatives red");
}
