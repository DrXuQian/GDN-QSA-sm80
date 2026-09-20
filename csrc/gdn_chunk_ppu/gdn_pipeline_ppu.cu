#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "cutlass/bfloat16.h"
#include "gdn_qsa/ppu/backend.h"
#include "gdn_qsa/ppu/matrix_ops.cuh"
#include "gdn_qsa/ppu/pipeline_types.cuh"

#if defined(__HGGCCC__)
#include <hggc_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace {

using Element = cutlass::bfloat16_t;
using gdn_qsa::ppu::ChunkWorkspace;
using gdn_qsa::ppu::WorkspaceLayout;
using gdn_qsa::ppu::WorkspaceView;
using gdn_qsa::ppu::kChunk;
using gdn_qsa::ppu::kD;
using gdn_qsa::ppu::kThreads;
using gdn_qsa::ppu::kWarps;

struct alignas(128) PrepareShared {
  Element kd[kChunk * kD];
  Element qd[kChunk * kD];
  Element ki[kChunk * kD];
  Element kr[kChunk * kD];
  Element l[kChunk * kChunk];
  Element inv[kChunk * kChunk];
  Element mqk[kChunk * kChunk];
  float gamma[kChunk];
  Element beta[kChunk];
};

struct alignas(128) GroupShared {
  Element state_a[kD * kD];
  Element state_b[kD * kD];
  Element state_scratch[kD * kD];
  Element kr[kChunk * kD];
  Element bkd[kChunk * kD];
  Element bkv[kChunk * kD];
  Element p[kChunk * kD];
  Element r[kChunk * kD];
  Element tmp_a[kChunk * kD];
  Element tmp_b[kChunk * kD];
  Element inv[kChunk * kChunk];
};

struct alignas(128) ReplayShared {
  Element state[kD * kD];
  Element state_scratch[kD * kD];
  Element kd[kChunk * kD];
  Element qd[kChunk * kD];
  Element kr[kChunk * kD];
  Element error[kChunk * kD];
  Element u[kChunk * kD];
  Element out_inter[kChunk * kD];
  Element out_intra[kChunk * kD];
  Element inv[kChunk * kChunk];
  Element mqk[kChunk * kChunk];
};

static_assert(sizeof(PrepareShared) == gdn_qsa::ppu::kPrepareSharedBytes);
static_assert(sizeof(GroupShared) == gdn_qsa::ppu::kGroupSharedBytes);
static_assert(sizeof(ReplayShared) == gdn_qsa::ppu::kReplaySharedBytes);
static_assert(sizeof(PrepareShared) < 48 * 1024);
static_assert(sizeof(GroupShared) < 160 * 1024);
static_assert(sizeof(ReplayShared) < 128 * 1024);

__device__ __forceinline__ float to_float(Element value) {
  return static_cast<float>(value);
}

__device__ __forceinline__ std::int64_t qk_offset(
    int batch, int token, int head, int dim,
    int sequence, int qk_heads) {
  return ((std::int64_t(batch) * sequence + token) * qk_heads + head) * kD + dim;
}

__device__ __forceinline__ std::int64_t vo_offset(
    int batch, int token, int head, int dim,
    int sequence, int value_heads) {
  return ((std::int64_t(batch) * sequence + token) * value_heads + head) * kD + dim;
}

__device__ __forceinline__ std::int64_t gate_offset(
    int batch, int token, int head, int sequence, int value_heads) {
  return (std::int64_t(batch) * sequence + token) * value_heads + head;
}

__global__ __launch_bounds__(kThreads) void prepare_kernel(
    Element const* __restrict__ q,
    Element const* __restrict__ k,
    Element const* __restrict__ g,
    Element const* __restrict__ beta,
    ChunkWorkspace* __restrict__ chunks,
    int batch,
    int sequence,
    int qk_heads,
    int value_heads,
    int chunks_per_head,
    float scale) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s = *reinterpret_cast<PrepareShared*>(storage);
  int const chunk = int(blockIdx.x);
  int const batch_head = int(blockIdx.y);
  int const b = batch_head / value_heads;
  int const vh = batch_head - b * value_heads;
  int const qh = vh / (value_heads / qk_heads);
  int const tid = int(threadIdx.x);
  int const first_token = chunk * kChunk;
  int const valid = sequence - first_token < kChunk
      ? sequence - first_token
      : kChunk;
  if (b >= batch || valid <= 0) return;

  if (tid < kChunk) {
    float cumulative = 0.0f;
    for (int row = 0; row <= tid && row < valid; ++row) {
      cumulative += to_float(g[gate_offset(
          b, first_token + row, vh, sequence, value_heads)]);
    }
    s.gamma[tid] = cumulative;
    s.beta[tid] = tid < valid
        ? beta[gate_offset(b, first_token + tid, vh, sequence, value_heads)]
        : Element(0.0f);
  }
  __syncthreads();

  float const total_gamma = valid > 0 ? s.gamma[valid - 1] : 0.0f;
  for (int index = tid; index < kChunk * kD; index += kThreads) {
    int const row = index / kD;
    int const dim = index - row * kD;
    if (row < valid) {
      float const gamma = s.gamma[row];
      float const kval = to_float(k[qk_offset(
          b, first_token + row, qh, dim, sequence, qk_heads)]);
      float const qval = to_float(q[qk_offset(
          b, first_token + row, qh, dim, sequence, qk_heads)]);
      s.kd[index] = Element(kval * expf(gamma));
      s.qd[index] = Element(qval * expf(gamma) * scale);
      s.ki[index] = Element(kval * expf(-gamma));
      s.kr[index] = Element(kval * expf(total_gamma - gamma));
    } else {
      s.kd[index] = Element(0.0f);
      s.qd[index] = Element(0.0f);
      s.ki[index] = Element(0.0f);
      s.kr[index] = Element(0.0f);
    }
  }
  __syncthreads();

  gdn_qsa::ppu::cta_gemm_bf16<kChunk, kChunk, kD, kWarps>(
      s.kd, kD, 1, s.ki, kD, 1, tid,
      [&] __device__(int row, int column, float value) {
        s.l[row * kChunk + column] = Element(value);
      });
  __syncthreads();
  gdn_qsa::ppu::cta_gemm_bf16<kChunk, kChunk, kD, kWarps>(
      s.qd, kD, 1, s.ki, kD, 1, tid,
      [&] __device__(int row, int column, float value) {
        s.mqk[row * kChunk + column] =
            row >= column && row < valid && column < valid
                ? Element(value)
                : Element(0.0f);
      });
  __syncthreads();

  // Independent columns solve (I + L)x=b, where L is the strictly-lower,
  // beta-row-scaled dot-product matrix used by torch.solve_triangular with
  // unitriangular=true.  The minus sign below is therefore part of the
  // recurrence, not an implementation convention.  This matches the actual
  // alternating Neumann series in the SM80 implementation without importing
  // its CUDA fp16 fragment layout into the PPU backend.
  if (tid < kChunk) {
    int const column = tid;
    for (int row = 0; row < kChunk; ++row) {
      if (row < column) {
        s.inv[row * kChunk + column] = Element(0.0f);
      } else if (row == column) {
        s.inv[row * kChunk + column] = Element(1.0f);
      } else {
        float sum = 0.0f;
        for (int inner = column; inner < row; ++inner) {
          float const lower =
              to_float(s.l[row * kChunk + inner]) * to_float(s.beta[row]);
          sum += lower * to_float(s.inv[inner * kChunk + column]);
        }
        s.inv[row * kChunk + column] = Element(-sum);
      }
    }
  }
  __syncthreads();

  ChunkWorkspace& destination =
      chunks[(batch_head * chunks_per_head) + chunk];
  for (int index = tid; index < kChunk * kD; index += kThreads) {
    destination.kd[index] = s.kd[index];
    destination.qd[index] = s.qd[index];
    destination.kr[index] = s.kr[index];
  }
  for (int index = tid; index < kChunk * kChunk; index += kThreads) {
    destination.inv[index] = s.inv[index];
    destination.mqk[index] = s.mqk[index];
  }
  if (tid == 0) destination.gt = expf(total_gamma);
}

__global__ __launch_bounds__(kThreads) void group_transfer_kernel(
    Element const* __restrict__ v,
    Element const* __restrict__ beta,
    ChunkWorkspace const* __restrict__ chunks,
    Element* __restrict__ group_a,
    Element* __restrict__ group_b,
    int sequence,
    int value_heads,
    int chunks_per_head,
    int groups_per_head,
    int group_chunks) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s = *reinterpret_cast<GroupShared*>(storage);
  int const linear = int(blockIdx.x);
  int const batch_head = linear / groups_per_head;
  int const group = linear - batch_head * groups_per_head;
  int const b = batch_head / value_heads;
  int const vh = batch_head - b * value_heads;
  int const first_chunk = group * group_chunks;
  int const count = chunks_per_head - first_chunk < group_chunks
      ? chunks_per_head - first_chunk
      : group_chunks;
  int const tid = int(threadIdx.x);

  for (int index = tid; index < kD * kD; index += kThreads) {
    int const row = index / kD;
    int const column = index - row * kD;
    s.state_a[index] = Element(row == column ? 1.0f : 0.0f);
    s.state_b[index] = Element(0.0f);
  }
  __syncthreads();

  for (int local = 0; local < count; ++local) {
    int const chunk = first_chunk + local;
    int const first_token = chunk * kChunk;
    int const valid = sequence - first_token < kChunk
        ? sequence - first_token
        : kChunk;
    ChunkWorkspace const& source = chunks[batch_head * chunks_per_head + chunk];
    for (int index = tid; index < kChunk * kD; index += kThreads) {
      int const row = index / kD;
      int const dim = index - row * kD;
      float const bet = row < valid
          ? to_float(beta[gate_offset(
                b, first_token + row, vh, sequence, value_heads)])
          : 0.0f;
      s.bkd[index] = Element(to_float(source.kd[index]) * bet);
      s.bkv[index] = row < valid
          ? Element(to_float(v[vo_offset(
                b, first_token + row, vh, dim, sequence, value_heads)]) * bet)
          : Element(0.0f);
      s.kr[index] = source.kr[index];
    }
    for (int index = tid; index < kChunk * kChunk; index += kThreads) {
      s.inv[index] = source.inv[index];
    }
    __syncthreads();

    gdn_qsa::ppu::cta_gemm_bf16_store<kChunk, kD, kChunk, kWarps>(
        s.inv, s.bkd, s.p, tid);
    __syncthreads();
    gdn_qsa::ppu::cta_gemm_bf16_store<kChunk, kD, kChunk, kWarps>(
        s.inv, s.bkv, s.r, tid);
    __syncthreads();
    gdn_qsa::ppu::cta_gemm_bf16_store<kChunk, kD, kD, kWarps>(
        s.p, s.state_a, s.tmp_a, tid);
    __syncthreads();
    gdn_qsa::ppu::cta_gemm_bf16_store<kChunk, kD, kD, kWarps>(
        s.p, s.state_b, s.tmp_b, tid);
    __syncthreads();

    gdn_qsa::ppu::cta_gemm_bf16<kD, kD, kChunk, kWarps>(
        s.kr, 1, kD, s.tmp_a, kD, 1, tid,
        [&] __device__(int row, int column, float value) {
          s.state_scratch[row * kD + column] = Element(value);
        });
    __syncthreads();
    for (int index = tid; index < kD * kD; index += kThreads) {
      s.state_a[index] = Element(
          source.gt * to_float(s.state_a[index]) -
          to_float(s.state_scratch[index]));
    }
    __syncthreads();

    gdn_qsa::ppu::cta_gemm_bf16<kD, kD, kChunk, kWarps>(
        s.kr, 1, kD, s.tmp_b, kD, 1, tid,
        [&] __device__(int row, int column, float value) {
          s.state_scratch[row * kD + column] = Element(value);
        });
    __syncthreads();
    for (int index = tid; index < kD * kD; index += kThreads) {
      s.state_b[index] = Element(
          source.gt * to_float(s.state_b[index]) -
          to_float(s.state_scratch[index]));
    }
    __syncthreads();
    gdn_qsa::ppu::cta_gemm_bf16<kD, kD, kChunk, kWarps>(
        s.kr, 1, kD, s.r, kD, 1, tid,
        [&] __device__(int row, int column, float value) {
          s.state_scratch[row * kD + column] = Element(value);
        });
    __syncthreads();
    for (int index = tid; index < kD * kD; index += kThreads) {
      s.state_b[index] = Element(
          to_float(s.state_b[index]) + to_float(s.state_scratch[index]));
    }
    __syncthreads();
  }

  std::int64_t const base = std::int64_t(linear) * kD * kD;
  for (int index = tid; index < kD * kD; index += kThreads) {
    group_a[base + index] = s.state_a[index];
    group_b[base + index] = s.state_b[index];
  }
}

__global__ __launch_bounds__(kThreads) void replay_kernel(
    Element const* __restrict__ v,
    Element const* __restrict__ beta,
    ChunkWorkspace const* __restrict__ chunks,
    Element const* __restrict__ prefix_b,
    Element* __restrict__ output,
    Element* __restrict__ final_state,
    int sequence,
    int value_heads,
    int chunks_per_head,
    int groups_per_head,
    int group_chunks) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& s = *reinterpret_cast<ReplayShared*>(storage);
  int const linear = int(blockIdx.x);
  int const batch_head = linear / groups_per_head;
  int const group = linear - batch_head * groups_per_head;
  int const b = batch_head / value_heads;
  int const vh = batch_head - b * value_heads;
  int const first_chunk = group * group_chunks;
  int const count = chunks_per_head - first_chunk < group_chunks
      ? chunks_per_head - first_chunk
      : group_chunks;
  int const tid = int(threadIdx.x);

  if (group == 0) {
    for (int index = tid; index < kD * kD; index += kThreads) {
      s.state[index] = Element(0.0f);
    }
  } else {
    std::int64_t const prefix =
        std::int64_t(linear - 1) * kD * kD;
    for (int index = tid; index < kD * kD; index += kThreads) {
      s.state[index] = prefix_b[prefix + index];
    }
  }
  __syncthreads();

  for (int local = 0; local < count; ++local) {
    int const chunk = first_chunk + local;
    int const first_token = chunk * kChunk;
    int const valid = sequence - first_token < kChunk
        ? sequence - first_token
        : kChunk;
    ChunkWorkspace const& source = chunks[batch_head * chunks_per_head + chunk];
    for (int index = tid; index < kChunk * kD; index += kThreads) {
      s.kd[index] = source.kd[index];
      s.qd[index] = source.qd[index];
      s.kr[index] = source.kr[index];
    }
    for (int index = tid; index < kChunk * kChunk; index += kThreads) {
      s.inv[index] = source.inv[index];
      s.mqk[index] = source.mqk[index];
    }
    __syncthreads();

    gdn_qsa::ppu::cta_gemm_bf16_store<kChunk, kD, kD, kWarps>(
        s.kd, s.state, s.error, tid);
    __syncthreads();
    for (int index = tid; index < kChunk * kD; index += kThreads) {
      int const row = index / kD;
      int const dim = index - row * kD;
      float const bet = row < valid
          ? to_float(beta[gate_offset(
                b, first_token + row, vh, sequence, value_heads)])
          : 0.0f;
      float const value = row < valid
          ? to_float(v[vo_offset(
                b, first_token + row, vh, dim, sequence, value_heads)])
          : 0.0f;
      s.error[index] = Element((value - to_float(s.error[index])) * bet);
    }
    __syncthreads();
    gdn_qsa::ppu::cta_gemm_bf16_store<kChunk, kD, kChunk, kWarps>(
        s.inv, s.error, s.u, tid);
    __syncthreads();
    gdn_qsa::ppu::cta_gemm_bf16_store<kChunk, kD, kD, kWarps>(
        s.qd, s.state, s.out_inter, tid);
    __syncthreads();
    gdn_qsa::ppu::cta_gemm_bf16_store<kChunk, kD, kChunk, kWarps>(
        s.mqk, s.u, s.out_intra, tid);
    __syncthreads();
    for (int index = tid; index < valid * kD; index += kThreads) {
      int const row = index / kD;
      int const dim = index - row * kD;
      output[vo_offset(
          b, first_token + row, vh, dim, sequence, value_heads)] = Element(
          to_float(s.out_inter[index]) + to_float(s.out_intra[index]));
    }

    gdn_qsa::ppu::cta_gemm_bf16<kD, kD, kChunk, kWarps>(
        s.kr, 1, kD, s.u, kD, 1, tid,
        [&] __device__(int row, int column, float value) {
          s.state_scratch[row * kD + column] = Element(value);
        });
    __syncthreads();
    for (int index = tid; index < kD * kD; index += kThreads) {
      s.state[index] = Element(
          source.gt * to_float(s.state[index]) +
          to_float(s.state_scratch[index]));
    }
    __syncthreads();
  }

  if (final_state != nullptr && group == groups_per_head - 1) {
    std::int64_t const base = std::int64_t(batch_head) * kD * kD;
    for (int index = tid; index < kD * kD; index += kThreads) {
      final_state[base + index] = s.state[index];
    }
  }
}

#if defined(__HGGCCC__)
using Stream = hggcStream_t;
template <class Kernel>
bool configure_smem(Kernel kernel, int bytes) {
  if (bytes < 48 * 1024) return true;
  return hggcFuncSetAttribute(
             kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes) ==
         hggcSuccess;
}
bool launch_ok() { return hggcPeekAtLastError() == hggcSuccess; }
void clear_error() { (void)hggcGetLastError(); }
#else
using Stream = cudaStream_t;
template <class Kernel>
bool configure_smem(Kernel kernel, int bytes) {
  if (bytes < 48 * 1024) return true;
  return cudaFuncSetAttribute(
             kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) ==
         cudaSuccess;
}
bool launch_ok() { return cudaPeekAtLastError() == cudaSuccess; }
void clear_error() { (void)cudaGetLastError(); }
#endif

bool configure_all() {
  return configure_smem(prepare_kernel, int(sizeof(PrepareShared))) &&
         configure_smem(group_transfer_kernel, int(sizeof(GroupShared))) &&
         configure_smem(replay_kernel, int(sizeof(ReplayShared)));
}

}  // namespace

extern "C" std::uint64_t gdn_qsa_ppu_workspace_size_v1(
    gdn_qsa_ppu_problem_v1 const* problem) {
  if (problem == nullptr) return 0;
  WorkspaceLayout const layout = gdn_qsa::ppu::make_workspace_layout(*problem);
  return layout.valid ? std::uint64_t(layout.total_bytes) : 0;
}

extern "C" int gdn_qsa_ppu_forward_bf16_v1(
    std::uint16_t const* q,
    std::uint16_t const* k,
    std::uint16_t const* v,
    std::uint16_t const* g,
    std::uint16_t const* beta,
    std::uint16_t* output,
    std::uint16_t* final_state,
    void* workspace,
    std::uint64_t workspace_bytes,
    gdn_qsa_ppu_problem_v1 const* problem,
    void* stream) {
  if (q == nullptr || k == nullptr || v == nullptr || g == nullptr ||
      beta == nullptr || output == nullptr || workspace == nullptr ||
      problem == nullptr) {
    return GDN_QSA_PPU_NULL_POINTER;
  }
  WorkspaceLayout const layout = gdn_qsa::ppu::make_workspace_layout(*problem);
  if (!layout.valid || workspace_bytes < layout.total_bytes) {
    return GDN_QSA_PPU_INVALID_PROBLEM;
  }
  if (!configure_all()) return GDN_QSA_PPU_RUNTIME_ERROR;
  clear_error();

  WorkspaceView const ws = gdn_qsa::ppu::workspace_view(workspace, layout);
  int const chunks_per_head =
      (problem->sequence + kChunk - 1) / kChunk;
  int const groups_per_head =
      (chunks_per_head + problem->group_chunks - 1) / problem->group_chunks;
  int const batch_heads = problem->batch * problem->value_heads;
  float const scale = 1.0f / sqrtf(float(kD));
  Stream const launch_stream = static_cast<Stream>(stream);

  prepare_kernel<<<dim3(chunks_per_head, batch_heads, 1), dim3(kThreads, 1, 1),
                   sizeof(PrepareShared), launch_stream>>>(
      reinterpret_cast<Element const*>(q),
      reinterpret_cast<Element const*>(k),
      reinterpret_cast<Element const*>(g),
      reinterpret_cast<Element const*>(beta),
      ws.chunks,
      problem->batch,
      problem->sequence,
      problem->qk_heads,
      problem->value_heads,
      chunks_per_head,
      scale);
  if (!launch_ok()) return GDN_QSA_PPU_RUNTIME_ERROR;

  group_transfer_kernel<<<dim3(layout.group_cells, 1, 1),
                          dim3(kThreads, 1, 1), sizeof(GroupShared),
                          launch_stream>>>(
      reinterpret_cast<Element const*>(v),
      reinterpret_cast<Element const*>(beta),
      ws.chunks,
      ws.a,
      ws.b,
      problem->sequence,
      problem->value_heads,
      chunks_per_head,
      groups_per_head,
      problem->group_chunks);
  if (!launch_ok()) return GDN_QSA_PPU_RUNTIME_ERROR;

  Element* current_a = ws.a;
  Element* current_b = ws.b;
  Element* next_a = ws.scratch_a;
  Element* next_b = ws.scratch_b;
  for (int offset = 1; offset < groups_per_head; offset <<= 1) {
    int const status = gdn_qsa_ppu_scan_round_bf16(
        reinterpret_cast<std::uint16_t const*>(current_a),
        reinterpret_cast<std::uint16_t const*>(current_b),
        reinterpret_cast<std::uint16_t*>(next_a),
        reinterpret_cast<std::uint16_t*>(next_b),
        batch_heads,
        groups_per_head,
        offset,
        stream);
    if (status != GDN_QSA_PPU_SUCCESS) return status;
    std::swap(current_a, next_a);
    std::swap(current_b, next_b);
  }

  replay_kernel<<<dim3(layout.group_cells, 1, 1), dim3(kThreads, 1, 1),
                  sizeof(ReplayShared), launch_stream>>>(
      reinterpret_cast<Element const*>(v),
      reinterpret_cast<Element const*>(beta),
      ws.chunks,
      current_b,
      reinterpret_cast<Element*>(output),
      reinterpret_cast<Element*>(final_state),
      problem->sequence,
      problem->value_heads,
      chunks_per_head,
      groups_per_head,
      problem->group_chunks);
  return launch_ok() ? GDN_QSA_PPU_SUCCESS : GDN_QSA_PPU_RUNTIME_ERROR;
}
