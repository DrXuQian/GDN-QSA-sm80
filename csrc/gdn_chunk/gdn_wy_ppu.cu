// Forward-only C64 WY candidate. Original reset/scan/serial kernels are
// deliberately separate. No approximate history reset occurs in this TU.
#include "gdn_target.cuh"
#include "gdn_qsa/wy_contract.hpp"
#include "gdn_qsa/ppu/wy_mma.cuh"

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
__device__ void commit_wait() {
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();
}
__device__ float gate(Inputs const& p, int64_t i) {
  return p.gate_fp32 ? static_cast<float const*>(p.g)[i]
                     : float(static_cast<BF16 const*>(p.g)[i]);
}

struct PrepareStorage {
  alignas(128) BF16 k[Chunk * Dim], v[Chunk * Dim];
  alignas(128) float lower[Chunk * Chunk], inverse[Chunk * Chunk];
  alignas(128) float temp[4][16 * 16];
  float prefix[Chunk], beta[Chunk];
};

__global__ void __launch_bounds__(ParallelThreads) gdn_wy_prepare(Inputs p, Workspace ws) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<PrepareStorage*>(storage);
  int const tid = int(threadIdx.x), warp = unsigned(threadIdx.x) >> 5, lane = unsigned(threadIdx.x) & 31;
  int const ct = int(blockIdx.x) % p.shape.chunks();
  int const bh = int(blockIdx.x) / p.shape.chunks();
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads;
  int const first = ct * Chunk, valid = min(Chunk, p.shape.sequence - first);
  int const qh = p.shape.q_head(h);
  int64_t const group = p.shape.group(b, h, ct);
  stage<Chunk, Dim, ParallelThreads>(sm.k, p.k + p.shape.input(b, first, qh, p.shape.q_heads),
                                    int64_t(p.shape.q_heads) * Dim, valid);
  stage<Chunk, Dim, ParallelThreads>(sm.v, p.v + p.shape.input(b, first, h, p.shape.value_heads),
                                    int64_t(p.shape.value_heads) * Dim, valid);
  if (tid < Chunk) {
    int64_t const gi = (int64_t(b) * p.shape.sequence + first + min(tid, valid - 1))
                      * p.shape.value_heads + h;
    float g = tid < valid ? gate(p, gi) : 0.0f;
    CUTE_UNROLL
    for (int offset = 1; offset < 32; offset *= 2) {
      float const other = __shfl_up_sync(0xffffffffu, g, offset);
      if (lane >= offset) g += other;
    }
    sm.prefix[tid] = g;
    sm.beta[tid] = tid < valid ? float(p.beta[gi]) : 0.0f;
  }
  commit_wait();
  if (tid >= 32 && tid < Chunk) sm.prefix[tid] += sm.prefix[31];
  __syncthreads();
  if (tid < Chunk) ws.gates[group * Chunk + tid] = sm.prefix[tid];

  // Ten lower triangular 16x16 products; upper entries are never read.
  for (int tile = warp; tile < 16; tile += 4) {
    int const br = tile / 4, bc = tile % 4;
    if (bc > br) continue;
    float acc[8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Dim; k += 16) {
      uint32_t a[4], bt[4];
      load<Chunk, Dim>(sm.k, br * 16, k, a);
      load<Chunk, Dim>(sm.k, bc * 16, k, bt);
      bf16_mma(acc, a, bt);
    }
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int const r = br * 16 + rc.row, c = bc * 16 + rc.col;
      sm.lower[r * Chunk + c] = r > c
          ? acc[s] * sm.beta[r] * expf(sm.prefix[r] - sm.prefix[c]) : 0.0f;
    }
  }
  __syncthreads();

  // Four independent unit-lower 16x16 diagonal solves in FP32. Each active
  // lane owns one entire inverse column; no CTA barrier per scalar row.
  float column[16];
  CUTE_UNROLL
  for (int r = 0; r < 16; ++r) {
    float x = r == (lane % 16) ? 1.0f : 0.0f;
    CUTE_UNROLL
    for (int k = 0; k < r; ++k)
      x -= sm.lower[(warp * 16 + r) * Chunk + warp * 16 + k] * column[k];
    column[r] = x;
    if (lane < 16) sm.inverse[(warp * 16 + r) * Chunk + warp * 16 + lane] = x;
  }
  __syncthreads();
  // Block forward substitution: R_ij = -R_ii sum(A_ip R_pj).
  #pragma unroll 1
  for (int gap = 1; gap < 4; ++gap) {
    int const br = warp + gap, bc = warp;
    if (br < 4) {
      float acc[8] = {};
      #pragma unroll 1
      for (int k = bc; k < br; ++k)
        tf32_product(acc, sm.lower + br * 16 * Chunk + k * 16, Chunk,
                     sm.inverse + k * 16 * Chunk + bc * 16, Chunk);
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.temp[warp][rc.row * 16 + rc.col] = acc[s];
      }
      __syncwarp();
      float merged[8] = {};
      tf32_product(merged, sm.inverse + br * 16 * Chunk + br * 16, Chunk, sm.temp[warp], 16);
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.inverse[(br * 16 + rc.row) * Chunk + bc * 16 + rc.col] = -merged[s];
      }
    }
    __syncthreads();
  }
  // lower is dead; reuse its storage for the BF16 inverse consumed by W/U.
  BF16* inv = reinterpret_cast<BF16*>(sm.lower);
  for (int i = tid; i < Chunk * Chunk; i += ParallelThreads) {
    int const r = i / Chunk, c = i % Chunk;
    inv[swizzle<Chunk, Chunk>(r, c)] = BF16(r >= c ? sm.inverse[i] : 0.0f);
  }
  for (int i = tid; i < Chunk * Dim; i += ParallelThreads) {
    int const r = i / Dim, c = i % Dim, at = swizzle<Chunk, Dim>(r, c);
    sm.k[at] = BF16(float(sm.k[at]) * sm.beta[r] * expf(sm.prefix[r]));
    sm.v[at] = BF16(float(sm.v[at]) * sm.beta[r]);
  }
  __syncthreads();
  #pragma unroll 1
  for (int tile = warp; tile < 32; tile += 4) {
    int const br = tile / 8, bc = tile % 8;
    float w[8] = {}, u[8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Chunk; k += 16) {
      uint32_t a[4], bk[4], bv[4];
      load<Chunk, Chunk>(inv, br * 16, k, a);
      load<Chunk, Dim, true>(sm.k, k, bc * 16, bk);
      load<Chunk, Dim, true>(sm.v, k, bc * 16, bv);
      bf16_mma(w, a, bk); bf16_mma(u, a, bv);
    }
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int64_t const at = tile_offset(group) + (br * 16 + rc.row) * Dim + bc * 16 + rc.col;
      ws.w[at] = BF16(w[s]); ws.u[at] = BF16(u[s]);
    }
  }
}

struct StateStorage {
  alignas(128) BF16 w[Chunk * Dim], k[Chunk * Dim];
  alignas(128) BF16 scaled_v[Chunk * ValueTile];
  float g[Chunk];
};

__global__ void __launch_bounds__(StateThreads) gdn_wy_state(Inputs p, Workspace ws, float* final) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<StateStorage*>(storage);
  int const tid = int(threadIdx.x), warp = unsigned(threadIdx.x) >> 5, lane = unsigned(threadIdx.x) & 31;
  int const slice = int(blockIdx.x) % (Dim / ValueTile);
  int const bh = int(blockIdx.x) / (Dim / ValueTile);
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads;
  int const v0 = state_v_start(slice, warp), qh = p.shape.q_head(h);
  float state[8][8];  // Lives across every chunk; native FP32 C fragments.
  CUTE_UNROLL
  for (int k = 0; k < 8; ++k) {
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int64_t const at = (int64_t(bh) * Dim + k * 16 + rc.row) * Dim + v0 + rc.col;
      state[k][s] = p.initial ? p.initial[at] : 0.0f;
    }
  }
  #pragma unroll 1
  for (int ct = 0; ct < p.shape.chunks(); ++ct) {
    int64_t const group = p.shape.group(b, h, ct);
    int const first = ct * Chunk, valid = min(Chunk, p.shape.sequence - first);
    stage<Chunk, Dim, StateThreads>(sm.w, ws.w + tile_offset(group), Dim, Chunk);
    stage<Chunk, Dim, StateThreads>(sm.k, p.k + p.shape.input(b, first, qh, p.shape.q_heads),
                                   int64_t(p.shape.q_heads) * Dim, valid);
    sm.g[tid] = ws.gates[group * Chunk + tid];
    CUTE_UNROLL
    for (int k = 0; k < 8; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        ws.snapshots[state_offset(group) + (k * 16 + rc.row) * Dim + v0 + rc.col] = BF16(state[k][s]);
      }
    }
    commit_wait();
    float value[4][8] = {};
    CUTE_UNROLL
    for (int k = 0; k < 8; ++k) {
      uint32_t hs[4];
      state_to_b(state[k], hs);
      CUTE_UNROLL
      for (int r = 0; r < 4; ++r) {
        uint32_t w[4];
        load<Chunk, Dim>(sm.w, r * 16, k * 16, w);
        bf16_mma(value[r], w, hs);
      }
    }
    float const last = sm.g[valid - 1];
    CUTE_UNROLL
    for (int r = 0; r < 4; ++r) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        int const row = r * 16 + rc.row, col = v0 + rc.col;
        int64_t const at = tile_offset(group) + row * Dim + col;
        float const x = row < valid ? float(ws.u[at]) - value[r][s] : 0.0f;
        ws.vnew[at] = BF16(x);
        sm.scaled_v[swizzle<Chunk, ValueTile>(row, warp * 16 + rc.col)] =
            BF16(x * expf(last - sm.g[row]));
      }
    }
    __syncthreads();
    float const decay = expf(last);
    CUTE_UNROLL
    for (int k = 0; k < 8; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) state[k][s] *= decay;
      #pragma unroll 1
      for (int r = 0; r < 4; ++r) {
        uint32_t a[4], v[4];
        load<Chunk, Dim, true>(sm.k, r * 16, k * 16, a);
        load<Chunk, ValueTile, true>(sm.scaled_v, r * 16, warp * 16, v);
        bf16_mma(state[k], a, v);
      }
    }
    __syncthreads();
  }
  if (final) {
    CUTE_UNROLL
    for (int k = 0; k < 8; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        final[(int64_t(bh) * Dim + k * 16 + rc.row) * Dim + v0 + rc.col] = state[k][s];
      }
    }
  }
}

struct OutputStorage {
  alignas(128) BF16 q[Chunk * Dim], v[Chunk * Dim], attn[Chunk * Chunk];
  union { alignas(128) BF16 k[Chunk * Dim]; alignas(128) BF16 h[Dim * Dim]; } stage;
  float g[Chunk];
};

__global__ void __launch_bounds__(ParallelThreads) gdn_wy_output(Inputs p, Workspace ws, BF16* output) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<OutputStorage*>(storage);
  int const tid = int(threadIdx.x), warp = unsigned(threadIdx.x) >> 5, lane = unsigned(threadIdx.x) & 31;
  int const ct = int(blockIdx.x) % p.shape.chunks(), bh = int(blockIdx.x) / p.shape.chunks();
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads, qh = p.shape.q_head(h);
  int const first = ct * Chunk, valid = min(Chunk, p.shape.sequence - first);
  int64_t const group = p.shape.group(b, h, ct);
  stage<Chunk, Dim, ParallelThreads>(sm.q, p.q + p.shape.input(b, first, qh, p.shape.q_heads),
                                    int64_t(p.shape.q_heads) * Dim, valid);
  stage<Chunk, Dim, ParallelThreads>(sm.stage.k, p.k + p.shape.input(b, first, qh, p.shape.q_heads),
                                    int64_t(p.shape.q_heads) * Dim, valid);
  if (tid < Chunk) sm.g[tid] = ws.gates[group * Chunk + tid];
  commit_wait();
  CUTE_UNROLL
  for (int c = 0; c < 4; ++c) {
    float acc[8] = {};
    CUTE_UNROLL
    for (int k = 0; k < 8; ++k) {
      uint32_t q[4], key[4];
      load<Chunk, Dim>(sm.q, warp * 16, k * 16, q);
      load<Chunk, Dim>(sm.stage.k, c * 16, k * 16, key);
      bf16_mma(acc, q, key);
    }
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int const row = warp * 16 + rc.row, col = c * 16 + rc.col;
      sm.attn[swizzle<Chunk, Chunk>(row, col)] = BF16(
          row >= col && row < valid ? acc[s] * expf(sm.g[row] - sm.g[col]) : 0.0f);
    }
  }
  __syncthreads();  // K's last reader must finish before the union becomes H.
  stage<Dim, Dim, ParallelThreads>(sm.stage.h, ws.snapshots + state_offset(group), Dim, Dim);
  stage<Chunk, Dim, ParallelThreads>(sm.v, ws.vnew + tile_offset(group), Dim, Chunk);
  commit_wait();
  #pragma unroll 1
  for (int c = 0; c < 8; ++c) {
    float acc[8] = {};
    CUTE_UNROLL
    for (int k = 0; k < 8; ++k) {
      uint32_t q[4], hreg[4];
      load<Chunk, Dim>(sm.q, warp * 16, k * 16, q);
      load<Dim, Dim, true>(sm.stage.h, k * 16, c * 16, hreg);
      bf16_mma(acc, q, hreg);
    }
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) acc[s] *= expf(sm.g[warp * 16 + result_coord(lane, s).row]);
    CUTE_UNROLL
    for (int k = 0; k < 4; ++k) {
      uint32_t a[4], v[4];
      load<Chunk, Chunk>(sm.attn, warp * 16, k * 16, a);
      load<Chunk, Dim, true>(sm.v, k * 16, c * 16, v);
      bf16_mma(acc, a, v);
    }
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int const row = warp * 16 + rc.row;
      if (row < valid)
        output[p.shape.input(b, first + row, h, p.shape.value_heads) + c * 16 + rc.col] =
            BF16(acc[s] * 0.08838834764831845f);
    }
  }
}
}  // namespace gdn_qsa::wy

extern "C" int gdn_wy_forward(
    void const* q, void const* k, void const* v, void const* g, void const* beta,
    float const* initial, void* output, float* final, void* w, void* u,
    void* snapshots, void* vnew, float* gates, int batch, int sequence,
    int q_heads, int value_heads, bool gate_fp32, gdn_arch::Stream stream) {
  using namespace gdn_qsa::wy;
  Inputs p{static_cast<BF16 const*>(q), static_cast<BF16 const*>(k),
           static_cast<BF16 const*>(v), static_cast<BF16 const*>(beta),
           g, initial, gate_fp32, {batch, sequence, q_heads, value_heads}};
  Workspace ws{static_cast<BF16*>(w), static_cast<BF16*>(u), static_cast<BF16*>(snapshots),
               static_cast<BF16*>(vnew), gates};
  // Match the original backend's explicit opt-in for >48 KiB shared memory.
  // An SDK/device resource refusal is a launch failure, never a fallback.
  auto status = hggcFuncSetAttribute(gdn_wy_prepare, hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(PrepareStorage));
  if (status != hggcSuccess) return int(status);
  status = hggcFuncSetAttribute(gdn_wy_state, hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(StateStorage));
  if (status != hggcSuccess) return int(status);
  status = hggcFuncSetAttribute(gdn_wy_output, hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(OutputStorage));
  if (status != hggcSuccess) return int(status);
  gdn_wy_prepare<<<unsigned(p.shape.groups()), ParallelThreads, sizeof(PrepareStorage), stream>>>(p, ws);
  status = hggcGetLastError();
  if (status != hggcSuccess) return int(status);
  gdn_wy_state<<<unsigned(int64_t(batch) * value_heads * (Dim / ValueTile)), StateThreads, sizeof(StateStorage), stream>>>(p, ws, final);
  status = hggcGetLastError();
  if (status != hggcSuccess) return int(status);
  gdn_wy_output<<<unsigned(p.shape.groups()), ParallelThreads, sizeof(OutputStorage), stream>>>(p, ws, static_cast<BF16*>(output));
  return int(hggcGetLastError());
}
