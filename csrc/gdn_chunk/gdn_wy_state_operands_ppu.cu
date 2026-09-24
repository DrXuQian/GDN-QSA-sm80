// Opt-in address-only descendant of state_ab<true,true>. Keep that admitted
// body untouched; the separate TU is also the machine-code control boundary.
#include "gdn_wy_state_copy.cuh"
#include "gdn_qsa/ppu/wy_tiles.cuh"
#include "gdn_qsa/ppu/wy_state_operands.cuh"

namespace gdn_qsa::wy {

template <unsigned Rows, bool Transpose = false>
CUTE_DEVICE void operand_load(BF16 const* shared, unsigned row_tile, unsigned col_tile,
                              uint32_t (&r)[4]) {
  cute::PPU0010_TSM_LD_SWZL<BF16, 16, 16, true, Transpose>::copy(
      r, const_cast<BF16*>(shared + operand_cube<Rows>(row_tile, col_tile)), 0, 0);
}

__global__ void __launch_bounds__(StateTile::Threads)
gdn_wy_state_operands(Inputs p, Workspace ws, float* final) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<TiledStateStorage*>(storage);
  unsigned const tid = unsigned(threadIdx.x), warp = tid >> 5, lane = tid & 31u;
  unsigned const col_tile = StateOperandTiles::column(warp);
  unsigned const k_tile = StateOperandTiles::key(warp, 0), value_tile = StateOperandTiles::value(warp, 0);
  int const slice = int(blockIdx.x) % (Dim / ValueTile);
  int const bh = int(blockIdx.x) / (Dim / ValueTile);
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads;
  int const v0 = slice * ValueTile, qh = p.shape.q_head(h);
  float state[StateTile::KFragments][8];
  CUTE_UNROLL
  for (unsigned k = 0; k < StateTile::KFragments; ++k) {
    CUTE_UNROLL
    for (unsigned s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      unsigned const row = (k_tile + k) * 16u + unsigned(rc.row);
      unsigned const col = unsigned(v0) + col_tile * 16u + unsigned(rc.col);
      state[k][s] = p.initial ? p.initial[(int64_t(bh) * Dim + row) * Dim + col] : 0.0f;
    }
  }
  #pragma unroll 1
  for (int ct = 0; ct < p.shape.chunks(); ++ct) {
    int64_t const group = p.shape.group(b, h, ct);
    int const first = ct * Chunk, valid = min(Chunk, p.shape.sequence - first);
    state_stage<Chunk, Dim, StateTile::Threads>(sm.w, ws.w + tile_offset(group), Dim, Chunk);
    state_stage<Chunk, Dim, StateTile::Threads>(sm.k,
        p.k + p.shape.input(b, first, qh, p.shape.q_heads), int64_t(p.shape.q_heads) * Dim, valid);
    if (tid < Chunk) sm.g[tid] = ws.gates[group * Chunk + tid];
    CUTE_UNROLL
    for (unsigned k = 0; k < StateTile::KFragments; ++k) {
      CUTE_UNROLL
      for (unsigned s = 0; s < 8; ++s)
        sm.snapshot[operand_store<Dim>(k_tile + k, col_tile, lane, s)] = BF16(state[k][s]);
    }
    commit_wait();
    state_publish_bf16<Dim, ValueTile, StateTile::Threads>(sm.snapshot,
        ws.snapshots + state_offset(group) + v0, Dim);
    state_stage<Chunk, ValueTile, StateTile::Threads>(sm.u,
        ws.u + tile_offset(group) + v0, Dim, Chunk);
    cute::cp_async_fence();
    float value[StateTile::ValueFragments][8] = {};
    CUTE_UNROLL
    for (unsigned k = 0; k < Dim / 16; ++k) {
      uint32_t hs[4];
      operand_load<Dim, true>(sm.snapshot, k, col_tile, hs);
      CUTE_UNROLL
      for (unsigned r = 0; r < StateTile::ValueFragments; ++r) {
        uint32_t w[4];
        operand_load<Chunk>(sm.w, value_tile + r, k, w);
        bf16_mma(value[r], w, hs);
      }
    }
    cute::cp_async_wait<0>();
    __syncthreads();
    float const last = sm.g[valid - 1];
    float row_gate[StateTile::ValueFragments][StateGateRows::Count];
    CUTE_UNROLL
    for (unsigned r = 0; r < StateTile::ValueFragments; ++r) {
      CUTE_UNROLL
      for (unsigned half = 0; half < StateGateRows::Count; ++half) {
        unsigned const row = (value_tile + r) * 16u + StateGateRows::row(lane, half);
        row_gate[r][half] = expf(last - sm.g[row]);
      }
    }
    CUTE_UNROLL
    for (unsigned r = 0; r < StateTile::ValueFragments; ++r) {
      CUTE_UNROLL
      for (unsigned s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        unsigned const row = (value_tile + r) * 16u + unsigned(rc.row);
        unsigned const at = operand_store<Chunk>(value_tile + r, col_tile, lane, s);
        float const x = row < unsigned(valid) ? float(sm.u[at]) - value[r][s] : 0.0f;
        sm.u[at] = BF16(x);
        sm.scaled_v[at] = BF16(x * row_gate[r][StateGateRows::half(s)]);
      }
    }
    __syncthreads();
    state_publish_bf16<Chunk, ValueTile, StateTile::Threads>(sm.u,
        ws.vnew + tile_offset(group) + v0, Dim);
    float const decay = expf(last);
    CUTE_UNROLL
    for (unsigned k = 0; k < StateTile::KFragments; ++k) {
      CUTE_UNROLL
      for (unsigned s = 0; s < 8; ++s) state[k][s] *= decay;
    }
    CUTE_UNROLL
    for (unsigned r = 0; r < Chunk / 16; ++r) {
      uint32_t v[4];
      operand_load<Chunk, true>(sm.scaled_v, r, col_tile, v);
      CUTE_UNROLL
      for (unsigned k = 0; k < StateTile::KFragments; ++k) {
        uint32_t key[4];
        operand_load<Chunk, true>(sm.k, r, k_tile + k, key);
        bf16_mma(state[k], key, v);
      }
    }
    __syncthreads();
  }
  if (final) {
    CUTE_UNROLL
    for (unsigned k = 0; k < StateTile::KFragments; ++k) {
      CUTE_UNROLL
      for (unsigned s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.final_h[((k_tile + k) * 16u + unsigned(rc.row)) * ValueTile +
            col_tile * 16u + unsigned(rc.col)] = state[k][s];
      }
    }
    __syncthreads();
    state_publish_fp32<Dim, ValueTile, StateTile::Threads>(sm.final_h,
        final + int64_t(bh) * Dim * Dim + v0, Dim);
  }
}

int configure_state_operands() {
  return int(hggcFuncSetAttribute(gdn_wy_state_operands,
      hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(TiledStateStorage)));
}
int launch_state_operands(Inputs p, Workspace ws, float* final, gdn_arch::Stream stream) {
  unsigned const grid = unsigned(int64_t(p.shape.batch) * p.shape.value_heads * (Dim / ValueTile));
  gdn_wy_state_operands<<<grid, StateTile::Threads, sizeof(TiledStateStorage), stream>>>(p, ws, final);
  return int(hggcGetLastError());
}
}  // namespace gdn_qsa::wy
