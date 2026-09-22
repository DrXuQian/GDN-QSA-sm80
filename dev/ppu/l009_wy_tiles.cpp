#include <array>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_tiles.cuh"

using namespace gdn_qsa::wy;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;

// Independent hardware cube formula already anchored in l007; do not use
// the producer's RowLayout again as the consumer oracle.
int hw(int r, int c) {
  int const line = r / 4, vec = ((r % 4) * 2 + c / 8) ^ (line % 2);
  return (line * 32 + vec * 4 + (c % 8) / 2) * 2 + c % 2;
}

int state_exchange(int plant = 0) {
  int bad = 0;
  std::vector<int> sm(Dim * ValueTile, -1), owners(sm.size());
  int const warps = StateTile::Threads / 32;
  for (int warp = 0; warp < warps - int(plant == 1); ++warp)
    for (int k = 0; k < StateTile::KFragments; ++k)
      for (int lane = 0; lane < 32; ++lane)
        for (int slot = 0; slot < 8; ++slot) {
          auto const rc = result_coord(lane, slot);
          bad += Native::CLayout{}(lane, slot) != rc.row + 16 * rc.col;
          int row = StateTile::k_row(warp, k) + rc.row;
          if (plant == 2) row %= 64;  // lost upper K ownership
          int const col = StateTile::column(warp) + rc.col;
          int const at = swizzle<Dim, ValueTile>(row, col);
          sm[at] = row * ValueTile + col;
          ++owners[at];
        }
  for (int n : owners) bad += n != 1;
  // Every WH consumer reads all K fragments, including other warps' rows.
  for (int warp = 0; warp < warps; ++warp)
    for (int k = 0; k < Dim; k += 16)
      for (int lane = 0; lane < 32; ++lane)
        for (int slot = 0; slot < 8; ++slot) {
          int const coord = Native::BLayout{}(lane, slot);
          int const n = coord % 16, kr = coord / 16;
          int const base = swizzle<Dim, ValueTile>(k, StateTile::column(warp));
          int const offset = plant == 3 ? hw(n, kr) : hw(kr, n);
          int const expected = (k + kr) * ValueTile + StateTile::column(warp) + n;
          bad += sm[base + offset] != expected;
        }
  return bad;
}

struct Trace {
  int rows, cols;
  std::vector<std::vector<int>> order;
  explicit Trace(int r, int c) : rows(r), cols(c), order(r * c) {}
  void atom(int row, int col, int reduction) {
    for (int lane = 0; lane < 32; ++lane)
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        if (Native::CLayout{}(lane, s) != rc.row + 16 * rc.col)
          throw std::runtime_error("native fragment anchor differs");
        order.at((row + rc.row) * cols + col + rc.col).push_back(reduction);
      }
  }
  int check(std::vector<int> const& expected) const {
    int bad = 0;
    for (auto const& actual : order) bad += actual != expected;
    return bad;
  }
};

int reduction_order(int plant = 0) {
  Trace wh(Chunk, ValueTile), update(Dim, ValueTile);
  for (int warp = 0; warp < StateTile::Threads / 32; ++warp) {
    for (int k = 0; k < Dim; k += 16)
      for (int r = 0; r < StateTile::ValueFragments; ++r)
        if (!(plant == 4 && k == 112))
          wh.atom(StateTile::value_row(warp, r), StateTile::column(warp), k);
    for (int r = 0; r < Chunk; r += 16)
      for (int k = 0; k < StateTile::KFragments; ++k)
        update.atom(StateTile::k_row(warp, k), StateTile::column(warp), plant == 5 ? 48 - r : r);
  }
  Trace qk(Chunk, Chunk), qh_pv(Chunk, Dim), w(Chunk, Dim), u(Chunk, Dim);
  for (int warp = 0; warp < OutputTile::Threads / 32; ++warp) {
    for (int k = 0; k < Dim; k += 16)
      for (int c = 0; c < OutputTile::Fragments; ++c)
        qk.atom(OutputTile::row(warp), OutputTile::column(warp, c), k);
    for (int panel = 0; panel < Dim - int(plant == 6) * OutputTile::Panel; panel += OutputTile::Panel) {
      for (int k = 0; k < Dim; k += 16)
        for (int c = 0; c < OutputTile::Fragments; ++c)
          qh_pv.atom(OutputTile::row(warp), panel + OutputTile::column(warp, c), k);
      // A sentinel records the gate scaling BEFORE every PV contribution.
      for (int c = 0; c < OutputTile::Fragments; ++c)
        qh_pv.atom(OutputTile::row(warp), panel + OutputTile::column(warp, c), -1);
      for (int k = 0; k < Chunk; k += 16)
        for (int c = 0; c < OutputTile::Fragments; ++c)
          qh_pv.atom(OutputTile::row(warp), panel + OutputTile::column(warp, c), 128 + k);
    }
  }
  for (int warp = 0; warp < PrepareTile::Threads / 32; ++warp)
    for (int panel = 0; panel < Dim; panel += PrepareTile::Panel)
      for (int k = 0; k < Chunk; k += 16)
        for (int n = 0; n < PrepareTile::Fragments; ++n) {
          w.atom(PrepareTile::row(warp), panel + PrepareTile::column(n), k);
          if (!(plant == 8 && k == 48))
            u.atom(PrepareTile::row(warp), panel + PrepareTile::column(n), k);
        }
  return wh.check({0, 16, 32, 48, 64, 80, 96, 112}) + update.check({0, 16, 32, 48}) +
      qk.check({0, 16, 32, 48, 64, 80, 96, 112}) +
      qh_pv.check({0, 16, 32, 48, 64, 80, 96, 112, -1, 128, 144, 160, 176}) +
      w.check({0, 16, 32, 48}) + u.check({0, 16, 32, 48});
}

// 3 modes^3 base stages plus 3 options x9 base masks with tiled state.
// Options on non-tiled state must not be accepted and silently ignored.
int selectors(int plant = 0) {
  int accepted = 0, bad = 0;
  for (unsigned mask = 0; mask < 512; ++mask) {
    bool expected = mask < 256 && (!(mask & 192u) || (mask & 16u));
    for (int stage = 0; stage < 3; ++stage)
      expected &= !((mask & (1u << stage)) && (mask & (8u << stage)));
    bool const actual = plant == 7 ? mask < 64 : valid_delivery(mask);
    accepted += actual;
    bad += actual != expected;
  }
  return bad + (accepted != 54) + valid_delivery(1u << 31);
}

int main() {
  static_assert(offsetof(TiledStateStorage, u) >=
                offsetof(TiledStateStorage, snapshot) + sizeof(TiledStateStorage::snapshot),
                "U prefetch must not clobber live H for W@H");
  static_assert(offsetof(TiledStateStorage, final_h) == offsetof(TiledStateStorage, w));
  static_assert(sizeof(TiledOutputStorage::stage.h) >= sizeof(TiledOutputStorage::stage.output));
  if (state_exchange() || reduction_order() || selectors()) return 1;
  for (int plant = 1; plant <= 8; ++plant) {
    int const bad = plant <= 3 ? state_exchange(plant) : plant == 7 ? selectors(plant) : reduction_order(plant);
    if (!bad) throw std::runtime_error("tiled negative escaped");
    std::printf("[WY tiled negative] plant=%d bad=%d EXPECTED-RED/PASS\n", plant, bad);
  }
  std::puts("[WY tiles] native H exchange=8192 reads; independent scalar-coordinate reduction order=34816 outputs (W/U counted separately); selectors=512/54 valid (old27+state-options27); PASS device_execution=NOT_RUN");
}
