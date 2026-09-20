#include <array>
#include <cstdint>
#include <cstdio>

#include "gdn_qsa/ppu/pipeline_types.cuh"
#include "gdn_qsa/ppu/warp_mma.cuh"

namespace {

using Mma = gdn_qsa::ppu::WarpMmaBf16M16N16K16;

struct Coordinate {
  int row;
  int column;
};

Coordinate expected_operand(int lane, int slot) {
  return {
      lane / 4 + 8 * ((slot / 4) % 2),
      2 * (lane % 4) + (slot % 2) + 8 * ((slot / 2) % 2)};
}

Coordinate expected_output(int lane, int slot) {
  return {
      lane / 4 + 8 * ((slot / 4) % 2),
      lane % 4 + 4 * (slot % 4)};
}

struct Result {
  int visits = 0;
  int holes = 0;
  int duplicates = 0;
  int out_of_bounds = 0;
  int coordinate_mismatches = 0;
};

template <bool IsB, bool PlantSwap>
Result operand() {
  std::array<int, 16 * 16> owners{};
  Result result{};
  for (int lane = 0; lane < 32; ++lane) {
    auto visitor = [&](int slot, int row, int reduction) {
      Coordinate actual{row, reduction};
      if constexpr (PlantSwap) {
        int const tmp = actual.row;
        actual.row = actual.column;
        actual.column = tmp;
      }
      Coordinate const expected = expected_operand(lane, slot);
      result.coordinate_mismatches +=
          actual.row != expected.row || actual.column != expected.column;
      if (actual.row < 0 || actual.row >= 16 || actual.column < 0 ||
          actual.column >= 16) {
        ++result.out_of_bounds;
      } else {
        ++owners[std::size_t(actual.row * 16 + actual.column)];
        ++result.visits;
      }
    };
    if constexpr (IsB) {
      Mma::for_each_b_coordinate(lane, visitor);
    } else {
      Mma::for_each_a_coordinate(lane, visitor);
    }
  }
  for (int count : owners) {
    result.holes += count == 0;
    result.duplicates += count > 1 ? count - 1 : 0;
  }
  return result;
}

template <bool PlantRotate>
Result output() {
  std::array<int, 16 * 16> owners{};
  Result result{};
  for (int lane = 0; lane < 32; ++lane) {
    Mma::for_each_c_coordinate(lane, [&](int slot, int row, int column) {
      if constexpr (PlantRotate) column = (column + 1) % 16;
      Coordinate const expected = expected_output(lane, slot);
      result.coordinate_mismatches +=
          row != expected.row || column != expected.column;
      if (row < 0 || row >= 16 || column < 0 || column >= 16) {
        ++result.out_of_bounds;
      } else {
        ++owners[std::size_t(row * 16 + column)];
        ++result.visits;
      }
    });
  }
  for (int count : owners) {
    result.holes += count == 0;
    result.duplicates += count > 1 ? count - 1 : 0;
  }
  return result;
}

bool good(Result const& x) {
  return x.visits == 256 && x.holes == 0 && x.duplicates == 0 &&
         x.out_of_bounds == 0 && x.coordinate_mismatches == 0;
}

}  // namespace

int main() {
  Result const a = operand<false, false>();
  Result const b = operand<true, false>();
  Result const c = output<false>();
  Result const swap = operand<true, true>();
  Result const rotate = output<true>();
  bool const ok = good(a) && good(b) && good(c) && !good(swap) && !good(rotate);
  std::printf(
      "[ppu warp mma ownership] %s A=%d/%d/%d B=%d/%d/%d C=%d/%d/%d "
      "swap=%s rotate=%s shared_bytes=%zu/%zu/%zu\n",
      ok ? "PASS" : "FAIL", a.visits, a.holes, a.coordinate_mismatches,
      b.visits, b.holes, b.coordinate_mismatches,
      c.visits, c.holes, c.coordinate_mismatches,
      !good(swap) ? "EXPECTED-RED/PASS" : "FAIL",
      !good(rotate) ? "EXPECTED-RED/PASS" : "FAIL",
      gdn_qsa::ppu::kPrepareSharedBytes,
      gdn_qsa::ppu::kGroupSharedBytes,
      gdn_qsa::ppu::kReplaySharedBytes);
  return ok ? 0 : 1;
}
