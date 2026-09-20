#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "gdn_qsa/ppu/superchunk_schedule.hpp"

namespace {

using gdn_qsa::ppu::Problem;
using gdn_qsa::ppu::SuperchunkScheduler;

struct Result {
  std::int64_t cases = 0;
  std::int64_t chunks = 0;
  std::int64_t holes = 0;
  std::int64_t duplicates = 0;
  std::int64_t bad_mapping = 0;
  std::int64_t bad_scan = 0;
};

Result run(bool plant_drop_last, bool plant_local_lock) {
  Result result{};
  constexpr std::array<int, 5> batches{1, 2, 3, 7, 17};
  constexpr std::array<int, 13> sequences{
      1, 15, 16, 17, 63, 64, 65, 511, 512, 513, 2048, 8192, 32768};
  constexpr std::array<int, 4> q_heads{1, 4, 16, 32};
  constexpr std::array<int, 4> ratios{1, 2, 4, 8};
  constexpr std::array<int, 7> group_chunks{1, 2, 3, 8, 16, 32, 64};

  for (int batch : batches) {
    for (int sequence : sequences) {
      for (int qh : q_heads) {
        for (int ratio : ratios) {
          for (int gc : group_chunks) {
            Problem const p{batch, sequence, qh, qh * ratio, 128, 16, gc};
            if (!SuperchunkScheduler::valid_problem(p)) {
              ++result.bad_mapping;
              continue;
            }
            ++result.cases;
            int const chunks = SuperchunkScheduler::chunks_per_head(p);
            int const groups = SuperchunkScheduler::groups_per_head(p);
            int const cells = SuperchunkScheduler::group_cells(p);
            // Chunk coverage is identical for every (batch,value-head), so
            // exhaust one head's complete group domain and multiply only the
            // accounting total. This keeps the sweep exhaustive without
            // turning large batch/head products into a billion-element test.
            std::vector<int> visits(std::size_t(chunks), 0);
            int const emitted_groups =
                plant_drop_last && groups > 0 ? groups - 1 : groups;
            for (int group = 0; group < emitted_groups; ++group) {
              int const linear = group;
              auto const w = SuperchunkScheduler::group_work(linear, p);
              if (!w.valid || w.group < 0 || w.group >= groups ||
                  w.first_chunk != w.group * gc || w.chunk_count <= 0 ||
                  w.chunk_count > gc || w.first_chunk + w.chunk_count > chunks) {
                ++result.bad_mapping;
                continue;
              }
              for (int c = 0; c < w.chunk_count; ++c) {
                int const index = w.first_chunk + c;
                ++visits[std::size_t(index)];
                result.chunks += std::int64_t(batch) * p.value_heads;
              }
            }
            for (int count : visits) {
              result.holes += count == 0;
              result.duplicates += count > 1 ? count - 1 : 0;
            }
            // The production lock/group identity is the linear cell itself,
            // hence a bijection over [0,cells). A local group id aliases all
            // batch/head planes whenever more than one exists.
            if (plant_local_lock && batch * p.value_heads > 1) {
              result.duplicates +=
                  std::int64_t(groups) * (batch * p.value_heads - 1);
              result.holes += std::int64_t(cells - groups);
            }
            for (int offset = 1; offset < groups; offset <<= 1) {
              // Scan coordinates also repeat identically across batch/head;
              // exhaust the complete group axis of one head.
              for (int linear = 0; linear < groups; ++linear) {
                auto const group = SuperchunkScheduler::group_work(linear, p);
                auto const scan =
                    SuperchunkScheduler::scan_work(linear, offset, p);
                if (!scan.valid || scan.sequence_index != group.sequence_index ||
                    scan.value_head != group.value_head ||
                    scan.group != group.group ||
                    scan.compose != (group.group >= offset) ||
                    (scan.compose && scan.predecessor != group.group - offset)) {
                  ++result.bad_scan;
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}

bool pass(Result const& x) {
  return x.cases == 5 * 13 * 4 * 4 * 7 && x.chunks > 0 && x.holes == 0 &&
         x.duplicates == 0 && x.bad_mapping == 0 && x.bad_scan == 0;
}

}  // namespace

int main() {
  Result const good = run(false, false);
  Result const drop = run(true, false);
  Result const local = run(false, true);
  bool const negative_drop = !pass(drop) && drop.holes > 0;
  bool const negative_lock = !pass(local) &&
                             (local.holes > 0 || local.duplicates > 0);
  bool const ok = pass(good) && negative_drop && negative_lock;
  std::printf(
      "[ppu superchunk schedule] %s cases=%lld chunk-visits=%lld holes=%lld "
      "duplicates=%lld mapping_bad=%lld scan_bad=%lld "
      "drop-last=%s local-lock=%s\n",
      ok ? "PASS" : "FAIL", static_cast<long long>(good.cases),
      static_cast<long long>(good.chunks),
      static_cast<long long>(good.holes),
      static_cast<long long>(good.duplicates),
      static_cast<long long>(good.bad_mapping),
      static_cast<long long>(good.bad_scan),
      negative_drop ? "EXPECTED-RED/PASS" : "FAIL",
      negative_lock ? "EXPECTED-RED/PASS" : "FAIL");
  return ok ? 0 : 1;
}
