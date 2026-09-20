#pragma once

#include <cstdint>
#include <limits>

namespace gdn_qsa::ppu {

constexpr int ceil_div(int value, int divisor) {
  return divisor > 0 ? (value + divisor - 1) / divisor : 0;
}

struct Problem {
  int batch = 0;
  int sequence = 0;
  int qk_heads = 0;
  int value_heads = 0;
  int head_dim = 0;
  int chunk = 16;
  int group_chunks = 0;
};

struct GroupWork {
  int sequence_index = 0;
  int value_head = 0;
  int group = 0;
  int first_chunk = 0;
  int chunk_count = 0;
  bool valid = false;
};

struct ScanWork {
  int sequence_index = 0;
  int value_head = 0;
  int group = 0;
  int predecessor = 0;
  bool compose = false;
  bool valid = false;
};

struct SuperchunkScheduler {
  static constexpr bool valid_problem(Problem const& p) {
    if (p.batch <= 0 || p.sequence <= 0 || p.qk_heads <= 0 ||
        p.value_heads <= 0 || p.head_dim != 128 || p.chunk != 16 ||
        p.group_chunks <= 0 || p.value_heads % p.qk_heads != 0) {
      return false;
    }
    auto const groups = int64_t(p.batch) * p.value_heads * groups_per_head(p);
    return groups > 0 && groups <= std::numeric_limits<int>::max();
  }

  static constexpr int chunks_per_head(Problem const& p) {
    return ceil_div(p.sequence, p.chunk);
  }

  static constexpr int groups_per_head(Problem const& p) {
    return ceil_div(chunks_per_head(p), p.group_chunks);
  }

  static constexpr int group_cells(Problem const& p) {
    if (!valid_problem_shallow(p)) return 0;
    auto const cells = int64_t(p.batch) * p.value_heads * groups_per_head(p);
    return cells <= std::numeric_limits<int>::max() ? int(cells) : 0;
  }

  static constexpr GroupWork group_work(int linear, Problem const& p) {
    GroupWork w{};
    int const groups = groups_per_head(p);
    int const cells = group_cells(p);
    if (linear < 0 || linear >= cells || groups <= 0) return w;
    int const bh = linear / groups;
    w.group = linear - bh * groups;
    w.sequence_index = bh / p.value_heads;
    w.value_head = bh - w.sequence_index * p.value_heads;
    w.first_chunk = w.group * p.group_chunks;
    int const remaining = chunks_per_head(p) - w.first_chunk;
    w.chunk_count = remaining < p.group_chunks ? remaining : p.group_chunks;
    w.valid = w.chunk_count > 0;
    return w;
  }

  static constexpr ScanWork scan_work(
      int linear, int offset, Problem const& p) {
    ScanWork w{};
    GroupWork const g = group_work(linear, p);
    if (!g.valid || offset <= 0) return w;
    w.sequence_index = g.sequence_index;
    w.value_head = g.value_head;
    w.group = g.group;
    w.predecessor = g.group - offset;
    w.compose = w.predecessor >= 0;
    w.valid = true;
    return w;
  }

 private:
  static constexpr bool valid_problem_shallow(Problem const& p) {
    return p.batch > 0 && p.sequence > 0 && p.value_heads > 0 &&
           p.chunk > 0 && p.group_chunks > 0;
  }
};

}  // namespace gdn_qsa::ppu
