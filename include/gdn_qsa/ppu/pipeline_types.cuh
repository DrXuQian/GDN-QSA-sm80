#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "cutlass/bfloat16.h"
#include "gdn_qsa/ppu/backend.h"
#include "gdn_qsa/ppu/superchunk_schedule.hpp"

namespace gdn_qsa::ppu {

constexpr int kChunk = 16;
constexpr int kD = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Dynamic shared-memory ledger for the exact shipping kernels.  Keep this in
// the common type header so host admission tests can reason about residency
// without maintaining a second copy of the kernel structs.
constexpr std::size_t kPrepareSharedBytes = align_up(
    std::size_t(4 * kChunk * kD + 3 * kChunk * kChunk + kChunk) *
            sizeof(cutlass::bfloat16_t) +
        std::size_t(kChunk) * sizeof(float),
    128);
constexpr std::size_t kGroupSharedBytes = align_up(
    std::size_t(3 * kD * kD + 7 * kChunk * kD + kChunk * kChunk) *
        sizeof(cutlass::bfloat16_t),
    128);
constexpr std::size_t kReplaySharedBytes = align_up(
    std::size_t(2 * kD * kD + 7 * kChunk * kD + 2 * kChunk * kChunk) *
        sizeof(cutlass::bfloat16_t),
    128);

struct alignas(128) ChunkWorkspace {
  cutlass::bfloat16_t kd[kChunk * kD];
  cutlass::bfloat16_t qd[kChunk * kD];
  cutlass::bfloat16_t kr[kChunk * kD];
  cutlass::bfloat16_t inv[kChunk * kChunk];
  cutlass::bfloat16_t mqk[kChunk * kChunk];
  float gt;
};

struct WorkspaceLayout {
  std::size_t chunks_offset = 0;
  std::size_t a_offset = 0;
  std::size_t b_offset = 0;
  std::size_t scratch_a_offset = 0;
  std::size_t scratch_b_offset = 0;
  std::size_t total_bytes = 0;
  int chunk_cells = 0;
  int group_cells = 0;
  bool valid = false;
};

constexpr Problem scheduler_problem(gdn_qsa_ppu_problem_v1 const& p) {
  return Problem{
      p.batch, p.sequence, p.qk_heads, p.value_heads, p.head_dim,
      p.chunk, p.group_chunks};
}

constexpr WorkspaceLayout make_workspace_layout(
    gdn_qsa_ppu_problem_v1 const& p) {
  WorkspaceLayout layout{};
  Problem const scheduler = scheduler_problem(p);
  if (p.schema_version != GDN_QSA_PPU_SCHEMA_V1 ||
      !SuperchunkScheduler::valid_problem(scheduler)) {
    return layout;
  }
  std::int64_t const chunk_cells64 =
      std::int64_t(p.batch) * p.value_heads *
      SuperchunkScheduler::chunks_per_head(scheduler);
  int const group_cells = SuperchunkScheduler::group_cells(scheduler);
  if (chunk_cells64 <= 0 ||
      chunk_cells64 > std::numeric_limits<int>::max() || group_cells <= 0) {
    return layout;
  }
  layout.chunk_cells = int(chunk_cells64);
  layout.group_cells = group_cells;
  std::size_t cursor = 0;
  layout.chunks_offset = cursor;
  cursor = align_up(
      cursor + std::size_t(layout.chunk_cells) * sizeof(ChunkWorkspace), 128);
  std::size_t const matrix_bytes =
      std::size_t(layout.group_cells) * kD * kD *
      sizeof(cutlass::bfloat16_t);
  layout.a_offset = cursor;
  cursor = align_up(cursor + matrix_bytes, 128);
  layout.b_offset = cursor;
  cursor = align_up(cursor + matrix_bytes, 128);
  layout.scratch_a_offset = cursor;
  cursor = align_up(cursor + matrix_bytes, 128);
  layout.scratch_b_offset = cursor;
  cursor = align_up(cursor + matrix_bytes, 128);
  layout.total_bytes = cursor;
  layout.valid = cursor > 0;
  return layout;
}

struct WorkspaceView {
  ChunkWorkspace* chunks = nullptr;
  cutlass::bfloat16_t* a = nullptr;
  cutlass::bfloat16_t* b = nullptr;
  cutlass::bfloat16_t* scratch_a = nullptr;
  cutlass::bfloat16_t* scratch_b = nullptr;
};

inline WorkspaceView workspace_view(void* storage, WorkspaceLayout const& l) {
  auto* bytes = reinterpret_cast<std::uint8_t*>(storage);
  return WorkspaceView{
      reinterpret_cast<ChunkWorkspace*>(bytes + l.chunks_offset),
      reinterpret_cast<cutlass::bfloat16_t*>(bytes + l.a_offset),
      reinterpret_cast<cutlass::bfloat16_t*>(bytes + l.b_offset),
      reinterpret_cast<cutlass::bfloat16_t*>(bytes + l.scratch_a_offset),
      reinterpret_cast<cutlass::bfloat16_t*>(bytes + l.scratch_b_offset)};
}

}  // namespace gdn_qsa::ppu
