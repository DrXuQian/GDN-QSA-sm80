#pragma once

#include <cstdint>
#include <type_traits>

namespace gdn_qsa::wy {
constexpr int Chunk = 64;
constexpr int Dim = 128;
constexpr int ValueTile = 32;
constexpr int StateThreads = 64;
constexpr int ParallelThreads = 128;

constexpr unsigned StateAddress = 64u;
constexpr unsigned StateRowReuse = 128u;
constexpr unsigned StateOptions = StateAddress | StateRowReuse;
constexpr unsigned PrepareAddress = 256u;
constexpr unsigned OutputAddress = 512u;
constexpr unsigned StageAddressOptions = PrepareAddress | OutputAddress;
constexpr unsigned PrepareRowsShared = 1024u;
constexpr unsigned PrepareRowsWarp = 2048u;
constexpr unsigned PrepareRowsOptions = PrepareRowsShared | PrepareRowsWarp;
constexpr unsigned StateOperands = 4096u;

constexpr bool state_operands_selected(unsigned delivery) { return (delivery & StateOperands) != 0; }

constexpr bool valid_prepare_rows(unsigned mask) {
  unsigned const rows = mask & PrepareRowsOptions;
  return rows != PrepareRowsOptions && (!rows || (mask & PrepareAddress));
}

// The same typed mode must select both the attribute target and actual launch.
template <class Visitor>
constexpr int visit_prepare_rows(unsigned options, Visitor visitor, int invalid) {
  switch (options) {
    case PrepareRowsShared: return visitor(std::integral_constant<int, 1>{});
    case PrepareRowsWarp: return visitor(std::integral_constant<int, 2>{});
    default: return invalid;
  }
}

struct StageAddressSelection { bool prepare, output; };
constexpr StageAddressSelection stage_address_selection(unsigned delivery) {
  return {bool(delivery & PrepareAddress), bool(delivery & OutputAddress)};
}

// Each stage has mutually exclusive legacy-packed and tiled selectors.
// State options require the tiled state; they are never silently ignored.
// PrepareAddress extends scalar prepare, never overrides packed/tiled prepare.
// OutputAddress requires the tiled output. Row caches require PrepareAddress
// and are mutually exclusive. Operand offsets extend state-both only.
// All old masks0..4095 keep their meaning.
constexpr bool valid_delivery(unsigned mask) {
  return mask < 8192 && ((mask & 7u) & ((mask >> 3) & 7u)) == 0 &&
         (!(mask & StateOptions) || (mask & 16u)) &&
         (!(mask & PrepareAddress) || !(mask & 9u)) &&
         (!(mask & OutputAddress) || (mask & 32u)) && valid_prepare_rows(mask) &&
         (!state_operands_selected(mask) || (mask & (StateOptions | 16u)) == (StateOptions | 16u));
}

// One typed selector for BOTH resource configuration and the actual launch.
// The host gate visits this same function; equal numerical answers cannot
// conceal an ignored performance-only option.
template <class Visitor>
constexpr int visit_state_options(unsigned options, Visitor visitor, int invalid) {
  switch (options) {
    case StateAddress: return visitor(std::true_type{}, std::false_type{});
    case StateRowReuse: return visitor(std::false_type{}, std::true_type{});
    case StateOptions: return visitor(std::true_type{}, std::true_type{});
    default: return invalid;
  }
}

struct Shape {
  int batch, sequence, q_heads, value_heads;
  constexpr int chunks() const { return (sequence - 1) / Chunk + 1; }
  constexpr int q_head(int h) const { return h / (value_heads / q_heads); }
  constexpr int64_t groups() const { return int64_t(batch) * value_heads * chunks(); }
  constexpr int64_t group(int b, int h, int t) const {
    return (int64_t(b) * value_heads + h) * chunks() + t;
  }
  constexpr int64_t input(int b, int t, int h, int heads) const {
    return ((int64_t(b) * sequence + t) * heads + h) * Dim;
  }
};

// One warp owns all K rows for 16 independent V columns. K is never split
// across CTAs: the update requires no inter-CTA reduction or publication.
constexpr int state_v_start(int slice, int warp) { return slice * ValueTile + warp * 16; }
constexpr int state_k_start(int tile) { return tile * 16; }
constexpr int64_t tile_offset(int64_t group) { return group * Chunk * Dim; }
constexpr int64_t state_offset(int64_t group) { return group * Dim * Dim; }
}  // namespace gdn_qsa::wy
