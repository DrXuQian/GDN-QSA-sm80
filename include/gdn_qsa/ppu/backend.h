#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

enum gdn_qsa_ppu_status {
  GDN_QSA_PPU_SUCCESS = 0,
  GDN_QSA_PPU_NULL_POINTER = 1,
  GDN_QSA_PPU_INVALID_PROBLEM = 2,
  GDN_QSA_PPU_RUNTIME_ERROR = 3,
};

enum { GDN_QSA_PPU_SCHEMA_V1 = 1 };

typedef struct gdn_qsa_ppu_problem_v1 {
  uint32_t schema_version;
  int32_t batch;
  int32_t sequence;
  int32_t qk_heads;
  int32_t value_heads;
  int32_t head_dim;
  int32_t chunk;
  int32_t group_chunks;
} gdn_qsa_ppu_problem_v1;

// Exact byte count required by gdn_qsa_ppu_forward_bf16_v1. The caller owns
// the workspace so allocation never hides inside a timed launch region.
uint64_t gdn_qsa_ppu_workspace_size_v1(
    gdn_qsa_ppu_problem_v1 const* problem);

// q/k: [B,S,Hq,128], v/out: [B,S,Hv,128], g/beta: [B,S,Hv]. All data
// pointers except final_state/workspace are BF16 bit patterns. final_state is
// BF16 [B,Hv,128,128]. g contains raw per-token natural-log decay.
int gdn_qsa_ppu_forward_bf16_v1(
    uint16_t const* q,
    uint16_t const* k,
    uint16_t const* v,
    uint16_t const* g,
    uint16_t const* beta,
    uint16_t* output,
    uint16_t* final_state,
    void* workspace,
    uint64_t workspace_bytes,
    gdn_qsa_ppu_problem_v1 const* problem,
    void* stream);

// One inclusive Hillis-Steele round over B*H independent group arrays.
// A/B tensors contain row-major BF16 bit patterns with logical shape
// [batch_heads, groups, 128, 128]. `stream` is an hggcStream_t cast to void*.
int gdn_qsa_ppu_scan_round_bf16(
    uint16_t const* src_a,
    uint16_t const* src_b,
    uint16_t* dst_a,
    uint16_t* dst_b,
    int batch_heads,
    int groups,
    int offset,
    void* stream);

#if defined(__cplusplus)
}
#endif
