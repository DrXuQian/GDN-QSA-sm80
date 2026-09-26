#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

namespace gdn::sm90 {
struct Arguments {
    void const* q;
    void const* k;
    void const* v;
    void const* gate; // BF16 or FP32, natural-log increments, [B,T,Hv]
    void const* beta; // BF16, [B,T,Hv]
    float const* initial; // optional FP32 [B,Hv,K,V], V contiguous
    void* output; // BF16 [B,T,Hv,128]
    float* final; // optional FP32 [B,Hv,K,V], V contiguous
    int batch, length, qk_heads, v_heads;
    bool gate_fp32;
};
void launch(Arguments const&, cudaStream_t);
#ifdef GDN_SM90_ROLE_TRACE
constexpr int TraceCtas=64, TraceChunks=64, TraceRoles=3, TracePoints=16;
constexpr int TraceWords=TraceCtas*TraceChunks*TraceRoles*TracePoints;
void reset_role_trace(cudaStream_t);
void read_role_trace(void*, cudaStream_t);
#endif
} // namespace gdn::sm90
