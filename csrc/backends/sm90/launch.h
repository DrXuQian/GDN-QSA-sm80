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
    void* prepared = nullptr; // optional private scratch, only explicit two-launch build
};
void launch(Arguments const&, cudaStream_t);
#ifdef GDN_SM90_PRECOMPUTED_AUX
struct KernelResources { int threads, shared_bytes, registers, local_bytes, blocks_per_sm; };
struct PreparedResources { KernelResources prepare, state; };
PreparedResources precomputed_resources(bool gate_fp32, bool initial);
#endif
} // namespace gdn::sm90
