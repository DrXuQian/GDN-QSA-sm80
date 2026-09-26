// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "prepared_aux_layout.cuh"
#include "scalar_gdn_aux.cuh"
#include <cutlass/device_kernel.h>

namespace gdn::sm90 {
namespace prepare_detail {

// Single already-resident tile, never reused. ONLY the preparation kernel
// below may use this adapter: it waits for all TMA inputs + a CTA metadata
// barrier before entering the existing arithmetic, and a CTA barrier after
// that arithmetic before publishing either complete operand. There is no
// overlapping producer/consumer to synchronize inside this single invocation.
struct ResidentTile {
    static constexpr int Stages = 1;
    using ProducerBarrierType = uint64_t;
    template<class S> CUTE_DEVICE void consumer_wait(S const&) {}
    template<class S> CUTE_DEVICE void consumer_release(S const&) {}
    template<class S> CUTE_DEVICE void producer_acquire(S const&) {}
    template<class S> CUTE_DEVICE void producer_commit(S const&) {}
};

template<class Base>
struct AuxBase : Base {
    using Element = typename Base::Element;
    using QKSmemLayoutQ = SingleStage<typename Base::QKSmemLayoutQ>;
    using QKSmemLayoutK = SingleStage<typename Base::QKSmemLayoutK>;
    using KVSmemLayoutK = SingleStage<typename Base::KVSmemLayoutK>;
    using QKQSmemLayoutAlpha = SingleStage<typename Base::QKQSmemLayoutAlpha>;
    using SmemLayoutBeta = decltype(cute::make_layout(cute::Shape<cute::_64,cute::_1>{}));
    using SmemLayoutQK = SingleStage<typename Base::SmemLayoutQK>;
    using SmemLayoutKK = SingleStage<typename Base::SmemLayoutKK>;
    using StagesAlpha = cutlass::gemm::collective::StageCount<1>;
    using MainloopQPipeline = ResidentTile;
    using MainloopKPipeline = ResidentTile;
    using MainloopQKPipeline = ResidentTile;
    using MainloopKKPipeline = ResidentTile;
    using MainloopAlphaPipeline = ResidentTile;
    using MainloopBetaPipeline = ResidentTile;
    using MainloopAlphaLastPipeline = ResidentTile;
    using QPipelineState = cutlass::PipelineState<1>;
    using KPipelineState = cutlass::PipelineState<1>;
    using QKPipelineState = cutlass::PipelineState<1>;
    using KKPipelineState = cutlass::PipelineState<1>;
    using AlphaPipelineState = cutlass::PipelineState<1>;
    using BetaPipelineState = cutlass::PipelineState<1>;
    using AlphaLastPipelineState = cutlass::PipelineState<1>;
    using LoadQ = kda::sm90::collective::CollectiveLoadTma<
        kda::sm90::collective::LoadKind::kQ,ResidentTile,Element,QKSmemLayoutQ,typename Base::TMA_Q>;
    using LoadK = kda::sm90::collective::CollectiveLoadTma<
        kda::sm90::collective::LoadKind::kK,ResidentTile,Element,KVSmemLayoutK,typename Base::TMA_K>;

    struct SharedStorage {
        alignas(128) cute::array_aligned<Element,cute::cosize_v<QKSmemLayoutQ>> smem_q;
        alignas(128) cute::array_aligned<Element,cute::cosize_v<KVSmemLayoutK>> smem_k;
        alignas(128) cute::array_aligned<Element,cute::cosize_v<SmemLayoutQK>> smem_qk;
        alignas(128) cute::array_aligned<typename Base::InverseType,cute::cosize_v<SmemLayoutKK>> smem_kk;
        cute::array_aligned<float,cute::cosize_v<QKQSmemLayoutAlpha>> smem_alpha;
        cute::array_aligned<float,cute::cosize_v<SmemLayoutBeta>> smem_beta;
    };
};

} // namespace prepare_detail

// One independent CTA per input chunk/value head, not per value slice. Gate
// and beta are Hv-specific even when several value heads share the same Q/K.
template<class Base, class Problem>
struct PrepareAuxKernel {
    using PrepBase = prepare_detail::AuxBase<Base>;
    using Arithmetic = ScalarGdnAux<PrepBase,true>;
    using Element = typename Base::Element;
    static constexpr int Threads = 128;
    struct SharedStorage {
        typename Arithmetic::SharedStorage tile;
        alignas(16) uint64_t inputs_ready;
    };
    struct Params {
        typename Base::Params mainloop;
        Problem problem;
        Element* prepared;
        int heads_per_group;
        int chunk_count;
    };
    CUTE_DEVICE void operator()(Params const& p, char* shared) {
        using namespace cute;
        auto& s = *reinterpret_cast<SharedStorage*>(shared);
        int tid = int(threadIdx.x);
        int head = int(blockIdx.x) % p.problem.num_v_heads;
        int batch = int(blockIdx.x) / p.problem.num_v_heads;
        int chunk = int(blockIdx.y);
        int valid = min(64,p.problem.sequence_length-chunk*64);
        kda::sm90::kernel::WorkDesc work{
            batch,head/p.heads_per_group,head,
            int64_t(batch)*p.problem.sequence_length+chunk*64,valid};
        if (tid == 0) initialize_barrier(s.inputs_ready,1);
        __syncthreads();

        prepare_detail::ResidentTile resident;
        typename PrepBase::LoadQ qload(p.mainloop.tma_load_q,resident,s.tile.smem_q);
        typename PrepBase::LoadK kload(p.mainloop.tma_load_k,resident,s.tile.smem_k);
        auto qs = qload.partition_SD(p.problem,typename Base::TileShape{},work);
        auto ks = kload.partition_SD(p.problem,typename Base::TileShape{},work);
        if (tid == 0) {
            set_barrier_transaction_bytes(s.inputs_ready,2*64*128*sizeof(Element));
            copy(p.mainloop.tma_load_q.with(s.inputs_ready),get<0>(qs)(_,_,_,0),get<1>(qs)(_,_,_,0));
            copy(p.mainloop.tma_load_k.with(s.inputs_ready),get<0>(ks)(_,_,_,0),get<1>(ks)(_,_,_,0));
        }
        if (tid < 32) {
            cutlass::PipelineState<1> aw;
            auto alpha = make_tensor(make_smem_ptr(s.tile.smem_alpha.data()),typename PrepBase::QKQSmemLayoutAlpha{});
            load_scalar_gate<64,128>(p.mainloop.gate_ptr,p.problem.num_v_heads,
                                     work,0,resident,aw,alpha);
        }
        if (tid < 64) {
            s.tile.smem_beta[tid] = tid < valid ? float(p.mainloop.beta_ptr[
                (work.tok_offset+tid)*p.problem.num_v_heads+head]) : 0.f;
        }
        wait_barrier(s.inputs_ready,0);
        __syncthreads(); // all inputs resident; the adapter never permits reuse

        cutlass::PipelineState<1> qr,kr,qw,kw,ar,br,lr;
        Arithmetic{}.compute_aux_safe(p.mainloop,p.problem,work,
            resident,qr,resident,kr,resident,qw,resident,kw,
            resident,ar,resident,br,resident,lr,s.tile);
        __syncthreads(); // every STSM/inverse owner has finished before vector stores

        static_assert(cosize(typename PrepBase::SmemLayoutQK{}) == PreparedAuxLayout::PlaneElements);
        static_assert(cosize(typename PrepBase::SmemLayoutKK{}) == PreparedAuxLayout::PlaneElements);
        auto* dst = reinterpret_cast<uint4*>(p.prepared + PreparedAuxLayout::element_offset(
            batch,head,chunk,p.problem.num_v_heads,p.chunk_count));
        auto const* qk = reinterpret_cast<uint4 const*>(s.tile.smem_qk.data());
        auto const* kk = reinterpret_cast<uint4 const*>(s.tile.smem_kk.data());
        CUTE_UNROLL
        for (int i=tid;i<PreparedAuxLayout::PlaneVectors;i+=Threads) {
            dst[i] = qk[i];
            dst[PreparedAuxLayout::PlaneVectors+i] = kk[i];
        }
    }
};

template<class Kernel>
__global__ __launch_bounds__(128,2) void prepare_aux_device(
    CUTLASS_GRID_CONSTANT typename Kernel::Params const params) {
    extern __shared__ __align__(128) char shared[];
    Kernel{}(params,shared);
}

} // namespace gdn::sm90
