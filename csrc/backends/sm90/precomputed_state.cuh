// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "prepared_aux_layout.cuh"
#include "scalar_gdn_state.cuh"
#include <cute/arch/copy_sm80.hpp>

namespace gdn::sm90 {

// Storage/participant specialization only. Matrix atoms and ScalarGdnState
// arithmetic stay inherited. In particular there is no legacy KDA scaled-Q
// scratch in this type, so accidentally selecting its compute is ill-formed.
template<class Base>
struct PreparedStateBase : Base {
    using Element = typename Base::Element;
    static_assert(Base::ValueTile == 64);
    static constexpr int NumAuxMmaWarpGroups = 0;
    static constexpr int NumAuxMmaThreads = 0;
    using SmemLayoutQK = SingleStage<typename Base::SmemLayoutQK>;
    using SmemLayoutKK = SingleStage<typename Base::SmemLayoutKK>;
    using MainloopQKPipeline = cutlass::PipelineAsync<1>;
    using MainloopKKPipeline = cutlass::PipelineAsync<1>;
    using QKPipelineState = cutlass::PipelineState<1>;
    using KKPipelineState = cutlass::PipelineState<1>;
    struct Arguments : Base::Arguments { Element* prepared = nullptr; };
    struct Params : Base::Params { Element const* prepared = nullptr; };
    template<class Problem>
    static Params to_underlying_arguments(Problem const& p, Arguments const& a, void* workspace) {
        Params out;
        static_cast<typename Base::Params&>(out) = Base::to_underlying_arguments(p,a,workspace);
        out.prepared = a.prepared;
        return out;
    }
    struct SharedStorage {
        alignas(128) cute::array_aligned<Element,cute::cosize_v<typename Base::QKSmemLayoutQ>> smem_q;
        alignas(128) cute::array_aligned<Element,cute::cosize_v<typename Base::KVSmemLayoutK>> smem_k;
        alignas(128) cute::array_aligned<Element,cute::cosize_v<typename Base::KVSmemLayoutV>> smem_v;
        cute::array_aligned<float,cute::cosize_v<typename Base::QKQSmemLayoutAlpha>> smem_alpha;
        alignas(128) cute::array_aligned<Element,cute::cosize_v<SmemLayoutQK>> smem_qk;
        alignas(128) cute::array_aligned<typename Base::InverseType,cute::cosize_v<SmemLayoutKK>> smem_kk;
        typename Base::SharedStorageO smem_o;
        cute::array_aligned<float,cute::cosize_v<typename Base::SmemLayoutBeta>> smem_beta;
        cute::array_aligned<float,cute::cosize_v<typename Base::SmemLayoutAlphaLast>> smem_alpha_last;
    };
};

template<class OriginalBase>
struct PrecomputedState : ScalarGdnState<PreparedStateBase<OriginalBase>,true> {
    using Base = PreparedStateBase<OriginalBase>;
    using Parent = ScalarGdnState<Base,true>;
    using Params = typename Base::Params;
    using SharedStorage = typename Parent::SharedStorage;

    template<class Problem, class Work>
    CUTE_DEVICE void load_prepared_aux(
        Params const& p, Problem const& problem, Work const& work,
        typename Base::MainloopQKPipeline& qkp, typename Base::QKPipelineState& qw,
        typename Base::MainloopKKPipeline& kkp, typename Base::KKPipelineState& kw,
        typename Base::MainloopBetaPipeline& bp, typename Base::BetaPipelineState& bw,
        SharedStorage& s) {
        using namespace cute;
        int lane = int(threadIdx.x)&31;
        int count = PreparedAuxLayout::chunks(int(work.seq_len));
        auto const* src = reinterpret_cast<uint128_t const*>(p.prepared+
            PreparedAuxLayout::element_offset(work.seq_idx,work.o_head_idx(),0,
                                              problem.num_v_heads,count));
        auto* qk = reinterpret_cast<uint128_t*>(s.smem_qk.data());
        auto* kk = reinterpret_cast<uint128_t*>(s.smem_kk.data());
        CUTE_NO_UNROLL
        for (int chunk=0;chunk<count;++chunk) {
            qkp.producer_acquire(qw);
            kkp.producer_acquire(kw);
            CUTE_UNROLL
            for (int i=lane;i<PreparedAuxLayout::PlaneVectors;i+=32) {
                SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>::copy(src[i],qk[i]);
                SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>::copy(src[PreparedAuxLayout::PlaneVectors+i],kk[i]);
            }
            cp_async_fence();
            cp_async_wait<0>();
            __syncwarp();
            cutlass::arch::fence_view_async_shared();
            qkp.producer_commit(qw); ++qw;
            kkp.producer_commit(kw); ++kw;
            // Preserve the state's existing beta wait/release contract. Beta
            // is mathematically consumed by prepare, but this publication
            // retains the unmodified state protocol for the first experiment.
            bp.producer_acquire(bw);
            CUTE_UNROLL
            for (int i=lane;i<64;i+=32) {
                int row=chunk*64+i;
                s.smem_beta[bw.index()*64+i] = row<work.seq_len ? float(p.beta_ptr[
                    (work.tok_offset+row)*problem.num_v_heads+work.o_head_idx()]) : 0.f;
            }
            cutlass::arch::fence_view_async_shared();
            bp.producer_commit(bw); ++bw;
            src += 2*PreparedAuxLayout::PlaneVectors;
        }
    }
};

} // namespace gdn::sm90
