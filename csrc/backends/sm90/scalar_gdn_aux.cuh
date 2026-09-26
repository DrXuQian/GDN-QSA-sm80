// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kda/sm90/collective/mainloop_kda_fwd.hpp"
#include "fragment_convert.cuh"

namespace gdn::sm90 {

// Scalar GDN specialization. A gate independent of K factors OUT of the dot
// product. Vector-gated KDA cannot use this identity and retains its original
// collective. TMA/layout/state/inverse contracts are inherited unchanged.
//
// Floating point schedule: BF16 Q/K -> FP32 dot -> scalar decay -> existing
// BF16 QK / FP16 KK storage. This is NOT bit-equivalent to gated TF32 operands.
template<class Base, bool AuxInverse = false>
struct ScalarGdnAux : Base {
    using Element = typename Base::Element;
    using Inverse = typename Base::InverseType;
    using Params = typename Base::Params;
    using SharedStorage = typename Base::SharedStorage;
    using QPipeline = typename Base::MainloopQPipeline;
    using KPipeline = typename Base::MainloopKPipeline;
    using QKPipeline = typename Base::MainloopQKPipeline;
    using KKPipeline = typename Base::MainloopKKPipeline;
    using AlphaPipeline = typename Base::MainloopAlphaPipeline;
    using BetaPipeline = typename Base::MainloopBetaPipeline;
    using AlphaLastPipeline = typename Base::MainloopAlphaLastPipeline;
    using QState = typename Base::QPipelineState;
    using KState = typename Base::KPipelineState;
    using QKState = typename Base::QKPipelineState;
    using KKState = typename Base::KKPipelineState;
    using AlphaState = typename Base::AlphaPipelineState;
    using BetaState = typename Base::BetaPipelineState;
    using AlphaLastState = typename Base::AlphaLastPipelineState;

    template<class Problem, class Work>
    CUTE_DEVICE void compute_aux_safe(
        Params const& params, Problem const&, Work const& work,
        QPipeline& qp, QState& qr, KPipeline& kp, KState& kr,
        QKPipeline& qkp, QKState& qw, KKPipeline& kkp, KKState& kw,
        AlphaPipeline& ap, AlphaState& ar, BetaPipeline& bp, BetaState& br,
        AlphaLastPipeline&, AlphaLastState&, SharedStorage& smem) {
        using namespace cute;
        static_assert(Base::BlkSeqKV == 64 && Base::HeadSize == 128);
        static_assert(Base::NumAuxMmaThreads == 128);
        // The inherited scalar gate image must be a K broadcast, not KDA data.
        static_assert(cosize(typename Base::QKQSmemLayoutAlpha{}) ==
                      Base::BlkSeqKV * Base::StagesAlpha::value);

        int tid = threadIdx.x % 128;
        auto q = make_tensor(make_smem_ptr(smem.smem_q.data()), typename Base::QKSmemLayoutQ{});
        auto k = make_tensor(make_smem_ptr(smem.smem_k.data()), typename Base::QKSmemLayoutK{});
        auto alpha = make_tensor(make_smem_ptr(smem.smem_alpha.data()), typename Base::QKQSmemLayoutAlpha{});
        auto beta = make_tensor(make_smem_ptr(smem.smem_beta.data()), typename Base::SmemLayoutBeta{});
        auto qk = make_tensor(make_smem_ptr(smem.smem_qk.data()), typename Base::SmemLayoutQK{});
        auto kk = make_tensor(make_smem_ptr(smem.smem_kk.data()), typename Base::SmemLayoutKK{});

        auto mma = typename Base::TiledMmaQK{};
        auto thread = mma.get_thread_slice(tid);
        auto qa = thread.make_fragment_A(thread.partition_A(q));
        auto ka = thread.make_fragment_A(thread.partition_A(k));
        auto kb = thread.make_fragment_B(thread.partition_B(k));
        auto coords = thread.partition_C(make_identity_tensor(Shape<_64,_64>{}));
        auto store_qk = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, Element>{}, mma);
        auto store_kk = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, Inverse>{}, mma);
        auto tq = store_qk.get_thread_slice(tid);
        auto tk = store_kk.get_thread_slice(tid);

        int chunks = ceil_div(work.seq_len, 64);
        CUTE_NO_UNROLL
        for (int chunk = 0; chunk < chunks; ++chunk) {
            int valid = min(int(work.seq_len - chunk * 64), 64);
            auto acc_qk = partition_fragment_C(mma, Shape<_64,_64>{});
            auto acc_kk = partition_fragment_C(mma, Shape<_64,_64>{});

            kp.consumer_wait(kr);
            warpgroup_fence_operand(acc_kk);
            warpgroup_arrive();
            kda::sm90::collective::gemm_zero_acc(mma, ka(_,_,_,kr.index()), kb(_,_,_,kr.index()), acc_kk);
            warpgroup_commit_batch();
            qp.consumer_wait(qr);
            warpgroup_fence_operand(acc_qk);
            warpgroup_arrive();
            kda::sm90::collective::gemm_zero_acc(mma, qa(_,_,_,qr.index()), kb(_,_,_,kr.index()), acc_qk);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(acc_qk);
            warpgroup_fence_operand(acc_kk);
            // Inputs can be overwritten only after both WGMMA groups retire.
            kp.consumer_release(kr); ++kr;
            qp.consumer_release(qr); ++qr;

            ap.consumer_wait(ar);
            bp.consumer_wait(br);
            CUTE_UNROLL
            for (int i = 0; i < size(coords); ++i) {
                auto [row, col] = coords(i);
                bool live = row >= col && row < valid && col < valid;
                // Mask BEFORE exp/cast: future/tail entries must never create
                // infinities which could leak through a later zero multiply.
                float decay = live ? exp2f(alpha(row,0,ar.index()) - alpha(col,0,ar.index())) : 0.f;
                acc_qk(i) = live ? acc_qk(i) * decay * params.scale : 0.f;
            }
            auto out_qk = convert_fragment<Element>(acc_qk);
            CUTE_UNROLL
            for (int i = 0; i < size(coords); ++i) {
                auto [row,col] = coords(i);
                bool live = row >= col && row < valid && col < valid;
                float decay = live ? exp2f(alpha(row,0,ar.index()) - alpha(col,0,ar.index())) : 0.f;
                // Inverse expects positive lower input, garbage diagonal and
                // zero upper triangle, then applies beta along its columns.
                acc_kk(i) = live ? acc_kk(i) * beta(row,br.index()) * decay : 0.f;
            }
            auto out_kk = convert_fragment<Inverse>(acc_kk);
            kkp.producer_acquire(kw);
            qkp.producer_acquire(qw);
            copy(store_qk, tq.retile_S(out_qk), tq.partition_D(qk(_,_,qw.index())));
            copy(store_kk, tk.retile_S(out_kk), tk.partition_D(kk(_,_,kw.index())));
            if constexpr (AuxInverse) {
                // KK remains private to the producer until BOTH inversion and
                // its beta-column conversion finish. State must not repeat it.
                using Barriers = kda::sm90::collective::KdaNamedBarriers;
                cutlass::arch::NamedBarrier::arrive_and_wait(128,Barriers::AuxMath);
                typename Base::CollectiveInverse solve(Barriers::AuxMath);
                solve.compute(kk(_,_,kw.index()));
                cutlass::arch::NamedBarrier::arrive_and_wait(128,Barriers::AuxMath);
                auto ld = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N,Inverse>{},mma);
                auto l = ld.get_thread_slice(tid);
                auto inv = make_fragment_like<Inverse>(acc_kk);
                auto operand = make_fragment_like<Element>(acc_kk);
                copy(ld,l.partition_S(kk(_,_,kw.index())),l.retile_D(inv));
                CUTE_UNROLL
                for (int i=0; i<size(inv); ++i) {
                    auto [row,col] = coords(i);
                    operand(i) = Element(float(inv(i))*beta(col,br.index()));
                }
                auto kk_bf16 = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem.smem_kk.data())),
                                          typename Base::SmemLayoutKK{});
                copy(store_qk,tq.retile_S(operand),tq.partition_D(kk_bf16(_,_,kw.index())));
            }
            cutlass::arch::fence_view_async_shared();
            qkp.producer_commit(qw); ++qw;
            kkp.producer_commit(kw); ++kw;
            ap.consumer_release(ar); ++ar;
            bp.consumer_release(br); ++br;
        }
    }
};

} // namespace gdn::sm90
