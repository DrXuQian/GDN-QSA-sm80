// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "value_types.cuh"
#include "precomputed_state.cuh"
#include "prepare_aux_kernel.cuh"

namespace gdn::sm90 {

// Unlike Kernel::ProblemShape, this type does not depend on Initial. Thus
// there are exactly two prepared-kernel images, one for each gate dtype.
struct PreparedProblem {
    int64_t total_seqlen;
    int num_seqs, num_qk_heads, num_v_heads, head_size, sequence_length;
};

template<class Gate, bool Initial>
struct PrecomputedKernelTypes {
    using Original = ValueKernelTypes<Gate,Initial,64,232>;
    using InputBase = typename ValueKernelTypes<Gate,false,64,232>::Builder::CollectiveMainloop;
    using Base = typename Original::Builder::CollectiveMainloop;
    using Collective = PrecomputedState<Base>;
    using Options = decltype(std::tuple_cat(typename Original::Options{},std::tuple<
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kPrecomputedAuxiliary,cute::true_type>>{}));
    using Kernel = kda::sm90::kernel::FlatKernelTmaWarpSpecializedKdaFwd<
        Collective,ValueTileScheduler<64>,Options>;
    using Prepare = PrepareAuxKernel<InputBase,PreparedProblem>;
    static_assert(Kernel::MaxThreadsPerBlock == 256);
    static_assert(Kernel::StateThreads == 128 && Kernel::AuxThreads == 0);
    static_assert(Kernel::QKInputConsumers == 128 && Kernel::AlphaConsumers == 160 &&
                  Kernel::BetaConsumers == 128);
    static_assert(Kernel::SharedStorageSize <= 116224,
                  "prepared state must fit half of H800 opt-in shared capacity");
};

} // namespace gdn::sm90
