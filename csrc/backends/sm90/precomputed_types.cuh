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

template<class Gate, bool Initial, int ValueTile = 64>
struct PrecomputedKernelTypes {
    using Original = ValueKernelTypes<Gate,Initial,ValueTile,232>;
    using InputBase = typename ValueKernelTypes<Gate,false,64,232>::Builder::CollectiveMainloop;
    using Base = typename Original::Builder::CollectiveMainloop;
    using Collective = PrecomputedState<Base>;
    using Options = decltype(std::tuple_cat(typename Original::Options{},std::tuple<
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kPrecomputedAuxiliary,cute::true_type>>{}));
    using Scheduler = typename Original::Scheduler;
    using Kernel = kda::sm90::kernel::FlatKernelTmaWarpSpecializedKdaFwd<Collective,Scheduler,Options>;
    using Prepare = PrepareAuxKernel<InputBase,PreparedProblem>;
    static_assert(ValueTile == 64 || ValueTile == 128);
    static_assert(Kernel::MaxThreadsPerBlock == 128+2*ValueTile);
    static_assert(Kernel::StateThreads == 2*ValueTile && Kernel::AuxThreads == 0);
    static_assert(Kernel::QKInputConsumers == 2*ValueTile && Kernel::AlphaConsumers == 2*ValueTile+32 &&
                  Kernel::BetaConsumers == 2*ValueTile);
    static_assert(Kernel::SharedStorageSize <= (ValueTile==64 ? 116224 : 232448),
                  "prepared state exceeds registered H800 resource scope");
};

} // namespace gdn::sm90
