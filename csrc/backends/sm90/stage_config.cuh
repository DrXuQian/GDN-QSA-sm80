#pragma once
#include <cute/tensor.hpp>
#include "kda/sm90/kernel/options.hpp"

#ifndef GDN_SM90_STAGE_CONFIG
#define GDN_SM90_STAGE_CONFIG 0
#endif

namespace gdn::sm90 {

// One bounded search, not a promise that arbitrary positive stages are legal.
// Bit order is K,V,O,alpha,beta; Q and all geometry/register roles are fixed.
template<int Id>
struct StageConfig {
    static_assert(Id >= 0 && Id < 32, "stage config must belong to the 32-cell space");
    static constexpr int Q = 2, K = 2 + (Id & 1), V = 1 + ((Id >> 1) & 1);
    static constexpr int O = 1 + ((Id >> 2) & 1);
    static constexpr int Alpha = 2 + 3 * ((Id >> 3) & 1);
    static constexpr int Beta = 2 + 3 * ((Id >> 4) & 1);

    template<class Gate, bool Initial>
    using OriginalOptions = std::tuple<
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kElementGateGmem, Gate>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kElementBetaGmem, cutlass::bfloat16_t>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kInitStateFromInput, cute::bool_constant<Initial>>>;

    template<class Gate, bool Initial>
    using Options = std::conditional_t<Id == 0, OriginalOptions<Gate, Initial>, std::tuple<
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kElementGateGmem, Gate>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kElementBetaGmem, cutlass::bfloat16_t>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kInitStateFromInput, cute::bool_constant<Initial>>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kStagesQ, cute::Int<Q>>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kStagesK, cute::Int<K>>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kStagesV, cute::Int<V>>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kStagesO, cute::Int<O>>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kStagesAlpha, cute::Int<Alpha>>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kStagesBeta, cute::Int<Beta>>>>;
};

using SelectedStages = StageConfig<GDN_SM90_STAGE_CONFIG>;
} // namespace gdn::sm90
