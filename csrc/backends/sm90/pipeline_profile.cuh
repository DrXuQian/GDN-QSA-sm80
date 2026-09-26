#pragma once
#include <cute/tensor.hpp>
#include "kda/sm90/kernel/options.hpp"

namespace gdn::sm90 {

// Compile-time configuration authority; geometry, maths and role allocation
// are not implied sweep axes. Generic vector-KDA defaults remain unchanged.
template<int Q, int K, int V, int O, int Alpha, int Beta>
struct PipelineProfile {
    static_assert(Q > 0 && K > 0 && V > 0 && O > 0 && Alpha > 0 && Beta > 0);
    template<class Gate, bool Initial>
    using Options = std::tuple<
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kElementGateGmem, Gate>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kElementBetaGmem, cutlass::bfloat16_t>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kInitStateFromInput, cute::bool_constant<Initial>>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kStagesQ, cute::Int<Q>>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kStagesK, cute::Int<K>>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kStagesV, cute::Int<V>>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kStagesO, cute::Int<O>>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kStagesAlpha, cute::Int<Alpha>>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kStagesBeta, cute::Int<Beta>>>;
};

using FlashInferPipelineProfile = PipelineProfile<2, 3, 2, 2, 5, 5>;

} // namespace gdn::sm90
