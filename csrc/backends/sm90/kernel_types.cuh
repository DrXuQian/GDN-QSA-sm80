#pragma once
#include "stage_config.cuh"
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "scalar_gdn_state.cuh"

namespace gdn::sm90 {
template<class Gate, bool Initial>
struct KernelTypes {
    using Options = typename SelectedStages::template Options<Gate, Initial>;
    using Stride = cute::tuple<int64_t, cute::_1, int32_t>;
    using Builder = kda::sm90::kernel::FlatBuilderKdaFwd<cutlass::bfloat16_t, float, float,
        cute::Shape<cute::_64, cute::_64, cute::_128>, Stride, Stride, Stride, Stride,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative, Options>;
    using Collective = ScalarGdnState<typename Builder::CollectiveMainloop, true>;
    using Kernel = kda::sm90::kernel::FlatKernelTmaWarpSpecializedKdaFwd<
        Collective, typename Builder::TileScheduler, Options>;
};
} // namespace gdn::sm90
