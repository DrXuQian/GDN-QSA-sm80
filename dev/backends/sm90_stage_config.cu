#include "kernel_types.cuh"
#include <cstdio>
#include <type_traits>

template<class Gate, bool Initial>
void inspect() {
    using Types = gdn::sm90::KernelTypes<Gate, Initial>;
    using C = typename Types::Collective;
    using S = gdn::sm90::SelectedStages;
    static_assert(C::StagesQ::value == S::Q && C::StagesK::value == S::K);
    static_assert(C::StagesV::value == S::V && C::StagesO::value == S::O);
    static_assert(C::StagesAlpha::value == S::Alpha && C::StagesBeta::value == S::Beta);
    static_assert(C::NumStateMmaWarpGroups == 2 && C::NumAuxMmaWarpGroups == 1);
    if constexpr(GDN_SM90_STAGE_CONFIG == 0)
        static_assert(std::is_same_v<typename Types::Options, typename S::template OriginalOptions<Gate, Initial>>);
    printf("{\"config\":%d,\"gate_fp32\":%d,\"initial\":%d,\"stages\":[%d,%d,%d,%d,%d,%d],\"shared_bytes\":%d}\n",
        GDN_SM90_STAGE_CONFIG, int(std::is_same_v<Gate,float>), int(Initial), C::StagesQ::value,
        C::StagesK::value,C::StagesV::value,C::StagesO::value,C::StagesAlpha::value,C::StagesBeta::value,
        Types::Kernel::SharedStorageSize);
}
int main() {
    inspect<cutlass::bfloat16_t, false>(); inspect<cutlass::bfloat16_t, true>();
    inspect<float, false>(); inspect<float, true>();
}
