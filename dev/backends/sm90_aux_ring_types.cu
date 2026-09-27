#include "kernel_types.cuh"
#include <cassert>
#include <cstdio>
#include <set>
#include <type_traits>

template<class Gate, bool Initial>
void inspect() {
    using T = gdn::sm90::KernelTypes<Gate, Initial>;
    using C = typename T::Collective;
    using R = gdn::sm90::AuxRings;
    using namespace cute;
    using QK = typename C::SmemLayoutQK;
#if defined(GDN_TEST_BAD_KK_AS_QK)
    using KK = typename C::SmemLayoutQK;
#else
    using KK = typename C::SmemLayoutKK;
#endif
    static_assert(C::StagesQK::value == R::QK && C::StagesKK::value == R::KK);
    static_assert(C::MainloopQKPipeline::Stages == R::QK);
    static_assert(C::MainloopKKPipeline::Stages == R::KK);
    static_assert(size<2>(typename C::SmemLayoutQK{}) == R::QK);
    static_assert(size<2>(KK{}) == R::KK);
    static_assert(cosize(typename C::SmemLayoutQK{}) == 4096*R::QK);
    static_assert(cosize(KK{}) == 4096*R::KK);
    if constexpr(GDN_SM90_AUX_RING_CONFIG == 0)
        static_assert(std::is_same_v<typename T::Options, typename R::template OriginalOptions<Gate, Initial>>);
    std::set<int> qk_offsets, kk_offsets;
    for(int stage=0;stage<R::QK;++stage)
        for(int r=0;r<64;++r) for(int c=0;c<64;++c) {
            int offset = QK{}(make_coord(r,c,stage));
            assert(offset>=stage*4096 && offset<(stage+1)*4096);
            assert(qk_offsets.insert(offset).second);
        }
    for(int stage=0;stage<R::KK;++stage)
        for(int r=0;r<64;++r) for(int c=0;c<64;++c) {
            int offset = KK{}(make_coord(r,c,stage));
            assert(offset>=stage*4096 && offset<(stage+1)*4096);
            assert(kk_offsets.insert(offset).second);
        }
    // Storage continues to be FP16 inverse overwritten by BF16 conditioned
    // operands at identical byte offsets, independently of stage count.
    static_assert(sizeof(typename C::InverseType) == sizeof(typename C::Element));
    printf("{\"config\":%d,\"gate_fp32\":%d,\"initial\":%d,\"qk\":%d,\"kk\":%d,"
           "\"qk_cells\":%zu,\"kk_cells\":%zu,\"shared_bytes\":%d,\"threads\":%d}\n",
           GDN_SM90_AUX_RING_CONFIG,int(std::is_same_v<Gate,float>),int(Initial),R::QK,R::KK,
           qk_offsets.size(),kk_offsets.size(),T::Kernel::SharedStorageSize,T::Kernel::MaxThreadsPerBlock);
}

int main() {
    inspect<cutlass::bfloat16_t,false>(); inspect<cutlass::bfloat16_t,true>();
    inspect<float,false>(); inspect<float,true>();
}
