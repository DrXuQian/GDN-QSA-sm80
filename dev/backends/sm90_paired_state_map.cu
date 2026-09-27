#include "scalar_gdn_state.cuh"
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace cute;
using namespace kda::sm90::kernel;
using BF16 = cutlass::bfloat16_t;
using GdnStride = cute::tuple<int64_t, _1, int32_t>;
using Builder = FlatBuilderKdaFwd<BF16, float, float, Shape<_64,_64,_128>,
    GdnStride, GdnStride, GdnStride, GdnStride,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    std::tuple<Option<Tag::kElementGateGmem,BF16>,Option<Tag::kElementBetaGmem,BF16>>>;
using Collective = gdn::sm90::ScalarGdnState<typename Builder::CollectiveMainloop,true>;
// Instantiate the device-only production conversion, never launch the probe.
__global__ void paired_h_conversion_compile_probe() {
    auto kv=typename Collective::TiledMmaKV{};
    auto h=partition_fragment_C(kv.get_slice(int(threadIdx.x)),Shape<_128,_128>{});
    using H1=decltype(kda::sm90::collective::make_acc_into_op<BF16>(h,typename Collective::TiledMmaO1::LayoutA_TV{}));
    using H2=decltype(kda::sm90::collective::make_acc_into_op<BF16>(h,typename Collective::TiledMmaSK::LayoutA_TV{}));
    static_assert(std::is_same_v<H1,H2>);
}
void need(bool p) { if(!p) throw std::runtime_error("O1/SK H operand map"); }
void map(bool wrong_value, bool missing_thread) {
    auto o1=typename Collective::TiledMmaO1{};
    auto sk=typename Collective::TiledMmaSK{};
    static_assert(std::is_same_v<typename Collective::TiledMmaO1::LayoutA_TV,
                                 typename Collective::TiledMmaSK::LayoutA_TV>);
    std::array<int,16384> owners{};
    int checked=0;
    for(int tid=0;tid<256-(missing_thread?1:0);++tid) {
        auto one=o1.get_slice(tid).partition_A(make_identity_tensor(Shape<_128,_128>{}));
        auto two=sk.get_slice(tid).partition_A(make_identity_tensor(Shape<_128,_128>{}));
        need(size(one)==size(two));
        for(int i=0;i<size(one);++i) {
            auto [v,k]=one(i);
            auto [w,l]=two((wrong_value && tid==0)?(i+1)%size(two):i);
            need(v==w && k==l);
            need(v>=0 && v<128 && k>=0 && k<128);
            ++owners[int(v)*128+int(k)];++checked;
        }
    }
    need(checked==16384);
    for(int n:owners) need(n==1);
}
template<class F> void red(F f) {
    try { f(); } catch(std::runtime_error const&) { return; }
    throw std::runtime_error("paired H negative escaped");
}
int main() {
    map(false,false);red([]{map(true,false);});red([]{map(false,true);});
    std::cout<<"[S49 actual H operands] identical production retile types and 16384/16384 coordinates; two negatives PASS\n";
}
