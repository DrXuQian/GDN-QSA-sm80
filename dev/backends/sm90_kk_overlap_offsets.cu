// Actual shipping four-type storage authority for the native mbarrier checker.
#include <cute/tensor.hpp>
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "scalar_gdn_state.cuh"
#include <iostream>
using namespace cute;
using namespace kda::sm90::kernel;
template<class Gate, bool Initial> void check() {
    using E=cutlass::bfloat16_t;
    using S=tuple<int64_t,_1,int32_t>;
    using Options=std::tuple<Option<Tag::kElementGateGmem,Gate>,
        Option<Tag::kElementBetaGmem,E>,Option<Tag::kInitStateFromInput,bool_constant<Initial>>>;
    using B=FlatBuilderKdaFwd<E,float,float,Shape<_64,_64,_128>,S,S,S,S,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative,Options>;
    using M=gdn::sm90::ScalarGdnState<typename B::CollectiveMainloop,true>;
    using K=FlatKernelTmaWarpSpecializedKdaFwd<M,typename B::TileScheduler,Options>;
    using Storage=typename K::SharedStorage;
    using QK=typename Storage::QKPipelineStorage;
    using KK=typename Storage::KKPipelineStorage;
    static_assert(offsetof(Storage,qk_pipeline_storage)+offsetof(QK,full_barrier_)==0x29060);
    static_assert(offsetof(Storage,qk_pipeline_storage)+offsetof(QK,empty_barrier_)==0x29070);
    static_assert(offsetof(Storage,kk_pipeline_storage)+offsetof(KK,full_barrier_)==0x29080);
    using Main=typename M::SharedStorage;
    static_assert(offsetof(Main,smem_qk)+sizeof(Main::smem_qk)<=offsetof(Main,smem_kk));
}
int main() {
    check<cutlass::bfloat16_t,false>(); check<cutlass::bfloat16_t,true>();
    check<float,false>(); check<float,true>();
    std::cout<<"[S57 native offsets] actual-types=4 QK-full/empty=0x29060/70 KK-full=0x29080 disjoint PASS\n";
}
