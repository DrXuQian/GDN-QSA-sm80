#include "pipeline_profile.cuh"
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "scalar_gdn_state.cuh"
#include <cstdio>
using namespace cute;
using namespace kda::sm90::kernel;
using GdnStride=tuple<int64_t,_1,int32_t>;

template<class Gate,bool Initial> void check() {
  using Options=typename gdn::sm90::FlashInferPipelineProfile::template Options<Gate,Initial>;
  using Builder=FlatBuilderKdaFwd<cutlass::bfloat16_t,float,float,Shape<_64,_64,_128>,
      GdnStride,GdnStride,GdnStride,GdnStride,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative,Options>;
  using Base=typename Builder::CollectiveMainloop;
  using Collective=gdn::sm90::ScalarGdnState<Base,true>;
  using Kernel=FlatKernelTmaWarpSpecializedKdaFwd<Collective,typename Builder::TileScheduler,Options>;
  static_assert(Base::StagesQ::value==2 && Base::StagesK::value==3 && Base::StagesV::value==2);
  static_assert(Base::StagesO::value==2 && Base::StagesAlpha::value==5 && Base::StagesBeta::value==5);
  static_assert(Kernel::AlphaConsumerThreads==384 && Kernel::BetaConsumerThreads==128);
  static_assert(!Collective::StateUsesBeta && !Collective::UsesAlphaLastPipeline);
  static_assert(Kernel::SharedStorageSize<227*1024,"profile exceeds H800 opt-in SMEM ceiling");
  printf("[actual pipeline] gate_fp32=%d initial=%d Q=%d K=%d V=%d O=%d alpha=%d beta=%d "
         "alpha_consumers=%d beta_consumers=%d shared_bytes=%d\n",
         int(is_same_v<Gate,float>),int(Initial),Base::StagesQ::value,Base::StagesK::value,
         Base::StagesV::value,Base::StagesO::value,Base::StagesAlpha::value,Base::StagesBeta::value,
         Kernel::AlphaConsumerThreads,Kernel::BetaConsumerThreads,Kernel::SharedStorageSize);
}

int main() {
  check<cutlass::bfloat16_t,false>(); check<cutlass::bfloat16_t,true>();
  check<float,false>(); check<float,true>();
  puts("[actual pipeline] PASS: four shipping specializations, not a parallel parameter model");
}
