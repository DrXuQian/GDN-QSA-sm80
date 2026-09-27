#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "scalar_gdn_state.cuh"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace cute;
using namespace kda::sm90::kernel;
using BF16=cutlass::bfloat16_t;
using InputStride=tuple<int64_t,_1,int32_t>;
void need(bool x){if(!x)throw std::runtime_error("causal sector ownership");}
template<class Gate,bool Initial>
void check(int plant=0){
    using Options=std::tuple<Option<Tag::kElementGateGmem,Gate>,Option<Tag::kElementBetaGmem,BF16>,
        Option<Tag::kInitStateFromInput,bool_constant<Initial>>>;
    using B=FlatBuilderKdaFwd<BF16,float,float,Shape<_64,_64,_128>,InputStride,InputStride,InputStride,InputStride,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative,Options>;
    using C=gdn::sm90::ScalarGdnState<typename B::CollectiveMainloop,true>;
    using Mma=typename C::TiledMmaQK;
    for(int valid=1;valid<=64;++valid){
        std::array<int,4096> owners{};int evaluated=0,skipped=0;
        for(int tid=0;tid<128;++tid){
            auto coords=Mma{}.get_thread_slice(tid).partition_C(make_identity_tensor(Shape<_64,_64>{}));
            auto visit=[&](auto index,bool compute){
                int i=int(index);if(plant==2)i=(i+1)%size(coords);
                if(plant==3 && tid==127 && i==size(coords)-1)return;
                auto [row,col]=coords(i);need(row>=0&&row<64&&col>=0&&col<64);
                need(int(row)/16==tid/32);
                int expected_band=int(col)/16;
                constexpr int actual_band=gdn::sm90::aux_column_sector<Mma,decltype(index)::value>();
                need(expected_band==actual_band);
                need(++owners[int(row)*64+int(col)]==1);
                if(!compute){need(row<col);++skipped;}
                else ++evaluated;
                if(row>=col&&row<valid&&col<valid)need(compute);
            };
            gdn::sm90::for_aux_causal_sectors<Mma>(plant==1?(tid+32)%128:tid,
                [&](auto i){visit(i,true);},[&](auto i){visit(i,false);});
        }
        for(int n:owners)need(n==1);
        need(evaluated==2560&&skipped==1536);
    }
}
int main(){
    check<BF16,false>();check<BF16,true>();check<float,false>();check<float,true>();
    for(int p=1;p<=3;++p){bool failed=false;try{check<BF16,false>(p);}catch(std::runtime_error const&){failed=true;}need(failed);}
    std::cout<<"[S56 actual causal map] 1048576 cells exact-once; every live cell computed; 1536/4096 skipped per CTA;3negativesPASS\n";
}
