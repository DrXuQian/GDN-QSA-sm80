#include "scalar_gdn_state.cuh"
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace cute;
using namespace kda::sm90::kernel;
using BF16=cutlass::bfloat16_t;
using GdnStride=tuple<int64_t,_1,int32_t>;
template<class Gate,bool Initial> struct Types {
    using Options=std::tuple<Option<Tag::kElementGateGmem,Gate>,
        Option<Tag::kElementBetaGmem,BF16>,Option<Tag::kInitStateFromInput,bool_constant<Initial>>>;
    using Builder=FlatBuilderKdaFwd<BF16,float,float,Shape<_64,_64,_128>,GdnStride,GdnStride,GdnStride,GdnStride,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative,Options>;
    using Collective=gdn::sm90::ScalarGdnState<typename Builder::CollectiveMainloop,true>;
    using Kernel=FlatKernelTmaWarpSpecializedKdaFwd<Collective,typename Builder::TileScheduler,Options>;
};
void need(bool ok) {if(!ok) throw std::runtime_error("pair-decay map/bank/denominator");}
template<class Gate,bool Initial> void type() {
    using T=Types<Gate,Initial>;using K=typename T::Kernel;
    static_assert(K::MaxThreadsPerBlock==512 && K::SharedStorageSize+1024<=232448);
    static_assert(K::LdStRegisterRequirement==32 && K::AuxMmaRegisterRequirement==96 && K::StateMmaRegisterRequirement==192);
    static_assert(T::Collective::StagesAlpha::value==2);
    static_assert(sizeof(std::declval<typename T::Collective::SharedStorage>().pair_decay)==32768);
    std::cout<<"S72 gate_float="<<std::is_same_v<Gate,float><<" initial="<<Initial<<" shared="<<K::SharedStorageSize<<"\n";
}
void check(int plant=0) {
    std::array<int,8192> memory{},writes{},reads{};
    for(int stage=0;stage<2;++stage) for(int col=0;col<32;++col)
        for(int half_row=0;half_row<2;++half_row) for(int half_col=0;half_col<2;++half_col) {
            std::array<int,32> banks{};
            for(int lane=0;lane<(plant==3?31:32);++lane) {
                int row=lane+32*half_row, c=col+32*half_col;
                int pos=gdn::sm90::pair_decay_index(stage,row,c);
                if(plant==1) pos^=1;
                if(plant==2) pos%=4096;
                need(pos>=0 && pos<8192);
                memory[pos]=stage*4096+row*64+c; ++writes[pos]; ++banks[pos%32];
            }
            for(int n:banks) need(n==1);
        }
    using C=typename Types<BF16,false>::Collective;
    auto mma=typename C::TiledMmaQK{};
    for(int stage=0;stage<2;++stage) for(int warp=0;warp<4;++warp)
        for(int value=0;value<32;++value) {
            std::array<int,32> banks{};
            for(int lane=0;lane<32;++lane) {
                auto thread=mma.get_thread_slice(warp*32+lane);
                auto coords=thread.partition_C(make_identity_tensor(Shape<_64,_64>{}));
                static_assert(size(coords)==32);
                auto [r,c]=coords(value);
                int pos=gdn::sm90::pair_decay_index(plant==4?0:stage,int(r),int(c));
                need(memory[pos]==stage*4096+int(r)*64+int(c));
                ++reads[pos]; ++banks[pos%32];
            }
            for(int n:banks) need(n==1);
        }
    for(int i=0;i<8192;++i) need(writes[i]==1 && reads[i]==1);
}
int main() {
    type<BF16,false>();type<BF16,true>();type<float,false>();type<float,true>();
    check();
    for(int plant=1;plant<=4;++plant) {
        bool red=false;try{check(plant);}catch(std::runtime_error const&){red=true;}
        need(red);
    }
    std::cout<<"S72 actual GMMA-C consumers:8192/8192 exact-once; producer+consumer32x4B scalar-bank model no alias;4negativesPASS (not hardware timing)\n";
}
