// Four actual KK layouts, physical B16 pointers, independently checked owners.
#include <cute/tensor.hpp>
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "scalar_gdn_state.cuh"
#include "inverse_prefix_owner.cuh"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace cute;
using namespace kda::sm90::kernel;
void need(bool p){if(!p)throw std::runtime_error("inverse subtree ownership");}

template<class Gate,bool Initial>void check(int plant=0){
 using E=cutlass::bfloat16_t;using S=tuple<int64_t,_1,int32_t>;
 using Opt=std::tuple<Option<Tag::kElementGateGmem,Gate>,Option<Tag::kElementBetaGmem,E>,
                      Option<Tag::kInitStateFromInput,bool_constant<Initial>>>;
 using B=FlatBuilderKdaFwd<E,float,float,Shape<_64,_64,_128>,S,S,S,S,
                         cutlass::gemm::KernelTmaWarpSpecializedCooperative,Opt>;
 using L=typename B::CollectiveMainloop::SmemLayoutKK;
 alignas(128)std::array<uint16_t,cosize(L{})> storage{};
 for(int i=0;i<int(storage.size());++i)storage[i]=uint16_t(i);
 auto image=make_tensor(make_smem_ptr(storage.data()),L{});
 for(int stage=0;stage<2;++stage){
  auto mat=image(_,_,stage);
  auto tiles8=flat_divide(mat,Shape<_8,_8>{});
  auto tiles16=flat_divide(mat,Shape<_16,_16>{});
  auto tiles32=flat_divide(mat,Shape<_32,_32>{});
  std::vector<int> owner(cosize(L{}),-1),visits(cosize(L{}),0);
  for(int tid=0;tid<64;++tid)for(int col=0;col<8;++col){
   int at=tiles8(tid%8,col,tid/8,tid/8);
   owner[at]=tid/32;++visits[at];
  }
  for(int warp=0;warp<(plant==1?4:2);++warp){
   for(int half=0;half<(plant==1?1:2);++half){
    if(plant==3 && half==1)continue;
    int tile=plant==1?warp:gdn::sm90::inverse_prefix_tile16(warp,half);
    if(plant==2)tile=(tile+1)%4;
    auto block=tiles16(_,_,tile,tile);
    for(int r=0;r<16;++r)for(int c=0;c<16;++c)if(r/8==c/8){
     int at=block(r,c);need(owner[at]==warp && visits[at]==1);
    }
    for(int r=8;r<16;++r)for(int c=0;c<8;++c){
     int at=block(r,c);need(visits[at]==0);owner[at]=warp;visits[at]=plant==4?2:1;
    }
   }
  }
  int produced=0;
  for(int warp=0;warp<2;++warp){
   auto block=tiles32(_,_,warp,warp);
   for(int r=0;r<32;++r)for(int c=0;c<32;++c){
    if(r/16==c/16 && (r%16)/8 >= (c%16)/8){
     int at=block(r,c);need(at==image(warp*32+r,warp*32+c,stage));
     need(owner[at]==warp && visits[at]==1);++produced;
    }
   }
  }
  need(produced==768);
 }
}
int main(){
 check<cutlass::bfloat16_t,false>();check<cutlass::bfloat16_t,true>();
 check<float,false>();check<float,true>();
 for(int p=1;p<=4;++p){bool red=false;try{check<cutlass::bfloat16_t,false>(p);}catch(std::runtime_error const&){red=true;}need(red);}
 std::cout<<"[S66 actual map] types4 x stages2 x768=6144 updated subtree cells exact-once/warp-local; four negative controls PASS\n";
}
