// Actual four shipping KK layouts; independent logical diagonal/read owners.
#include <cute/tensor.hpp>
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "scalar_gdn_state.cuh"
#include "inverse_diagonal_owner.cuh"
#include <iostream>
#include <array>
#include <stdexcept>
#include <vector>
using namespace cute;
using namespace kda::sm90::kernel;
void need(bool x){if(!x)throw std::runtime_error("first inverse diagonal ownership");}

template<class Gate,bool Initial> void check(int plant=0){
    using E=cutlass::bfloat16_t;using S=tuple<int64_t,_1,int32_t>;
    using Opt=std::tuple<Option<Tag::kElementGateGmem,Gate>,
        Option<Tag::kElementBetaGmem,E>,Option<Tag::kInitStateFromInput,bool_constant<Initial>>>;
    using B=FlatBuilderKdaFwd<E,float,float,Shape<_64,_64,_128>,S,S,S,S,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative,Opt>;
    using L=typename B::CollectiveMainloop::SmemLayoutKK;
    // Keep the actual B16 smem pointer flag and pointer swizzle. A flat int
    // counting iterator is not this physical layout and is correctly rejected.
    alignas(128) std::array<uint16_t,cosize(L{})> storage{};
    for(int i=0;i<int(storage.size());++i)storage[i]=uint16_t(i);
    auto image=make_tensor(make_smem_ptr(storage.data()),L{});
    need(size<0>(image)==64 && size<1>(image)==64 && size<2>(image)==2);
    for(int stage=0;stage<2;++stage){
        auto matrix=image(_,_,stage);
        auto diagonals=flat_divide(matrix,Shape<_8,_8>{});
        std::vector<int> owners(cosize(L{}),-1),visits(cosize(L{}),0);
        int written=0,read=0;
        for(int tid=0;tid<128;++tid){
            auto owner=gdn::sm90::inverse_diagonal_owner(tid);
            if(plant==1)owner={tid/8,tid%8,tid<64}; // old producer + new warp fence
            if(plant==2)owner.tile=(owner.tile+1)%8;
            if(plant==3)owner.publishes=true;
            if(plant==4 && tid/32==3)owner.publishes=false;
            if(!owner.publishes)continue;
            need(owner.tile>=0 && owner.tile<8 && owner.row>=0 && owner.row<8);
            auto tile=diagonals(_,_,owner.tile,owner.tile);
            for(int col=0;col<8;++col){
                int address=tile(owner.row,col);
                need(address>=0 && address<int(owners.size()));
                ++visits[address];owners[address]=tid/32;++written;
            }
        }
        auto merges=flat_divide(matrix,Shape<_16,_16>{});
        for(int warp=0;warp<4;++warp){
            auto merge=merges(_,_,warp,warp);
            for(int r=0;r<16;++r)for(int c=0;c<16;++c)if(r/8==c/8){
                int address=merge(r,c);
                need(address==image(warp*16+r,warp*16+c,stage));
                need(visits[address]==1 && owners[address]==warp);++read;
            }
        }
        need(written==512 && read==512);
        for(int r=0;r<64;++r)for(int c=0;c<64;++c)
            need(visits[image(r,c,stage)]==int(r/8==c/8));
    }
}
int main(){
    check<cutlass::bfloat16_t,false>();check<cutlass::bfloat16_t,true>();
    check<float,false>();check<float,true>();
    for(int p=1;p<=4;++p){bool red=false;
        try{check<cutlass::bfloat16_t,false>(p);}catch(std::runtime_error const&){red=true;}
        need(red);
    }
    std::cout<<"[S61 actual map] types4 x stages2 x512=4096 diagonal stores/read owners exact-once/warp-local; old-owner/wrong-tile/duplicate/omitted-warp negatives PASS\n";
}
