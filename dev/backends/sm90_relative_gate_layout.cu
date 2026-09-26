#include <cute/tensor.hpp>
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "relative_gate_layout.cuh"
#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
using namespace cute;
using S=tuple<int64_t,_1,int32_t>;
using B=kda::sm90::kernel::FlatBuilderKdaFwd<cutlass::bfloat16_t,float,float,
    Shape<_64,_64,_128>,S,S,S,S,cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
void require(bool b) { if(!b) throw std::runtime_error("relative gate coverage"); }
int main() {
    std::array<int,8192> owners{};
    auto mma=typename B::CollectiveMainloop::TiledMmaKV{};
    int distinct_thread_tokens=0;
    for(int tid=0;tid<256;++tid) {
        auto coordinates=mma.get_slice(tid).partition_A(make_identity_tensor(Shape<_128,_64>{}));
        std::set<int> tokens;
        for(int i=0;i<size(coordinates);++i) {
            auto [v,t]=coordinates(i);
            require(v>=0 && v<128 && t>=0 && t<64);
            ++owners[int(v)*64+int(t)]; tokens.insert(int(t));
        }
        distinct_thread_tokens+=int(tokens.size());
    }
    for(int count:owners) require(count==1);
    require(distinct_thread_tokens==4096);
    for(int valid=1;valid<=64;++valid) {
        int last=gdn::sm90::relative_gate_last_lane(valid)+
                 (gdn::sm90::relative_gate_last_is_hi(valid)?32:0);
        require(last==valid-1);
        require(((last+1)%64)!=valid-1); // wrong source lane negative
        std::array<int,128> writers{};
        for(int stage=0;stage<2;++stage) for(int lane=0;lane<32;++lane)
            for(int half=0;half<2;++half) {
                int token=lane+half*32;
                int index=gdn::sm90::relative_gate_index(stage,token);
                require(index>=0 && index<128); ++writers[index];
                require(index==stage*64+token);
            }
        for(int count:writers) require(count==1);
    }
    std::cout<<"relative-gate: actual8192/8192 owners;4096 per-thread token evaluations ->64 producer tokens;64 tails/2stages;wrong-last EXPECTED_RED/PASS\n";
}
