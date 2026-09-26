// CPU proof using the same visitor and actual auxiliary fragment coordinates.
#include <cute/tensor.hpp>
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "aux_chunk_loop.cuh"
#include <array>
#include <climits>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace cute;
using S=tuple<int64_t,_1,int32_t>;
using B=kda::sm90::kernel::FlatBuilderKdaFwd<cutlass::bfloat16_t,float,float,
    Shape<_64,_64,_128>,S,S,S,S,cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
void require(bool ok) { if (!ok) throw std::runtime_error("aux chunk coverage"); }
bool schedule(int length, bool omit_tail) {
    std::vector<int> counts(length,0);
    int calls=0,full=0,dynamic=0;
    gdn::sm90::for_each_aux_chunk(length,[&](int chunk, auto valid){
        constexpr bool stat = cute::is_static<decltype(valid)>::value;
        if (omit_tail && !stat) return;
        ++calls; full += stat; dynamic += !stat;
        require(int(valid)>0 && int(valid)<=64);
        for(int i=0;i<int(valid);++i) {
            int token=chunk*64+i;
            require(token>=0 && token<length);
            ++counts[token];
        }
    });
    if(calls!=(length+63)/64 || full!=calls-1 || dynamic!=1) return false;
    for(int n:counts) if(n!=1) return false;
    return true;
}
int main() {
    static_assert(gdn::sm90::aux_chunk_count(INT_MAX)==33554432);
    static_assert(gdn::sm90::aux_chunk_count(1)==1);
    static_assert(gdn::sm90::aux_chunk_count(64)==1);
    static_assert(gdn::sm90::aux_chunk_count(65)==2);
    std::array<int,4096> owners{};
    auto mma=typename B::CollectiveMainloop::TiledMmaQK{};
    auto id=make_identity_tensor(Shape<_64,_64>{});
    for(int tid=0;tid<128;++tid) {
        auto coords=mma.get_slice(tid).partition_C(id);
        for(int i=0;i<size(coords);++i) {
            auto [r,c]=coords(i);
            require(r>=0 && r<64 && c>=0 && c<64);
            ++owners[int(r)*64+int(c)];
        }
    }
    for(int n:owners) require(n==1);
    for(int length=1;length<=4096;++length) {
        require(schedule(length,false));
        require(!schedule(length,true));
    }
    std::cout << "[aux schedule] lengths=4096 exact-once; actual-fragment=4096/4096; omitted-tail EXPECTED_RED/PASS\n";
}
