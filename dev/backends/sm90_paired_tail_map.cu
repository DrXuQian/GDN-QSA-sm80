#include "value_types.cuh"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace cute;
using namespace kda::sm90::kernel;
using BF16=cutlass::bfloat16_t;
using Types=gdn::sm90::ValueKernelTypes<BF16,false,64,232>;
using Collective=typename Types::Collective;
static_assert(Collective::NumStateMmaWarpGroups==1 && Types::Kernel::MaxThreadsPerBlock==384);
static_assert(Types::Kernel::StateMmaRegisterRequirement==192 && Types::Kernel::AuxMmaRegisterRequirement==232);
void need(bool p) { if(!p) throw std::runtime_error("O2/KV delta ownership"); }
void map(bool wrong,bool missing) {
    auto o2=typename Collective::TiledMmaO2{};
    auto kv=typename Collective::TiledMmaKV{};
    static_assert(std::is_same_v<typename Collective::TiledMmaO2::LayoutA_TV,
                                 typename Collective::TiledMmaKV::LayoutA_TV>);
    std::array<int,4096> owners{};int checked=0;
    for(int tid=0;tid<128-(missing?1:0);++tid) {
        auto a=o2.get_slice(tid).partition_A(make_identity_tensor(Shape<_64,_64>{}));
        auto b=kv.get_slice(tid).partition_A(make_identity_tensor(Shape<_64,_64>{}));
        need(size(a)==size(b));
        for(int i=0;i<size(a);++i) {
            auto [v,t]=a(i);auto [w,u]=b(wrong?(i+1)%size(b):i);
            need(v==w && t==u);need(v>=0 && v<64 && t>=0 && t<64);
            ++owners[int(v)*64+int(t)];++checked;
        }
    }
    need(checked==4096);for(int x:owners)need(x==1);
}
template<class F> void red(F f) { try{f();}catch(std::runtime_error const&){return;}
    throw std::runtime_error("negative escaped"); }
int main(){map(false,false);red([]{map(true,false);});red([]{map(false,true);});
    std::cout<<"[S52 actual delta operands] 4096/4096 perV64CTA exact-once, O2/KV identical coordinates;2negativesPASS\n";}
