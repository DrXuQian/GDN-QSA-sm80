#include "value_types.cuh"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace cute;
using BF16=cutlass::bfloat16_t;
using Types=gdn::sm90::ValueKernelTypes<BF16,false,128,168,true,2,160>;
using Collective=Types::Collective;
using H=gdn::sm90::SharedStateLayout<Collective>;
using Atom=Copy_Atom<SM90_U32x4_STSM_N,BF16>;
void need(bool ok) { if(!ok) throw std::runtime_error("shared H map/protocol mismatch"); }

// Execute the real store atom's cross-lane map on tags, not an invented
// per-thread identity copy. Real partition/retile offsets come from CuTe.
void map(int plant=0) {
    static_assert(cosize_v<typename H::Layout> == 16384);
    auto mma=typename Collective::TiledMmaKV{};
    auto store=typename H::Store{};
    // Pointer-swizzled layouts require a 16-bit shared iterator. Encode each
    // physical halfword address in actual aligned storage; no BF16 rounding.
    alignas(1024) std::array<uint16_t,16384> address_tags{};
    for(int i=0;i<16384;++i) address_tags[i]=uint16_t(i);
    auto physical=make_tensor(make_smem_ptr(address_tags.data()),typename H::Layout{})(_,_,_0{});
    std::array<std::array<int,64>,256> values{},destinations{};
    for(int tid=0;tid<256;++tid) {
        auto th=mma.get_slice(tid);
        auto coords=th.partition_C(make_identity_tensor(Shape<_128,_128>{}));
        auto tags=make_fragment_like<int>(partition_fragment_C(th,Shape<_128,_128>{}));
        need(size(tags)==64);
        for(int i=0;i<size(tags);++i) {
            auto [v,k]=coords(i); tags(i)=int(v)*128+int(k);
        }
        auto sh=store.get_slice(tid);
        auto src=sh.retile_S(tags);
        auto dst=sh.partition_D(physical);
        need(size(src)==64 && size(dst)==64);
        for(int i=0;i<64;++i) { values[tid][i]=src(i);destinations[tid][i]=dst(i); }
    }
    constexpr int ns=Atom::NumValSrc,nd=Atom::NumValDst;
    static_assert(ns==8 && nd==8);
    auto src_layout=typename Atom::ValLayoutSrc{};
    auto dst_layout=typename Atom::ValLayoutDst{};
    std::array<int,16384> written{},count{};
    for(int tid=0;tid<(plant==3?255:256);++tid) for(int i=0;i<64;++i) {
        int bit=src_layout(tid%32,i%ns),receiver=-1,component=-1;
        for(int lane=0;lane<32;++lane) for(int val=0;val<nd;++val)
            if(dst_layout(lane,val)==bit) { need(receiver<0);receiver=lane;component=val; }
        need(receiver>=0);
        int address=destinations[(tid/32)*32+receiver][(i/ns)*nd+component];
        if(plant==1) address=address%128*128+address/128;
        if(plant==2) address%=8192;
        need(address>=0 && address<16384);
        written[address]=values[tid][i];++count[address];
    }
    for(int v=0;v<128;++v) for(int k=0;k<128;++k) {
        int address=physical(v,k);
        need(count[address]==1 && written[address]==v*128+k);
    }
    // Both real SS readers consume that physical tile under the same A
    // layout. Validate their actual partition coordinates against the tags.
    static_assert(std::is_same_v<typename Collective::CollectiveMmaO1::SmemLayoutA,
                                 typename Collective::CollectiveMmaSK::SmemLayoutA>);
    auto reader=[&](auto consumer) {
        for(int tid=0;tid<256;++tid) {
            auto th=consumer.get_slice(tid);
            auto positions=th.partition_A(physical);
            auto coords=th.partition_A(make_identity_tensor(Shape<_128,_128>{}));
            for(int i=0;i<size(coords);++i) {
                auto [v,k]=coords(i);need(written[positions(i)]==int(v)*128+int(k));
            }
        }
    };
    reader(typename H::O1Mma{});reader(typename H::SKMma{});
}
template<class Gate,bool Initial> void actual_type() {
    using T=gdn::sm90::ValueKernelTypes<Gate,Initial,128,168,true,2,160>;
    using K=typename T::Kernel;
    static_assert(T::Collective::NumStateMmaWarpGroups==2 && K::MaxThreadsPerBlock==512);
    static_assert(K::LdStRegisterRequirement==24 && K::StateMmaRegisterRequirement==160 && K::AuxMmaRegisterRequirement==168);
    static_assert(K::SharedStorageSize<=232448);
    static_assert(K::QKInputConsumers==384 && K::AlphaConsumers==416 && K::BetaConsumers==384);
    std::cout<<"S43 gate_fp32="<<std::is_same_v<Gate,float><<" initial="<<Initial
             <<" threads="<<K::MaxThreadsPerBlock<<" shared="<<K::SharedStorageSize<<"\n";
}
int main() {
    actual_type<BF16,false>();actual_type<BF16,true>();actual_type<float,false>();actual_type<float,true>();
    map();
    for(int plant=1;plant<=3;++plant) {
        bool red=false;try{map(plant);}catch(std::runtime_error const&){red=true;}
        need(red);
    }
    std::cout<<"shared H: actual STSM lane map -> SS O1/SK, 16384/16384 EXACT-ONCE; transpose/alias/missing negatives PASS\n";
}
