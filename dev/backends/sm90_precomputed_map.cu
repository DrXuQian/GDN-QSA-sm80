#include "precomputed_types.cuh"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace cute;
using BF16 = cutlass::bfloat16_t;
using Types = gdn::sm90::PrecomputedKernelTypes<BF16,false,128>;
using Prepare = typename Types::Prepare;
using Prep = typename Prepare::PrepBase;
using State = typename Types::Collective;
using Storage = gdn::sm90::PreparedAuxLayout;
using Atom = Copy_Atom<SM90_U32x4_STSM_N,BF16>;
void need(bool ok) { if (!ok) throw std::runtime_error("prepared operand map/coverage mismatch"); }

// Execute actual STSM cross-lane delivery; source tags are independent logical
// (row,col), destination/consumer offsets come from the instantiated CuTe types.
template<class ProducerLayout,class ConsumerLayout,class Consumer>
void plane_map(Consumer consumer, int plant=0) {
    static_assert(cosize_v<ProducerLayout> == 4096 && cosize_v<ConsumerLayout> == 4096);
    alignas(1024) std::array<uint16_t,4096> physical_addresses{};
    for (int i=0;i<4096;++i) physical_addresses[i]=uint16_t(i);
    auto producer = make_tensor(make_smem_ptr(physical_addresses.data()),ProducerLayout{})(_,_,_0{});
    auto reader = make_tensor(make_smem_ptr(physical_addresses.data()),ConsumerLayout{})(_,_,_0{});
    auto mma = typename Prep::TiledMmaQK{};
    auto store = make_tiled_copy_C(Atom{},mma);
    std::array<std::array<int,32>,128> sources{},destinations{};
    for (int tid=0;tid<128;++tid) {
        auto th=mma.get_slice(tid);
        auto coords=th.partition_C(make_identity_tensor(Shape<_64,_64>{}));
        auto tags=make_fragment_like<int>(partition_fragment_C(th,Shape<_64,_64>{}));
        need(size(tags)==32);
        for (int i=0;i<32;++i) { auto [r,c]=coords(i);tags(i)=int(r)*64+int(c); }
        auto st=store.get_slice(tid);
        auto src=st.retile_S(tags);
        auto dst=st.partition_D(producer);
        need(size(src)==32 && size(dst)==32);
        for(int i=0;i<32;++i) {sources[tid][i]=src(i);destinations[tid][i]=dst(i);}
    }
    auto src_layout=typename Atom::ValLayoutSrc{};
    auto dst_layout=typename Atom::ValLayoutDst{};
    constexpr int ns=Atom::NumValSrc,nd=Atom::NumValDst;
    static_assert(ns==8 && nd==8);
    std::array<int,4096> prepared{},count{},reloaded{};
    for(int tid=0;tid<(plant==2?127:128);++tid) for(int i=0;i<32;++i) {
        int bit=src_layout(tid%32,i%ns),receiver=-1,component=-1;
        for(int lane=0;lane<32;++lane) for(int v=0;v<nd;++v)
            if(dst_layout(lane,v)==bit) {need(receiver<0);receiver=lane;component=v;}
        need(receiver>=0);
        int address=destinations[(tid/32)*32+receiver][(i/ns)*nd+component];
        need(address>=0 && address<4096);
        prepared[address]=sources[tid][i];++count[address];
    }
    // Actual producer128-thread/vector8 and consumer32-thread/vector8 loops.
    // Count global writers and consumers separately: no hidden missing word.
    std::array<int,4096> global_writers{},state_writers{};
    for(int tid=0;tid<128;++tid) for(int i=tid;i<Storage::PlaneVectors;i+=128)
        for(int h=0;h<8;++h) ++global_writers[i*8+h];
    for(int tid=0;tid<32;++tid) for(int i=tid;i<Storage::PlaneVectors;i+=32)
        for(int h=0;h<8;++h) {
            int word=i*8+h;
            reloaded[word]=prepared[plant==1 ? word^1 : word];++state_writers[word];
        }
    for(int i=0;i<4096;++i) need(count[i]==1 && global_writers[i]==1 && state_writers[i]==1);
    for(int tid=0;tid<size(consumer);++tid) {
        auto th=consumer.get_slice(tid);
        auto pos=th.partition_B(reader);
        auto coords=th.partition_B(make_identity_tensor(Shape<_64,_64>{}));
        for(int i=0;i<size(pos);++i) {
            auto [r,c]=coords(i);need(reloaded[pos(i)]==int(r)*64+int(c));
        }
    }
}

template<class Gate,bool Initial> void actual_type() {
    using T=gdn::sm90::PrecomputedKernelTypes<Gate,Initial,128>;
    using K=typename T::Kernel;
    static_assert(K::MaxThreadsPerBlock==384 && K::StateMmaRegisterRequirement==192);
    static_assert(K::NumAuxMmaWarpGroups==0 && K::LdStRegisterRequirement==24);
    static_assert(K::QKInputConsumers==256 && K::AlphaConsumers==288 && K::BetaConsumers==256);
    static_assert(T::Collective::MainloopQKPipeline::Stages==1 && T::Collective::MainloopKKPipeline::Stages==1);
    std::cout<<"S48 gate_fp32="<<std::is_same_v<Gate,float><<" initial="<<Initial
             <<" state_threads="<<K::MaxThreadsPerBlock<<" state_shared="<<K::SharedStorageSize
             <<" prepare_shared="<<sizeof(typename T::Prepare::SharedStorage)<<"\n";
}

int main(int argc,char** argv) {
    actual_type<BF16,false>();actual_type<BF16,true>();actual_type<float,false>();actual_type<float,true>();
    auto verify=[](int plant) {
        plane_map<typename Prep::SmemLayoutQK,typename State::SmemLayoutQK>(typename State::TiledMmaO2{},plant);
        plane_map<typename Prep::SmemLayoutKK,typename State::SmemLayoutKK>(typename State::TiledMmaNewV{},plant);
    };
    verify(0);
    for(int plant=1;plant<=2;++plant) {
        bool red=false;try{verify(plant);}catch(std::runtime_error const&){red=true;}
        need(red);
    }
    if(argc==5) {
        int batches=std::stoi(argv[1]),length=std::stoi(argv[2]),hq=std::stoi(argv[3]),hv=std::stoi(argv[4]);
        need(batches>0 && length>0 && hq>0 && hv%hq==0);
        int chunks=length/64+(length%64!=0);
        int64_t tiles=0,last_end=0;
        for(int b=0;b<batches;++b) for(int h=0;h<hv;++h) for(int c=0;c<chunks;++c) {
            int64_t offset=Storage::element_offset(b,h,c,hv,chunks);
            need(offset==last_end && offset%64==0);
            for(int plane=0;plane<2;++plane)
                need(Storage::element_offset(b,h,c,hv,chunks,plane)==offset+plane*4096);
            last_end=offset+8192;++tiles;
        }
        need(last_end==Storage::elements(batches,hv,length) && tiles==int64_t(batches)*hv*chunks);
        std::cout<<"scratch_coverage="<<tiles<<"/"<<tiles<<" chunks exact-once elements="<<last_end<<"\n";
    } else need(argc==1);
    std::cout<<"prepared: actual STSM -> vector image -> cp.async -> O2/NewV;8192/8192 exact-once;2 negatives PASS\n";
}
