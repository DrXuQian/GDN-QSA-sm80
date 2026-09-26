// CPU execution of the actual shipping CuTe types, not a parallel index model.
#include <cute/tensor.hpp>
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "scalar_gdn_state.cuh"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace cute;
using namespace kda::sm90::kernel;
using BF16 = cutlass::bfloat16_t;
using S = cute::tuple<int64_t, _1, int32_t>;
void require(bool v) { if (!v) throw std::runtime_error("state WG ownership"); }

template<class Gate, bool Initial>
void check(bool stale_writer) {
    using Options = std::tuple<Option<Tag::kElementGateGmem, Gate>,
        Option<Tag::kElementBetaGmem, BF16>,
        Option<Tag::kInitStateFromInput, cute::bool_constant<Initial>>>;
    using Builder = FlatBuilderKdaFwd<BF16,float,float,Shape<_64,_64,_128>,
        S,S,S,S,cutlass::gemm::KernelTmaWarpSpecializedCooperative,Options>;
    using M = gdn::sm90::ScalarGdnState<typename Builder::CollectiveMainloop,true>;
    using Old = gdn::sm90::ScalarGdnState<typename Builder::CollectiveMainloop,false>;
    static_assert(std::is_same_v<typename M::OrderedMathBarriers,gdn::sm90::IndependentStateIssue>);
    static_assert(!std::is_same_v<typename Old::OrderedMathBarriers,gdn::sm90::IndependentStateIssue>);
    auto state_id = make_identity_tensor(Shape<_128,_128>{});
    auto output_id = make_identity_tensor(Shape<_128,_64>{});
    std::vector<int> state(128*128), output(128*64);
    typename M::SmemLayoutO olayout;
    typename M::SharedStorage storage{};
    auto physical = make_tensor(make_smem_ptr(storage.smem_o.data()), olayout);
    std::vector<int> physical_owners(cosize(olayout));
    auto mma = typename M::TiledMmaO1{};
    auto copy_o = make_tiled_copy_C(typename M::CollectiveStoreO::CopyAtomR2S{}, mma);
    for (int tid=0; tid<256; ++tid) {
        int mapped_tid = stale_writer ? tid%128 : tid;
        int wg = tid/128;
        auto st = typename M::TiledMmaKV{}.get_slice(mapped_tid).partition_C(state_id);
        for (int i=0;i<size(st);++i) {
            auto [v,k] = st(i);
            require(int(v)/64 == wg);
            require(++state[int(k)*128+int(v)] == 1);
        }
        auto ot = mma.get_slice(mapped_tid).partition_C(output_id);
        for (int i=0;i<size(ot);++i) {
            auto [v,t] = ot(i);
            require(int(v)/64 == wg);
            require(++output[int(t)*128+int(v)] == 1);
        }
        auto store = copy_o.get_slice(mapped_tid);
        for (int stage=0;stage<size<2>(olayout);++stage) {
            auto dst = store.partition_D(physical(_,_,stage));
            require(size(dst) == 32);
            for (int i=0;i<size(dst);++i) {
                int offset = &dst(i) - storage.smem_o.data();
                require(offset>=0 && offset<int(physical_owners.size()));
                require(++physical_owners[offset] == 1);
            }
        }
    }
    for (int n:state) require(n==1);
    for (int n:output) require(n==1);
    int physical_count=0;
    for (int n:physical_owners) physical_count += n;
    require(physical_count == 8192*size<2>(olayout));
    for (int t=0;t<64;++t) for(int v=0;v<128;++v)
    for(int stage=0;stage<size<2>(olayout);++stage)
        require(physical_owners[&physical(v,t,stage)-storage.smem_o.data()]==1);
    std::cout << "[state WG owners] gate_bytes=" << sizeof(Gate)
              << " initial=" << Initial << " state=16384 output=8192 physical="
              << physical_count << " WG0=V0..63 WG1=V64..127 DISJOINT/PASS\n";
}
int main() {
    check<BF16,false>(false); check<BF16,true>(false);
    check<float,false>(false); check<float,true>(false);
    try { check<BF16,false>(true); }
    catch (std::runtime_error const&) {
        std::cout << "[state WG owners] stale WG0 writer EXPECTED_RED/PASS\n";
        return 0;
    }
    return 1;
}
