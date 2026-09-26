// Bind the actual kernel arrival count, then exhaust bounded ring schedules.
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "scalar_gdn_state.cuh"
#include <array>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <vector>
using namespace cute;
using namespace kda::sm90::kernel;
using GdnStride = tuple<int64_t,_1,int32_t>;

template<class Gate, bool Initial> struct Actual {
  using Options = std::tuple<Option<Tag::kElementGateGmem,Gate>,
      Option<Tag::kElementBetaGmem,cutlass::bfloat16_t>,
      Option<Tag::kInitStateFromInput,bool_constant<Initial>>>;
  using Builder = FlatBuilderKdaFwd<cutlass::bfloat16_t,float,float,Shape<_64,_64,_128>,
      GdnStride,GdnStride,GdnStride,GdnStride,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative,Options>;
  using Scalar = FlatKernelTmaWarpSpecializedKdaFwd<
      gdn::sm90::ScalarGdnState<typename Builder::CollectiveMainloop,true>,
      typename Builder::TileScheduler,Options>;
  using Vector = typename Builder::Kernel;
  static_assert(!Scalar::UsesAlphaLastPipeline);
  static_assert(Scalar::AlphaConsumerThreads == 384);
  static_assert(Vector::UsesAlphaLastPipeline);
  static_assert(Vector::AlphaConsumerThreads == 416);
  static_assert(Builder::CollectiveMainloop::StagesAlpha::value == 2);
};

void require(bool ok) { if (!ok) throw std::runtime_error("alpha ring contract"); }

size_t exhaust(int chunks, int consumer_threads) {
  using State = std::array<int,4>; // producer publications, completed aux/state0/state1
  std::set<State> visited;
  std::vector<State> pending{{0,0,0,0}};
  while (!pending.empty()) {
    auto now=pending.back(); pending.pop_back();
    if (!visited.insert(now).second) continue;
    bool terminal=true;
    for (int i:now) terminal=terminal && i==chunks;
    if (terminal) continue;
    bool enabled=false;
    int p=now[0], arrivals=0;
    for (int c=1;c<=3;++c) arrivals+=128*(now[c]>p-2);
    if (p<chunks && (p<2 || arrivals>=consumer_threads)) {
      auto next=now; ++next[0]; pending.push_back(next); enabled=true;
    }
    for (int c=1;c<=3;++c) if (now[c]<p) {
      // A consumer must never observe the next generation in its held slot.
      require(p-now[c]<=2);
      auto next=now; ++next[c]; pending.push_back(next); enabled=true;
    }
    require(enabled); // nonterminal state with no actor able to progress
  }
  return visited.size();
}

int main() {
  using A=Actual<cutlass::bfloat16_t,false>;
  static_assert(sizeof(Actual<cutlass::bfloat16_t,true>) &&
                sizeof(Actual<float,false>) && sizeof(Actual<float,true>));
  size_t states=0;
  for (int chunks=1;chunks<=64;++chunks)
    states+=exhaust(chunks,A::Scalar::AlphaConsumerThreads);
  for (int count:{416,256}) {
    bool red=false;
    try { exhaust(8,count); } catch(std::runtime_error const&) { red=true; }
    require(red);
  }
  printf("[scalar alpha] actual4specializations scalar384 vector416 stages2 "
         "chunks1..64 reachable_states=%zu PASS; stale416/omittedconsumer EXPECTED_RED\n",states);
}
