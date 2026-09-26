#include "state_chunk_loop.cuh"
#include <array>
#include <cstdio>
#include <stdexcept>
void require(bool v) { if (!v) throw std::runtime_error("state chunk contract"); }

template<bool Initial> long check(int plant=0) {
  long visited=0;
  for (int tokens=1;tokens<=4096;++tokens) {
    int count=0,rows=0;
    std::array<int,64> owners{};
    gdn::sm90::for_each_state_chunk<Initial>(tokens,[&](int chunk,auto first,auto boundary){
      if (plant==1 && chunk==0) return;
      int valid=boundary ? gdn::sm90::state_chunk_valid(tokens,chunk) : 64;
      if (plant==2 && chunk==0) valid=tokens;
      require(chunk>=0 && chunk<int(owners.size()));
      require(first==(!Initial && chunk==0));
      require(boundary==(chunk==0 || (chunk+1)*64>=tokens));
      require(valid>0 && valid<=64);
      require(valid==((tokens-chunk*64)<64 ? tokens-chunk*64 : 64));
      ++owners[chunk]; ++count; ++visited; rows+=valid;
    });
    require(count==(tokens+63)/64 && rows==tokens);
    for (int c=0;c<64;++c) require(owners[c]==(c*64<tokens));
  }
  return visited;
}

int main() {
  long count=check<false>()+check<true>();
  for (int plant:{1,2}) {
    bool red=false;
    try {check<false>(plant);} catch(std::runtime_error const&) {red=true;}
    require(red);
  }
  printf("[actual state loop] T1..4096 initial0/1 chunk_visits=%ld exact-once/PASS; "
         "missing-first/unclamped-first EXPECTED_RED\n",count);
}
