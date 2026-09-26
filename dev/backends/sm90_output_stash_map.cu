#include "output_stash.cuh"
#include <array>
#include <cstdio>

int main() {
    std::array<int,8192> count{};
    int bad=0, alias_bad=0, wrong_component=0;
    for (int thread=0;thread<256;++thread) {
        for (int component=0;component<32;++component) {
            int address=gdn::sm90::output_stash_index(thread,component);
            // Independentinverse, includingwarpgroup ownership.
            int owner=(address/4)%256, member=(address/1024)*4+address%4;
            if (address<0 || address>=8192 || owner!=thread || member!=component) ++bad;
            else ++count[address];
            if (gdn::sm90::output_stash_index(thread&127,component)!=address) ++alias_bad;
            if (gdn::sm90::output_stash_index(thread,component^1)!=address) ++wrong_component;
            if (component%4==0 && address%4!=0) ++bad;
        }
    }
    for (int n:count) if(n!=1) ++bad;
    // Each128B subtransaction has one word per32-bank location.
    for(int warp=0;warp<8;++warp) for(int group=0;group<8;++group)
        for(int phase=0;phase<4;++phase) {
            std::array<int,32> banks{};
            for(int lane=phase*8;lane<phase*8+8;++lane) for(int v=0;v<4;++v)
                ++banks[gdn::sm90::output_stash_index(warp*32+lane,group*4+v)%32];
            for(int n:banks) if(n!=1) ++bad;
        }
    --count[gdn::sm90::output_stash_index(255,31)];
    int missing=0; for(int n:count) if(n!=1) ++missing;
    std::printf("[output-stash] slots=8192 owners256 fragment32 raw-shared-bytes=32768 "
                "bad=%d alias_negative=%d component_negative=%d missing_negative=%d\n",
                bad,alias_bad,wrong_component,missing);
    return bad || alias_bad!=4096 || wrong_component!=8192 || missing!=1;
}
