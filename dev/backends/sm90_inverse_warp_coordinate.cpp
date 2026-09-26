#include "kerutils/device/sm80/inverse_warp_coordinate.hpp"
#include <array>
#include <cstdio>
#include <stdexcept>

int main() {
    std::array<int,4> owners{};
    int count=0, planted=0;
    for(unsigned thread=0;thread<1024;++thread) {
        // Independent thread -> warp -> warpgroup quotient/remainder anchor.
        int warp=thread/32, group=thread/128;
        int want=warp-group*4;
        int actual=kerutils::inverse_warp_in_group(thread);
        if(actual!=want || actual<0 || actual>=4)
            throw std::runtime_error("actual inverse coordinate helper mismatch");
        ++owners[actual]; ++count;
        planted+=int((thread>>4)&3u)!=want;
    }
    if(count!=1024 || owners!=std::array<int,4>{256,256,256,256} || planted==0)
        throw std::runtime_error("coverage/wrong-bit negative escaped");
    std::printf("inverse warp coordinate actual-helper PASS threads=%d owners=256/256/256/256 wrong-bit EXPECTED_RED bad=%d\n",count,planted);
}
