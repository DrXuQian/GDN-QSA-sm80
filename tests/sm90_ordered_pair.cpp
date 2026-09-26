// Exercise the actual ID selectors; protocol simulation is not a GPU race test.
#include "ordered_pair.cuh"
#include <array>
#include <cstdio>
#include <type_traits>

using Pair = gdn::sm90::OrderedPair<4,5>;
static_assert(std::is_empty_v<Pair>, "no per-thread indexed barrier storage");
static_assert(Pair::Participants == 256);
static_assert(Pair::wait_id(0)==4 && Pair::wait_id(1)==5);
static_assert(Pair::notify_id(0)==5 && Pair::notify_id(1)==4);

bool exercise(bool wrong_notify) {
    // Count arrivals in units of 128 participating threads. Init WG1 -> WG0.
    std::array<int,2> arrivals{1,0};
    for (int iteration=0; iteration<64; ++iteration) {
        for (int wg=0; wg<2; ++wg) {
            int own = Pair::wait_id(wg)-4;
            if (++arrivals[own] != 2) return false;
            arrivals[own]=0;
            int next = (wrong_notify ? Pair::wait_id(wg) : Pair::notify_id(wg))-4;
            ++arrivals[next];
        }
    }
    return arrivals == std::array<int,2>{1,0};
}
int main() {
    if (!exercise(false) || exercise(true)) return 1;
    std::puts("[SM90 ordered pair] 128 ordered turns PASS; wrong-notify EXPECTED_RED/PASS");
}
