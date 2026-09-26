#pragma once
#include <cute/config.hpp>

namespace kerutils {
// Four complete warps per SM90 warpgroup. Keep this nonnegative bound visible
// to the compiler; subtracting two independently shuffled indices loses it.
CUTE_HOST_DEVICE constexpr int inverse_warp_in_group(unsigned thread) {
    return int((thread >> 5) & 3u);
}
} // namespace kerutils
