// Actual auxiliary MMA/STSM view versus independently normalized logical input.
#include <cute/tensor.hpp>
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "inverse_unit_input.cuh"
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace cute;
using GdnStride = tuple<int64_t, _1, int32_t>;
using Builder = kda::sm90::kernel::FlatBuilderKdaFwd<cutlass::bfloat16_t, float, float,
    Shape<_64, _64, _128>, GdnStride, GdnStride, GdnStride, GdnStride,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
using Mainloop = Builder::CollectiveMainloop;

void require(bool ok) {
    if (!ok) throw std::runtime_error("inverse input/writer contract");
}

uint16_t bits(cutlass::half_t value) {
    uint16_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void check(int plant) {
    constexpr int stages = Mainloop::StagesKK::value;
    constexpr auto layout = Mainloop::SmemLayoutKK{};
    auto mma = Mainloop::TiledMmaQK{};
    auto writer = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>{}, mma);
    auto identity = make_identity_tensor(Shape<_64, _64>{});
    // Real 16-bit shared-pointer view required by the actual STSM trait. Tags
    // are physical half-element offsets, not numeric fp16 values or bytes.
    std::array<uint16_t, cosize_v<decltype(layout)>> tags{};
    for (int i = 0; i < int(tags.size()); ++i) tags[i] = uint16_t(i);
    auto physical = make_tensor(make_smem_ptr(tags.data()), layout);
    for (int valid = 1; valid <= 64; ++valid) {
        std::array<int, cosize_v<decltype(layout)>> owners{};
        for (int stage = 0; stage < stages; ++stage) {
            for (int tid = 0; tid < 128 - int(plant == 4); ++tid) {
                auto logical = mma.get_slice(tid).partition_C(identity);
                auto slice = writer.get_slice(tid);
                auto source = slice.retile_S(logical);
                auto destination = slice.partition_D(physical(_, _, stage));
                require(size(source) == size(destination));
                for (int i = 0; i < size(source); ++i) {
                    auto [row, col] = source(i);
                    int r = int(row), c = int(col), offset = destination(i);
                    require(r >= 0 && r < 64 && c >= 0 && c < 64);
                    require(offset == layout(r, c, stage));
                    ++owners.at(offset);
                    bool live = r >= c && r < valid && c < valid;
                    // Distinct exact values; the diagonal is deliberately not1.
                    float raw = float(1 + r * 64 + c) / 4096.0f;
                    float masked = live ? raw : 0.0f;
                    float got = gdn::sm90::inverse_unit_input(r, c, masked);
                    if (plant == 1) got = masked;  // old diagonal garbage
                    if (plant == 2 && r < c) got = raw;  // unmasked upper
                    if (plant == 3 && r == c && r >= valid) got = 0.0f;
                    // Independent: old storage was fp16, then solver installed I.
                    cutlass::half_t old_storage(masked);
                    cutlass::half_t want = r == c ? cutlass::half_t(1.0f)
                        : (r < c ? cutlass::half_t(0.0f) : old_storage);
                    require(bits(cutlass::half_t(got)) == bits(want));
                }
            }
        }
        for (int count : owners) require(count == 1);
    }
}

int main() {
    check(0);
    for (int plant = 1; plant <= 4; ++plant) {
        bool rejected = false;
        try { check(plant); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected);
    }
    std::cout << "inverse unit input: actual MMA/STSM writer 64tails*2stages*4096cells "
                 "byte-equal normalized input; old-diagonal/upper/padded-diagonal/missing-owner "
                 "EXPECTED_RED/PASS\n";
}
