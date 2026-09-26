// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cutlass/arch/barrier.h>

namespace gdn::sm90 {

// Auxiliary-owned inverse only: the two state WGs own disjoint V rows of H
// and O. Their TMA and shared pipelines still wait for ALL consumers before
// recycling a stage, and O publication still requires all256 producers.
// This policy removes only a tensor-issue preference, never data completion.
// It must not be selected for the older state-WG-owned shared inverse.
struct IndependentStateIssue {
    CUTE_DEVICE void init(int) {}
    CUTE_DEVICE void ordered_or_wait(int) {}
    CUTE_DEVICE void notify_next_blocked(int) {}
};

// Two state warpgroups, fixed named IDs. Do not materialize an indexed array:
// nvcc otherwise lowers the runtime WG lookup to LDL on the critical path.
// This is the same ordered protocol, NOT a removed wait or a polling shortcut.
template<uint32_t First, uint32_t Second>
struct OrderedPair {
    static_assert(First != Second, "state warpgroups need distinct barriers");
    static constexpr int NumWG = 2;
    static constexpr int Participants = 256;
    using Id = cutlass::arch::ReservedNamedBarriers;

    CUTE_HOST_DEVICE static constexpr uint32_t wait_id(int wg) {
        return wg == 0 ? First : Second;
    }
    CUTE_HOST_DEVICE static constexpr uint32_t notify_id(int wg) {
        return wg == 0 ? Second : First;
    }
    CUTE_DEVICE void init(int wg) {
        // WG1 supplies the first half of WG0's initial arrival count.
        if (wg == 1) cutlass::arch::NamedBarrier::arrive(Participants, Id(First));
    }
    CUTE_DEVICE void ordered_or_wait(int wg) {
        cutlass::arch::NamedBarrier::sync(Participants, Id(wait_id(wg)));
    }
    CUTE_DEVICE void notify_next_blocked(int wg) {
        cutlass::arch::NamedBarrier::arrive(Participants, Id(notify_id(wg)));
    }
};

} // namespace gdn::sm90
