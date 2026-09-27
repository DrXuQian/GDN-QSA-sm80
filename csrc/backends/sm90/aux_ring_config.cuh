// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kda/sm90/kernel/options.hpp"
#include <cutlass/bfloat16.h>
#include <cute/numeric/integral_constant.hpp>
#include <type_traits>

#ifndef GDN_SM90_AUX_RING_CONFIG
#define GDN_SM90_AUX_RING_CONFIG 0
#endif

namespace gdn::sm90 {
// This is a bounded experimental inventory, not production shape dispatch.
struct AuxRings {
    static_assert(GDN_SM90_AUX_RING_CONFIG >= 0 && GDN_SM90_AUX_RING_CONFIG < 4);
    static constexpr int QK = (GDN_SM90_AUX_RING_CONFIG & 1) ? 1 : 2;
    static constexpr int KK = (GDN_SM90_AUX_RING_CONFIG & 2) ? 1 : 2;

    template<class Gate, bool Initial>
    using OriginalOptions = std::tuple<
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kElementGateGmem, Gate>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kElementBetaGmem, cutlass::bfloat16_t>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kInitStateFromInput, cute::bool_constant<Initial>>>;

    template<class Gate, bool Initial>
    using Options = std::conditional_t<GDN_SM90_AUX_RING_CONFIG == 0,
        OriginalOptions<Gate, Initial>,
        decltype(std::tuple_cat(OriginalOptions<Gate, Initial>{}, std::tuple<
            kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kStagesQK, cute::Int<QK>>,
            kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kStagesKK, cute::Int<KK>>>{}))>;
};
} // namespace gdn::sm90
