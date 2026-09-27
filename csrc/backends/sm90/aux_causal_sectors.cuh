// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/tensor.hpp>

namespace gdn::sm90 {

// Derive the sector from the real MMA fragment, not a second lane formula.
// Within a warp, lane-dependent columns remain inside that same16-column band.
template<class Mma,int I>
CUTE_HOST_DEVICE constexpr int aux_column_sector() {
    using namespace cute;
    constexpr auto coords=Mma{}.get_thread_slice(0).partition_C(
        make_identity_tensor(Shape<_64,_64>{}));
    return int(get<1>(coords(Int<I>{})))/16;
}

template<class Mma,int Sector,class Visit>
CUTE_HOST_DEVICE void visit_aux_sector(Visit visit) {
    using namespace cute;
    constexpr auto coords=Mma{}.get_thread_slice(0).partition_C(
        make_identity_tensor(Shape<_64,_64>{}));
    for_each(make_seq<size(coords)>{},[&](auto i) {
        if constexpr (aux_column_sector<Mma,decltype(i)::value>()==Sector) visit(i);
    });
}

template<class Mma,class Compute,class Clear>
CUTE_HOST_DEVICE void for_aux_causal_sectors(int local_tid,Compute compute,Clear clear) {
    using namespace cute;
    int warp=local_tid/32;
    for_each(make_seq<4>{},[&](auto sector) {
        constexpr int column_band=decltype(sector)::value;
        if(warp>=column_band) visit_aux_sector<Mma,column_band>(compute);
        else visit_aux_sector<Mma,column_band>(clear);
    });
}
} // namespace gdn::sm90
