// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace gdn::sm90 {

// One physical, swizzled BF16 H tile. Both SS consumers have the same
// logical (V,K) operand. This is private scratch, never a public state format.
template<class Base>
struct SharedStateLayout {
    using Layout = decltype(kda::sm90::collective::unstage_smem_layout(
        typename Base::CollectiveMmaO1::SmemLayoutA{}, cute::_1{}));
    using O1Mma = typename Base::CollectiveMmaO1::TiledMma;
    using SKMma = typename Base::CollectiveMmaSK::TiledMma;
    using Store = decltype(cute::make_tiled_copy_C(
        cute::Copy_Atom<cute::SM90_U32x4_STSM_N,typename Base::Element>{},
        typename Base::TiledMmaKV{}));
};

template<class Parent, bool Enabled>
struct StateOperandStorage : Parent::SharedStorage {};

template<class Parent>
struct StateOperandStorage<Parent,true> : Parent::SharedStorage {
    using Layout = typename SharedStateLayout<Parent>::Layout;
    alignas(128) cute::array_aligned<typename Parent::Element,cute::cosize_v<Layout>> state_operand;
};

} // namespace gdn::sm90
