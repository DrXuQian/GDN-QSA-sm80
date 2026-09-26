#include "target.cuh"
#include "launch.h"
#include <cute/tensor.hpp>
#include "kda/sm90/device/device_universal.hpp"
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "value_types.cuh"
#ifdef GDN_SM90_PRECOMPUTED_AUX
#include "precomputed_types.cuh"
#endif
#include <climits>
#include <stdexcept>

namespace gdn::sm90 {
using namespace cute;
using namespace kda::sm90::kernel;
using BF16 = cutlass::bfloat16_t;

template <class Gate, bool Initial>
void run(Arguments const& a, cudaStream_t stream) {
#ifdef GDN_SM90_PRECOMPUTED_AUX
    using Types = PrecomputedKernelTypes<Gate,Initial,GDN_SM90_PRECOMPUTED_VALUE_TILE>;
#elif defined(GDN_SM90_VALUE_SPLIT_AUX_REGS)
    using Types = ValueKernelTypes<Gate,Initial,64,GDN_SM90_VALUE_SPLIT_AUX_REGS>;
#else
    using Types = ValueKernelTypes<Gate,Initial>;
#endif
    using Kernel = typename Types::Kernel;
    using Operation = cutlass::device::Universal<Kernel>;
    typename Operation::Arguments args{};
    args.problem_size.total_seqlen = int64_t(a.batch) * a.length;
    args.problem_size.num_seqs = a.batch;
    args.problem_size.num_qk_heads = a.qk_heads;
    args.problem_size.num_v_heads = a.v_heads;
    args.problem_size.head_size = 128;
    args.problem_size.sequence_length = a.length;
    auto& m = args.mainloop;
    auto set_inputs = [&](auto& m) {
        m.ptr_Q = static_cast<BF16 const*>(a.q); m.dQ = {int64_t(a.qk_heads)*128, _1{}, 128};
        m.ptr_K = static_cast<BF16 const*>(a.k); m.dK = m.dQ;
        m.ptr_V = static_cast<BF16 const*>(a.v); m.dV = {int64_t(a.v_heads)*128, _1{}, 128};
        m.ptr_O = static_cast<BF16*>(a.output); m.dO = m.dV;
        m.ptr_Gate = static_cast<Gate const*>(a.gate);
        m.beta_ptr = static_cast<BF16 const*>(a.beta); m.beta_stride = {a.v_heads, 1};
        m.ptr_input_state = a.initial; m.ptr_output_state = a.final;
        m.scale = 0.08838834764831844f; // 1/sqrt(128), public GDN contract
    };
    set_inputs(m);
#ifdef GDN_SM90_PRECOMPUTED_AUX
    m.prepared = static_cast<BF16*>(a.prepared);
#endif
    args.hw_info.sm_count = 0; // grid=B*Hv, no SM-dependent scratch/dispatch
    Operation op;
    if (op.get_workspace_size(args) != 0 || op.can_implement(args) != cutlass::Status::kSuccess)
        throw std::runtime_error("GDN fused SM90 invalid kernel arguments/workspace");
    if (op.initialize(args, nullptr, stream) != cutlass::Status::kSuccess)
        throw std::runtime_error("GDN fused SM90 initialize failed");
#ifdef GDN_SM90_PRECOMPUTED_AUX
    using Prepare = typename Types::Prepare;
    using InputBase = typename Types::InputBase;
    typename InputBase::Arguments input{};
    set_inputs(input);
    PreparedProblem problem{int64_t(a.batch)*a.length,a.batch,a.qk_heads,a.v_heads,128,a.length};
    typename Prepare::Params p{InputBase::to_underlying_arguments(problem,input,nullptr),
        problem,static_cast<BF16*>(a.prepared),a.v_heads/a.qk_heads,
        PreparedAuxLayout::chunks(a.length)};
    constexpr int shared = sizeof(typename Prepare::SharedStorage);
    if (cudaFuncSetAttribute(prepare_aux_device<Prepare>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared) != cudaSuccess)
        throw std::runtime_error("GDN prepare shared attribute failed");
    prepare_aux_device<Prepare><<<dim3(a.batch*a.v_heads,p.chunk_count),128,shared,stream>>>(p);
    if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("GDN parallel auxiliary preparation launch failed");
#endif
    if (op.run(stream) != cutlass::Status::kSuccess)
        throw std::runtime_error("GDN fused SM90 initialize/launch failed");
}

#ifdef GDN_SM90_PRECOMPUTED_AUX
template<class Gate, bool Initial>
PreparedResources query_precomputed_resources() {
    using Types = PrecomputedKernelTypes<Gate,Initial,GDN_SM90_PRECOMPUTED_VALUE_TILE>;
    using Kernel = typename Types::Kernel;
    using Prepare = typename Types::Prepare;
    cudaFuncAttributes pa{},sa{};
    int prep_blocks = -1;
    constexpr int shared = sizeof(typename Prepare::SharedStorage);
    if (cudaFuncSetAttribute(prepare_aux_device<Prepare>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared) != cudaSuccess ||
        cudaFuncGetAttributes(&pa,prepare_aux_device<Prepare>) != cudaSuccess ||
        cudaFuncGetAttributes(&sa,cutlass::device_kernel<Kernel>) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&prep_blocks,prepare_aux_device<Prepare>,128,shared) != cudaSuccess)
        throw std::runtime_error("GDN prepared resources query failed");
    int state_blocks = cutlass::device::Universal<Kernel>::maximum_active_blocks();
    if (state_blocks < 0) throw std::runtime_error("GDN state occupancy query failed");
    return {{128,shared,pa.numRegs,int(pa.localSizeBytes),prep_blocks},
            {Kernel::MaxThreadsPerBlock,Kernel::SharedStorageSize,sa.numRegs,int(sa.localSizeBytes),state_blocks}};
}
PreparedResources precomputed_resources(bool fp32, bool initial) {
    if (fp32) return initial ? query_precomputed_resources<float,true>() : query_precomputed_resources<float,false>();
    return initial ? query_precomputed_resources<BF16,true>() : query_precomputed_resources<BF16,false>();
}
#endif

void launch(Arguments const& a, cudaStream_t stream) {
    if (a.batch <= 0 || a.length <= 0 || a.qk_heads <= 0 || a.v_heads <= 0 ||
        a.v_heads % a.qk_heads || int64_t(a.batch)*a.length > INT32_MAX ||
        int64_t(a.batch)*a.v_heads > INT32_MAX)
        throw std::invalid_argument("GDN fused SM90 invalid shape/GVA/grid");
#ifdef GDN_SM90_PRECOMPUTED_AUX
    if (!a.prepared || reinterpret_cast<uintptr_t>(a.prepared)%128 ||
        PreparedAuxLayout::chunks(a.length) > 65535)
        throw std::invalid_argument("GDN prepared scratch/grid admission failed");
#endif
    if (a.gate_fp32) {
        if (a.initial) run<float, true>(a, stream); else run<float, false>(a, stream);
    } else {
        if (a.initial) run<BF16, true>(a, stream); else run<BF16, false>(a, stream);
    }
}
} // namespace gdn::sm90
