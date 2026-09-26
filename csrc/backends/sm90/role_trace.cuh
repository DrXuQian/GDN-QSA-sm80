#pragma once
#include <cute/config.hpp>
#include "launch.h"

namespace gdn::sm90 {
#ifdef GDN_SM90_ROLE_TRACE
// Diagnostic-only: unique writer per (CTA, chunk, role, point), no atomics or
// synchronization. Never use these probe intervals as uninstrumented timing.
// This header is instantiated only in launch.cu; host bindings include launch.h.
static __device__ unsigned long long role_trace_data[TraceWords];
#endif

CUTE_DEVICE void trace_role(int chunk, int role, int point) {
#ifdef GDN_SM90_ROLE_TRACE
    if ((int(threadIdx.x) & 127) == 0) {
        unsigned long long stamp;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamp) :: "memory");
        int index = (((int(blockIdx.x)*TraceChunks+chunk)*TraceRoles+role)*TracePoints+point);
        reinterpret_cast<volatile unsigned long long*>(role_trace_data)[index] = stamp;
    }
#endif
}
} // namespace gdn::sm90
