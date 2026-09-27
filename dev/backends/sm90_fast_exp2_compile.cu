#include <cuda_runtime.h>
#include "gate_math.cuh"

__global__ void gdn_gate_exp2_compile(float const* input, float* output) {
    int lane = int(threadIdx.x);
    output[lane] = gdn::sm90::gate_exp2(input[lane]);
}
