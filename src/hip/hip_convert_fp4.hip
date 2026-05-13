#include "hip_convert.h"
#include <hip/hip_runtime.h>
#if HIP_VERSION >= 70000000
#include <hip/hip_fp4.h>

__global__ void float_to_fp4(float2 *input, size_t num_elements,
                              __hip_fp4_storage_t *output)
{
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = __hip_cvt_float2_to_fp4x2(input[idx], __HIP_E2M1, hipRoundNearest);
  }
}


void float_to_fp4_helper(float2 *input, size_t num_elements, __hip_fp8_storage_t *output, size_t num_blocks, size_t block_size) {
  float_to_fp4<<<num_blocks, block_size>>>(input, num_elements,
                                             output);
}
#endif

