#include "hip_convert.h"
#include <hip/hip_runtime.h>
#if HIP_VERSION >= 70000000
#include <hip/hip_fp6.h>
#include <mblas_data.h>

__global__ void float_to_fp6(float *input, size_t num_elements,
                              __hip_fp6_storage_t *output, __hip_fp6_interpretation_t interp)
{
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = __hip_cvt_float_to_fp6(input[idx], interp, hipRoundNearest);
  }
}

void float_to_fp6_helper(float *input, size_t num_elements, __hip_fp8_storage_t *output, mblas_fp6_interpretation_t minterp, size_t num_blocks, size_t block_size) {
  __hip_fp6_interpretation_t interp;
  switch(minterp) {
    case MBLAS_F6_E3M2:
      interp = __HIP_E3M2;
      break;
    case MBLAS_F6_E2M3:
      interp = __HIP_E2M3;
      break;
  }
  float_to_fp6<<<num_blocks, block_size>>>(input, num_elements,
                                             output, interp);
}
#endif

