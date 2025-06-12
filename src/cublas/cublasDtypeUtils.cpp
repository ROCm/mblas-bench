#include "cublasDtypeUtils.h"

#include <cuda_runtime.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>
using namespace std;

static size_t roundoff(size_t  x, size_t granul) {
  return granul * ((x + (granul - 1)) / granul);
}

bool match_gemm_type(mblasDataType precision, std::string function, mblasDataType desiredPrec, std::vector<string> acceptable) {
  if (precision != desiredPrec) {
    return false;
  }
  for (auto afunc : acceptable) {
    if (function == afunc) {
      return true;
    }
  }
  return false;
}

// From https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/Common/helpers.h
// Block scales used for mxfp8 and nvfp8 require a special layout: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout for more details.

#if (CUDART_VERSION >= 12080)
std::pair<size_t, size_t> get_scale_tensor_size(int rows, int cols, cublasLtMatmulMatrixScale_t ScaleMode) {
  if (ScaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F)
    return std::pair<size_t, size_t>(1, 1);

  if (ScaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 || ScaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3) {
    static const size_t S_VSCALE = ScaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 ? 32 : 16;
    static const size_t S_BLOCK_COLS = 32;
    static const size_t S_BLOCK_ROWS = 4;
    static const size_t S_BLOCK_INNER = 4;

    static const size_t BLOCK_ROWS = S_BLOCK_INNER * S_VSCALE;
    static const size_t BLOCK_COLS = S_BLOCK_COLS * S_BLOCK_ROWS;

    size_t s_rows = roundoff(size_t(rows), BLOCK_ROWS) / S_VSCALE;
    size_t s_cols = roundoff(size_t(cols), BLOCK_COLS);

    return std::pair<size_t, size_t>(s_rows, s_cols);
  }

  return std::pair<size_t, size_t>(0, 0);
}
#endif