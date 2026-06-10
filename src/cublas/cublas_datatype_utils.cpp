#include "generic_setup.h"
#include "cublas_datatype_utils.h"

#include <cuda_runtime.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>
using namespace std;

static size_t roundoff(size_t  x, size_t granul) {
  return granul * ((x + (granul - 1)) / granul);
}

bool match_gemm_type(mblas_data_type precision, std::string function, mblas_data_type desiredPrec, std::vector<string> acceptable) {
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

#if (ENABLE_CUDA_FP4)
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


// Reference - https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuBLAS/utils/cublas_utils.h#L348
uint64_t get_fixed_point_workspace_size_in_bytes(
  int m, int n, int k, int batchCount, bool isComplex, int mantissaControl, int maxMantissaBitCount) 
{
  uint64_t mult = isComplex ? 2 : 1;
  if (maxMantissaBitCount == 0) {
      maxMantissaBitCount = 79;
  }
  uint64_t numSlices = ceil_division(maxMantissaBitCount + 1, 8);
  uint64_t padded_m = ceil_division(m, 1024) * 1024;
  uint64_t padded_n = ceil_division(n, 1024) * 1024;
  uint64_t padded_k = ceil_division(k, 128) * 128;
  uint64_t num_blocks_k = ceil_division(k, 64);

  uint64_t gemm_workspace = sizeof(int8_t) *
      ((uint64_t)padded_m * padded_k + (uint64_t)padded_n * padded_k) * mult * numSlices;

  gemm_workspace += sizeof(int32_t) * ((uint64_t)padded_m + padded_n) * mult;
  if (isComplex) {
      gemm_workspace += sizeof(double) * (uint64_t)m * n * mult * mult;
  }

  uint64_t adp_workspace = 0;
  if (mantissaControl == 0) {
      adp_workspace = sizeof(int32_t) * ((uint64_t)m * num_blocks_k + (uint64_t)n * num_blocks_k + (uint64_t)m * n) * mult;
  }

  constexpr uint64_t CONSTANT_SIZE = 128 * 1024 * 1024;
  uint64_t min_workspace = std::max(gemm_workspace, adp_workspace) * batchCount;
  uint64_t total_workspace = min_workspace + min_workspace / 4 + CONSTANT_SIZE; // 1.25x (avoiding double rounding/precision loss) plus constant size for safety margin (CUDA sample code)
  return total_workspace;
}
