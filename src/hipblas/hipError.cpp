#include <assert.h>
#include <hip/hipblas.h>
#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>

const char *hipblasGetErrorString(hipblasStatus_t status) {
  switch (status) {
    case HIPBLAS_STATUS_SUCCESS:
      return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:
      return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:
      return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:
      return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_MAPPING_ERROR:
      return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED:
      return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:
      return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED:
      return "HIPBLAS_STATUS_NOT_SUPPORTED";
    case HIPBLAS_STATUS_ARCH_MISMATCH:
      return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
      return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
    case HIPBLAS_STATUS_INVALID_ENUM:
      return "HIPBLAS_STATUS_INVALID_ENUM";
    case HIPBLAS_STATUS_UNKNOWN:
      return "HIPBLAS_STATUS_UNKNOWN";
  }
  return "unknown error";
}

// // Convenience function for checking CUDA runtime API results
// // can be wrapped around any runtime API call. No-op in release builds.
// inline cudaError_t checkCuda(cudaError_t result) {
//   if (result != cudaSuccess) {
//     fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
//     assert(result == cudaSuccess);
//   }
//   return result;
// }

// inline cublasStatus_t checkCublas(cublasStatus_t result) {
//   if (result != CUBLAS_STATUS_SUCCESS) {
//     fprintf(stderr, "CUDA Runtime Error: %s\n",
//     cublasGetErrorString(result)); assert(result == CUBLAS_STATUS_SUCCESS);
//   }
//   return result;
// }