#include <assert.h>
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>

const char *rocblasGetErrorString(hipblasStatus_t status) {
  switch (status) {
    case rocblas_status_success:
      return "rocblas_status_success";
    case rocblas_status_invalid_handle:
      return "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented:
      return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer:
      return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:
      return "rocblas_status_invalid_size";
    case rocblas_status_memory_error:
      return "rocblas_status_memory_error";
    case rocblas_status_internal_error:
      return "rocblas_status_internal_error";
    case rocblas_status_perf_degraded:
      return "rocblas_status_perf_degraded";
    case rocblas_status_size_query_mismatch:
      return "rocblas_status_size_query_mismatch";
    case rocblas_status_size_increased:
      return "rocblas_status_size_increased";
    case rocblas_status_size_unchanged:
      return "rocblas_status_size_unchanged";
    case rocblas_status_invalid_value:
      return "rocblas_status_invalid_value";
    case rocblas_status_continue:
      return "rocblas_status_continue";
    case rocblas_status_check_numerics_fail:
      return "rocblas_status_check_numerics_fail";
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