#include <assert.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>

const char *rocblas_get_error_string(rocblas_status status) {
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
    case rocblas_status_excluded_from_build:
      return "rocblas_status_excluded_from_build";
    case rocblas_status_arch_mismatch:
      return "rocblas_status_arch_mismatch";
  }
  return "unknown error";
}

const char *hipblas_get_error_string(hipblasStatus_t status) {
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
// inline cudaError_t check_cuda(cudaError_t result) {
//   if (result != cudaSuccess) {
//     fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
//     assert(result == cudaSuccess);
//   }
//   return result;
// }

// inline cublasStatus_t check_cublas(cublasStatus_t result) {
//   if (result != CUBLAS_STATUS_SUCCESS) {
//     fprintf(stderr, "CUDA Runtime Error: %s\n",
//     cublas_get_error_string(result)); assert(result == CUBLAS_STATUS_SUCCESS);
//   }
//   return result;
// }