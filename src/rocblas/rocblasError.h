#pragma once
#include <assert.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>

const char *rocblas_get_error_string(rocblas_status status);
const char *hipblas_get_error_string(hipblasStatus_t status);
//
//// Convenience function for checking CUDA runtime API results
//// can be wrapped around any runtime API call. No-op in release builds.
// inline cudaError_t check_cuda(cudaError_t result);
//
// inline cublasStatus_t check_cublas(cublasStatus_t result);

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
static inline hipError_t check_hip(hipError_t result) {
  if (result != hipSuccess) {
    std::cerr << "HIP Runtime Error: " << hipGetErrorString(result)
              << std::endl;
    // fprintf(stderr, "HIP Runtime Error: %s\n", hipGetErrorString(result));
    assert(result == hipSuccess);
  }
  return result;
}

static inline rocblas_status check_rocblas(rocblas_status result) {
  if (result != rocblas_status_success) {
    std::cerr << "rocBLAS Runtime Error: " << rocblas_status_to_string(result)
              << std::endl;
    // fprintf(stderr, "HIP Runtime Error: %s\n",
    // hipGetErrorString(result));
    assert(result == rocblas_status_success);
  }
  return result;
}

static inline hipblasStatus_t check_hipblas(hipblasStatus_t result) {
  if (result != HIPBLAS_STATUS_SUCCESS) {
    std::cerr << "hipBLAS Runtime Error: " << hipblasStatusToString(result)
              << std::endl;
    // fprintf(stderr, "HIP Runtime Error: %s\n",
    // hipGetErrorString(result));
    assert(result == HIPBLAS_STATUS_SUCCESS);
  }
  return result;
}