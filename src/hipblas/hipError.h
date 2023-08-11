#pragma once
#include <assert.h>
#include <hip/hipblas.h>
#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>

const char *hipblasGetErrorString(hipblasStatus_t status);
//
//// Convenience function for checking CUDA runtime API results
//// can be wrapped around any runtime API call. No-op in release builds.
// inline cudaError_t checkCuda(cudaError_t result);
//
// inline cublasStatus_t checkCublas(cublasStatus_t result);

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
static inline hipdaError_t checkHip(hipError_t result) {
  if (result != hipSuccess) {
    std::cerr << "HIP Runtime Error: " << hipGetErrorString(result)
              << std::endl;
    // fprintf(stderr, "HIP Runtime Error: %s\n", hipGetErrorString(result));
    assert(result == hipSuccess);
  }
  return result;
}

static inline hipblasStatus_t checkHipblas(hipblasStatus_t result) {
  if (result != HIPBLAS_STATUS_SUCCESS) {
    std::cerr << "hipBLAS Runtime Error: " << hipblasStatusToString(result)
              << std::endl;
    // fprintf(stderr, "HIP Runtime Error: %s\n",
    // hipblasStatusToString(result));
    assert(result == HIPBLAS_STATUS_SUCCESS);
  }
  return result;
}