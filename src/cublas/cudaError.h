#pragma once
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>

const char *cublas_get_error_string(cublasStatus_t status);
//
//// Convenience function for checking CUDA runtime API results
//// can be wrapped around any runtime API call. No-op in release builds.
// inline cudaError_t check_cuda(cudaError_t result);
//
// inline cublasStatus_t check_cublas(cublasStatus_t result);

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
static inline cudaError_t check_cuda(cudaError_t result) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result)
              << std::endl;
    // fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

static inline cublasStatus_t check_cublas(cublasStatus_t result) {
  if (result != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS Runtime Error: " << cublas_get_error_string(result)
              << std::endl;
    // fprintf(stderr, "CUDA Runtime Error: %s\n",
    // cublas_get_error_string(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}