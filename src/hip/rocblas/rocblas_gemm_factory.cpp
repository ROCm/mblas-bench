#include <cxxopts.hpp>
#include <generic_gemm.h>
#include <rocblas_gemm.h>
#include <rocblas_gemm_factory.h>

void rocblas_gemm_factory::create_gemm(cxxopts::ParseResult result) {
  gemm = new rocblas_gemm(result);
}