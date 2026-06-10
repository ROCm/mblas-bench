#include <cxxopts.hpp>
#include <generic_gemm.h>
#include <cublas_gemm.h>
#include <cublas_gemm_factory.h>

//generic_gemm* cublas_gemm_factory::create_gemm(cxxopts::ParseResult result) const {
//  return new cublas_gemm(result);
//}

void cublas_gemm_factory::create_gemm(cxxopts::ParseResult result) {
  gemm = new cublas_gemm(result);
}