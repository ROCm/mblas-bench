#include <cxxopts.hpp>
#include <generic_gemm.h>
#include <cublaslt_gemm.h>
#include <cublaslt_gemm_factory.h>

void cublaslt_gemm_factory::create_gemm(cxxopts::ParseResult result) {
  gemm = new cublaslt_gemm(result);
}