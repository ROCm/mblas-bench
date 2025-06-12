#include <cxxopts.hpp>
#include <genericGemm.h>
#include <cublasLtGemm.h>
#include <cublasLtGemmFactory.h>

void cublasLtGemmFactory::create_gemm(cxxopts::ParseResult result) {
  gemm = new cublasLtGemm(result);
}