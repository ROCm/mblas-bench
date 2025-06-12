#include <cxxopts.hpp>
#include <genericGemm.h>
#include <cublasGemm.h>
#include <cublasGemmFactory.h>

//genericGemm* cublasGemmFactory::create_gemm(cxxopts::ParseResult result) const {
//  return new cublasGemm(result);
//}

void cublasGemmFactory::create_gemm(cxxopts::ParseResult result) {
  gemm = new cublasGemm(result);
}