#include <cxxopts.hpp>
#include <genericGemm.h>
#include <rocblasGemm.h>
#include <rocblasGemmFactory.h>

void rocblasGemmFactory::create_gemm(cxxopts::ParseResult result) {
  gemm = new rocblasGemm(result);
}