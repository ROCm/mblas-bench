#include <cxxopts.hpp>
#include <genericGemm.h>
#include <hipblasLtGemm.h>
#include <hipblasLtGemmFactory.h>

void hipblasLtGemmFactory::create_gemm(cxxopts::ParseResult result) {
  gemm = new hipblasLtGemm(result);
}