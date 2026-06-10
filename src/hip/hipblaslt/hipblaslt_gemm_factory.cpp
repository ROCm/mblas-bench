#include <cxxopts.hpp>
#include <generic_gemm.h>
#include <hipblaslt_gemm.h>
#include <hipblaslt_gemm_factory.h>

void hipblaslt_gemm_factory::create_gemm(cxxopts::ParseResult result) {
  gemm = new hipblaslt_gemm(result);
}