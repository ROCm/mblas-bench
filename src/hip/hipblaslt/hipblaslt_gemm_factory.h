#pragma once 
#include <generic_gemm_factory.h>

class hipblaslt_gemm_factory : public generic_gemm_factory {
 public:
  void create_gemm(cxxopts::ParseResult) override;
};