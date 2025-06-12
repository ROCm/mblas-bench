#pragma once 
#include <genericGemmFactory.h>
#include <cxxopts.hpp>

class cublasLtGemmFactory : public genericGemmFactory {
 public:
  void create_gemm(cxxopts::ParseResult) override;
};