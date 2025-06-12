#pragma once 
#include <genericGemmFactory.h>
#include <cxxopts.hpp>

class cublasGemmFactory : public genericGemmFactory {
 public:
  void create_gemm(cxxopts::ParseResult) override;
};