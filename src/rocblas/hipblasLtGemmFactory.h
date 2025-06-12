#pragma once 
#include <genericGemmFactory.h>

class hipblasLtGemmFactory : public genericGemmFactory {
 public:
  void create_gemm(cxxopts::ParseResult) override;
};