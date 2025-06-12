#pragma once 
#include <genericGemmFactory.h>

class rocblasGemmFactory : public genericGemmFactory {
 public:
  void create_gemm(cxxopts::ParseResult) override;
};