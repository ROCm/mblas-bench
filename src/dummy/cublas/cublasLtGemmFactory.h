#pragma once 
#include <genericGemmFactory.h>
#include <cxxopts.hpp>

class cublasLtGemmFactory : public genericGemmFactory {
 public:
  void create_gemm(cxxopts::ParseResult) override {
    throw std::runtime_error("Support for cublasLt backend not compiled");
  }
};