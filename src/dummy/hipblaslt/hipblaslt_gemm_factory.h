#pragma once 
#include <generic_gemm_factory.h>
#include <exception>

class hipblaslt_gemm_factory : public generic_gemm_factory {
 public:
  void create_gemm(cxxopts::ParseResult) override {
    throw std::runtime_error("Support for hipblasLt backend not compiled");
  }
};