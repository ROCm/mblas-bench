#pragma once 
#include <generic_gemm_factory.h>
#include <cxxopts.hpp>

class cublaslt_gemm_factory : public generic_gemm_factory {
 public:
  void create_gemm(cxxopts::ParseResult) override {
    throw std::runtime_error("Support for cublasLt backend not compiled");
  }
};