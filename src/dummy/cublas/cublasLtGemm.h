#pragma once

#include "genericGemm.h"
#include <exception>
#include "cxxopts.hpp"

class cublasLtGemm : public genericGemm {
 public:
  cublasLtGemm(cxxopts::ParseResult result) : genericGemm(result) {
    throw std::runtime_error("Support for cublaslt backend not compiled");
  }

  std::string prepareArray() { return ""; }
  double test() { return 0.0;}
  std::string getResultString() { return "";}
  void freeMem() {}
};
