#pragma once
#include <string>

#include "third_party/cxxopts.hpp"

class genericGemm {
 protected:
  int m;
  int n;
  int k;

  int lda;
  int ldb;
  int ldc;
  int ldd;

  long long int stride_a;
  long long int stride_b;
  long long int stride_c;
  long long int stride_d;

  bool strided;
  bool batched;
  int batchct;

  int iters;
  int cold_iters;

  double gflop_per_second = 0;
  double gbyte_per_second = 0;
  double iter_time_us = 0;
  std::string function;

 public:
  genericGemm(cxxopts::ParseResult);

  // virtual void setSize();
  // virtual void setTypes();
  int setLd(std::string ld, std::string OP, int x, int y);

  virtual std::string prepareArray() = 0;

  virtual double test() = 0;

  virtual std::string getResultString() = 0;
  virtual void freeMem() = 0;
};
