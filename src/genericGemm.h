#pragma once
#include <string>
#include <utility>

#include "third_party/cxxopts.hpp"

class genericGemm {
 protected:
  int m;
  int n;
  int k;

  int rowsA;
  int colsA;
  int rowsB;
  int colsB;
  int rowsC;
  int colsC;
  int rowsD;
  int colsD;

  int rowsMemA;
  int colsMemA;
  int rowsMemB;
  int colsMemB;
  int rowsMemC;
  int colsMemC;
  int rowsMemD;
  int colsMemD;

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

  bool controlA = false;
  bool controlB = false;
  bool controlC = false;
  bool controlD = false;

  float constantA = 2.f;
  float constantB = 3.f;
  float constantC = 1.f;
  float constantD = 1.f;

  double gflop_per_second = 0;
  double gbyte_per_second = 0;
  double iter_time_us = 0;
  std::string function;

  std::string initialization;

 public:
  genericGemm(cxxopts::ParseResult);

  // virtual void setSize();
  // virtual void setTypes();
  int setLd(std::string ld, std::string OP, int x, int y);
  std::pair<int, int> setRowCol(std::string OP, int d1, int d2);

  virtual std::string prepareArray() = 0;

  virtual double test() = 0;

  virtual std::string getResultString() = 0;
  virtual void freeMem() = 0;
};
