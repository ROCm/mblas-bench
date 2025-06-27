#pragma once
#include <string>
#include <utility>

#include "cxxopts.hpp"

class genericGemm {
 protected:
  int m;
  int n;
  int k;

  int rows_a;
  int cols_a;
  int rows_b;
  int cols_b;
  int rows_c;
  int cols_c;
  int rows_d;
  int cold_d;

  int rows_mem_a;
  int cols_mem_a;
  int rows_mem_b;
  int cols_mem_b;
  int rows_mem_c;
  int cols_mem_c;
  int rows_mem_d;
  int cols_mem_d;

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

  int iters;
  int cold_iters;

  int batch_count;
  int flush_batch_count;
  int flush_memory_size;

  bool controlA = false;
  bool controlB = false;
  bool controlC = false;
  bool controlD = false;

  float constantA = 2.f;
  float constantB = 3.f;
  float constantC = 1.f;
  float constantD = 1.f;

  std::string filenameA;
  std::string filenameB;
  std::string filenameC;
  std::string filenameD;

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

  void set_flush_batch_count(uint64_t & a_offset, uint64_t & b_offset, uint64_t & c_offset, uint64_t & d_offset,
                        int a_type_size,  int b_type_size, int c_type_size, int d_type_size, 
                        int a_type_packing,  int b_type_packing, int c_type_packing, int d_type_packing, 
                        bool inplace);

};
