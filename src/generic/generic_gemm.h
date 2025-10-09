#pragma once
#include <string>
#include <utility>

#include "cxxopts.hpp"
enum class scaling_type {None, Scalar, Vector, Block};
class generic_gemm {
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

  bool strided = false;
  bool batched = false;
  bool pure_batched = false;

  int iters;
  int cold_iters;

  int batch_count;
  int flush_batch_count;
  int flush_memory_size;

  bool control_a = false;
  bool control_b = false;
  bool control_c = false;
  bool control_d = false;

  float constant_a;
  float constant_b;
  float constant_c;
  float constant_d;

  float scale_factor_a;
  float scale_factor_b;
  float scale_factor_c;
  float scale_factor_d;

  std::string filename_a;
  std::string filename_b;
  std::string filename_c;
  std::string filename_d;

  double gflop_per_second = 0;
  double gbyte_per_second = 0;
  double iter_time_us = 0;

  float avg_sysclk_mhz = 0;
  float med_sysclk_mhz = 0;
  float avg_memclk_mhz = 0;
  float med_memclk_mhz = 0;

  std::string function;

  std::string initialization;

  int requested_solution_count = 1;
  int returned_algo_count = 0;

 public:
  generic_gemm(cxxopts::ParseResult);

  // virtual void setSize();
  // virtual void setTypes();
  int set_ld(std::string ld, std::string OP, int x, int y);
  std::pair<int, int> set_row_col(std::string OP, int d1, int d2);

  virtual std::string prepare_array() = 0;

  virtual double test(const int &ith_solution) = 0;

  virtual std::string get_result_string() = 0;
  virtual void free_mem() = 0;

  static long long int fix_stride(long long int stride, long rows_a, long cols_a, std::string matrix_id);
  void set_init_params();

  void set_flush_batch_count(int a_type_size,  int b_type_size, int c_type_size, int d_type_size, 
                        int a_type_packing,  int b_type_packing, int c_type_packing, int d_type_packing, 
                        bool inplace);

  int get_returned_algo_count();

};
