#pragma once
#include <string>
#include <utility>

#include "cxxopts.hpp"
enum class scaling_type {None, Scalar, Vector, Block};
std::string scaling_string(scaling_type input);
class generic_gemm {
 protected:
  struct matrix_desc {
    int rows;
    int cols;
    int rows_mem;
    int cols_mem;
    long long int stride;
    bool control;
    float constant;
    float scale_factor;
    scaling_type scale_mode = scaling_type::None;
    std::string init;
  };

  matrix_desc a_props;
  matrix_desc b_props;
  matrix_desc c_props;
  matrix_desc d_props;

  int m;
  int n;
  int k;

  int & rows_a = a_props.rows;
  int & cols_a = a_props.cols;
  int & rows_b = b_props.rows;
  int & cols_b = b_props.cols;
  int & rows_c = c_props.rows;
  int & cols_c = c_props.cols;
  int & rows_d = d_props.rows;
  int & cols_d = d_props.cols;

  int & rows_mem_a = a_props.rows_mem;
  int & cols_mem_a = a_props.cols;
  int & rows_mem_b = b_props.rows_mem;
  int & cols_mem_b = b_props.cols;
  int & rows_mem_c = c_props.rows_mem;
  int & cols_mem_c = c_props.cols;
  int & rows_mem_d = d_props.rows_mem;
  int & cols_mem_d = d_props.cols_mem;

  int lda;
  int ldb;
  int ldc;
  int ldd;

  long long int & stride_a = a_props.stride;
  long long int & stride_b = b_props.stride;
  long long int & stride_c = c_props.stride;
  long long int & stride_d = d_props.stride;

  bool strided = false;
  bool batched = false;
  bool pure_batched = false;

  int iters;
  int cold_iters;

  int batch_count;
  int flush_batch_count;
  int flush_memory_size;

  bool & control_a = a_props.control;
  bool & control_b = b_props.control;
  bool & control_c = c_props.control;
  bool & control_d = d_props.control;

  float & constant_a = a_props.constant;
  float & constant_b = b_props.constant;
  float & constant_c = c_props.constant;
  float & constant_d = d_props.constant;

  float & scale_factor_a = a_props.scale_factor;
  float & scale_factor_b = b_props.scale_factor;
  float & scale_factor_c = c_props.scale_factor;
  float & scale_factor_d = d_props.scale_factor;

  scaling_type & scale_mode_a = a_props.scale_mode;
  scaling_type & scale_mode_b = b_props.scale_mode;
  scaling_type & scale_mode_c = c_props.scale_mode;
  scaling_type & scale_mode_d = d_props.scale_mode;

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
  std::string scale_init;

  int requested_solution_count = 0;
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
  static scaling_type set_scale_mode(std::string input);
  static std::string set_init(matrix_desc desc, std::string init, std::string mx_init);

  void set_flush_batch_count(int a_type_size,  int b_type_size, int c_type_size, int d_type_size, 
                        int a_type_packing,  int b_type_packing, int c_type_packing, int d_type_packing, 
                        bool inplace);

  int get_returned_algo_count();

};

