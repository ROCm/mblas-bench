#pragma once
#include <hipblaslt/hipblaslt.h>
// #include <hip/hip_runtime.h>
#include <cxxabi.h>

#include <iostream>
#include <string>

#include "generic_gemm.h"
#include "mblas_hip_data_type.h"
#include "mblas_hipblas_compute_type.h"
#include "mblas_hipblas_operation.h"

#if HIP_VERSION >= 70000000
struct scale_size {
  long rows;
  long cols;
  inline size_t get_size() { return rows*cols; }
  scale_size(std::pair<size_t,size_t> size) : rows(size.first), cols(size.second) {}
  scale_size(){};
};
#endif

struct matmul_prec_type {
  mblas_hipblas_compute_type compute;
  mblas_hip_data_type scalar;
  mblas_hip_data_type a_type;
  mblas_hip_data_type b_type;
  mblas_hip_data_type c_type;
  mblas_hip_data_type d_type;
  mblas_hip_data_type bias_type;
  bool operator==(const matmul_prec_type rhs) const {
    return rhs.compute == compute && rhs.scalar == scalar &&
           rhs.a_type == a_type && rhs.b_type == b_type &&
           rhs.c_type == c_type && rhs.d_type == d_type &&
           // Omitting bias type is acceptable
           (rhs.bias_type == bias_type ||
            rhs.bias_type == mblas_hip_data_type(MBLAS_ANY));
  }
};

struct matmul_prec_type_f8 {
  mblas_hip_data_type scalar;
  mblas_hip_data_type c_type;
  mblas_hip_data_type d_type;
  mblas_hip_data_type bias_type;
  bool operator==(const matmul_prec_type_f8 rhs) const {
    return rhs.scalar == scalar &&
           rhs.c_type == c_type && rhs.d_type == d_type &&
           // Omitting bias type is acceptable
           (rhs.bias_type == bias_type ||
            rhs.bias_type == mblas_hip_data_type(MBLAS_ANY));
  }
};

struct hipblaslt_gemm_inst {
  int devIDX;
  double gflops = 0;
  double gbytes = 0;
  double time_us = 0;
  //void *devA;
  //void *devB;
  //void *devC;
  //void *devD;
  void **ptr_dev_a;
  void **ptr_dev_b;
  void **ptr_dev_c;
  void **ptr_dev_d;
#if HIP_VERSION >= 70000000
  void *scale_dev_a;
  void *scale_dev_b;
  void *scale_dev_c;
  void *scale_dev_d;
#endif
  hipblasLtMatmulDesc_t desc_op;
  hipblasLtMatrixLayout_t desc_a;
  hipblasLtMatrixLayout_t desc_b;
  hipblasLtMatrixLayout_t desc_c;
  hipblasLtMatrixLayout_t desc_d;
  hipblasLtMatmulPreference_t pref;
  hipblasLtMatmulHeuristicResult_t algo;
  void *devWork;
  long wSZ;
  hipblaslt_gemm_inst(int devID) : devIDX(devID)
#if HIP_VERSION >= 70000000
    , scale_dev_a(nullptr), scale_dev_b(nullptr), scale_dev_c(nullptr), scale_dev_d(nullptr)
#endif
  {}
};

class hipblaslt_gemm : public generic_gemm {
 private:
  // void *host_a;
  // void *host_b;
  // void *host_c;

  void **ptr_host_a;
  void **ptr_host_b;
  void **ptr_host_c;
  void **ptr_host_d;

  void *alpha;
  void *beta;

  bool inplace = false;
  mblas_hipblas_operation transA;
  mblas_hipblas_operation transB;

  mblas_hip_data_type precision;
  mblas_hipblas_compute_type compute;
  mblas_hip_data_type scalar;
  mblas_hip_data_type a_type;
  mblas_hip_data_type b_type;
  mblas_hip_data_type c_type;
  mblas_hip_data_type d_type;
  mblas_hip_data_type bias_type;

  bool use_scaling = false;
  
  mblas_hip_data_type a_scale_type;
  mblas_hip_data_type b_scale_type;
  mblas_hip_data_type c_scale_type;
  mblas_hip_data_type d_scale_type;
  
#if HIP_VERSION >= 70000000
  scale_size a_scale_size;
  scale_size b_scale_size;
  scale_size c_scale_size;
  scale_size d_scale_size;
  
  hipblasLtMatmulMatrixScale_t a_scale_mode;
  hipblasLtMatmulMatrixScale_t b_scale_mode;
  hipblasLtMatmulMatrixScale_t c_scale_mode;
  hipblasLtMatmulMatrixScale_t d_scale_mode;
#endif

  void *scale_host_a;
  void *scale_host_b;
  void *scale_host_c;
  void *scale_host_d;

  int workspace_size = 64 * 1024 * 1024;

  static std::vector<matmul_prec_type> matmul_supported;
  static std::vector<matmul_prec_type_f8> matmulSupportedF8;
#if HIP_VERSION >= 70000000
  static std::vector<matmul_prec_type> mx_matmul_supported;
#endif
  std::vector<hipblaslt_gemm_inst> mat_ptrs;

 private:
  // mblas_hip_data_type precisionStringToHipblasDType(std::string stringPrecision);
  // void parse_problem_type(std::string a, std::string b, std::string c);
  void parse_problem_type(std::string computeTStr, std::string scalarTStr,
                  std::string aStr, std::string bStr, std::string cStr,
                  std::string dStr);
  void validate_parameters();
  void parse_dev_iters(std::string);
  void alloc_host();
  void alloc_dev(hipblaslt_gemm_inst *);
  void fill_host();
  void copy_host_to_dev(hipblaslt_gemm_inst *);
  void prepare_matrix(hipblaslt_gemm_inst *);
  void no_tuning(hipblaslt_gemm_inst *);
  void auto_tuning(hipblaslt_gemm_inst *);
  void run_threaded(void (hipblaslt_gemm::*func)(hipblaslt_gemm_inst *));
  std::tuple<double, double, double> calculate_figure_of_merit(double totalTime_ms);
  void test_matmul(hipblaslt_gemm_inst *mat);
#if HIP_VERSION >= 70000000
  std::tuple<mblas_hip_data_type, hipblasLtMatmulMatrixScale_t, scale_size> 
    configure_scaling(matrix_desc desc, mblas_hip_data_type type, std::string matrix_id);
#endif

 public:
  hipblaslt_gemm(cxxopts::ParseResult result);
  std::string prepare_array();
  double test();
  std::string get_result_string();
  virtual void free_mem();
};