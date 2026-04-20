#pragma once

#include <cstdint>

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cxxabi.h>
#include <barrier.h>

#include <iostream>
#include <string>
#include <vector>

#include "generic_gemm.h"
#include "mblas_cuda_data_type.h"
#include "mblas_cuda_compute_type.h"
#include "mblas_cuda_operation.h"

struct matmul_prec_type {
  mblas_compute_type compute;
  mblas_data_type scalar;
  mblas_data_type a_type;
  mblas_data_type b_type;
  mblas_data_type c_type;
  mblas_data_type d_type;
  mblas_data_type bias_type;
  bool operator==(const matmul_prec_type rhs) const {
    return compute == rhs.compute && scalar == rhs.scalar &&
           a_type == rhs.a_type && b_type == rhs.b_type &&
           c_type == rhs.c_type && d_type == rhs.d_type &&
           // Omitting bias type is acceptable
           (bias_type == rhs.bias_type ||
            rhs.bias_type == mblas_data_type::MBLAS_ANY);
  }
};

struct cublaslt_gemm_inst {
  int devIDX;
  double gflops = 0;
  double gbytes = 0;
  double time_us = 0;
  //void *dataDev;
  //void **devA;
  //void **devB;
  //void **devC;
  //void **devD;
  void **ptr_dev_a;
  void **ptr_dev_b;
  void **ptr_dev_c;
  void **ptr_dev_d;
  void **scale_dev_a;
  void **scale_dev_b;
  void **scale_dev_c;
  void **scale_dev_d;
  std::vector<cublasLtMatmulDesc_t> desc_ops;
  cublasLtMatrixLayout_t desc_a;
  cublasLtMatrixLayout_t desc_b;
  cublasLtMatrixLayout_t desc_c;
  cublasLtMatrixLayout_t desc_d;
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulHeuristicResult_t algo;
#if defined(HAS_CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT)
    cublasLtEmulationDesc_t emulation_desc;
    int32_t algo_emulation_support = 0;
#endif
  void *devWork;
  uint64_t wSZ;
  cublaslt_gemm_inst(int devID)
    : devIDX(devID), scale_dev_a(nullptr), scale_dev_b(nullptr),
      scale_dev_c(nullptr), scale_dev_d(nullptr) {}
};

struct scale_size {
  long rows;
  long cols;
  inline size_t get_size() {
    return rows*cols;
  }
  scale_size(std::pair<size_t,size_t> size) : rows(size.first), cols(size.second) {}
  scale_size(){};
};

class cublaslt_gemm : public generic_gemm {
 private:
  void **ptr_host_a;
  void **ptr_host_b;
  void **ptr_host_c;
  void **ptr_host_d;

  void **scale_host_a = nullptr;
  void **scale_host_b = nullptr;
  void **scale_host_c = nullptr;
  void **scale_host_d = nullptr;

  void *alpha;
  void *beta;

  bool inplace = false;
  bool use_scaling = false;
  mblas_cuda_operation transA;
  mblas_cuda_operation transB;

  mblas_cuda_data_type precision;
  mblas_cuda_compute_type compute;
  mblas_cuda_data_type scalar;
  mblas_cuda_data_type a_type;
  mblas_cuda_data_type b_type;
  mblas_cuda_data_type c_type;
  mblas_cuda_data_type d_type;

  mblas_cuda_data_type a_scale_type;
  mblas_cuda_data_type b_scale_type;
  mblas_cuda_data_type c_scale_type;
  mblas_cuda_data_type d_scale_type;
  mblas_cuda_data_type bias_type;

  scale_size a_scale_size;
  scale_size b_scale_size;
  scale_size c_scale_size;
  scale_size d_scale_size;

#if (ENABLE_CUDA_FP4)
  cublasLtMatmulMatrixScale_t a_scale_mode;
  cublasLtMatmulMatrixScale_t b_scale_mode;
  cublasLtMatmulMatrixScale_t c_scale_mode;
  cublasLtMatmulMatrixScale_t d_scale_mode;
#endif
  uint64_t a_offset_host;
  uint64_t b_offset_host;
  uint64_t c_offset_host;
  uint64_t d_offset_host;

  uint64_t a_offset_dev;
  uint64_t b_offset_dev;
  uint64_t c_offset_dev;
  uint64_t d_offset_dev;

  uint64_t total_block_size_host;
  uint64_t total_block_size_dev;

  uint64_t workspace_size = 64 * 1024 * 1024;

#if defined(HAS_CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT)
    // Emulated FP64 configuration
    bool use_f64_emulation = false;
    int32_t emulation_strategy = 0;      // CUBLAS_EMULATION_STRATEGY_DEFAULT
    int32_t emulation_mantissa_control = 0; // CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC
    int32_t emulation_max_mantissa_bits = 0;
    int32_t emulation_mantissa_bit_offset = 0;
    int32_t emulation_special_values_support = 0xFFFF; // CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT
#endif

  static std::vector<matmul_prec_type> matmul_supported;
  std::vector<cublaslt_gemm_inst> mat_ptrs;

 private:
  // cudaDataType_t precisionStringToDType(std::string stringPrecision);
  // void parse_problem_type(std::string a, std::string b, std::string c);
  void parse_problem_type(std::string computeTStr, std::string scalarTStr,
                  std::string aStr, std::string bStr, std::string cStr,
                  std::string dStr);
  void validate_parameters();
  void parse_dev_iters(std::string);
  void alloc_host();
  void alloc_dev(cublaslt_gemm_inst *);
  void fill_host();
  void copy_host_to_dev(cublaslt_gemm_inst *);
  void prepare_matrix(cublaslt_gemm_inst *);
  void no_tuning(cublaslt_gemm_inst *);
  void auto_tuning(cublaslt_gemm_inst *);
  void run_threaded(void (cublaslt_gemm::*func)(cublaslt_gemm_inst *));
  std::tuple<double, double, double> calculate_figure_of_merit(double total_time_ms);
  void test_matmul(cublaslt_gemm_inst *mat);
  std::tuple<mblas_cuda_data_type, cublasLtMatmulMatrixScale_t, scale_size> configure_scaling(matrix_desc desc, mblas_cuda_data_type type, std::string matrix_id);
  //static std::tuple<mblas_cuda_data_type, cublasLtMatmulMatrixScale_t, scale_size> configure_scaling(matrix_desc desc, mblas_cuda_data_type type, std::string matrix_id);

 public:
  cublaslt_gemm(cxxopts::ParseResult result);
  std::string prepare_array();
  double test();
  std::string get_result_string();
  virtual void free_mem();
};
