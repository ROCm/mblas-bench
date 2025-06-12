#pragma once
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cxxabi.h>
#include <barrier.h>

#include <iostream>
#include <string>

#include "genericGemm.h"
#include "mblasCuDataType.h"
#include "mblasCuComputeType.h"
#include "mblasCuOperation.h"

struct matmul_prec_type {
  mblasComputeType compute;
  mblasDataType scalar;
  mblasDataType a_type;
  mblasDataType b_type;
  mblasDataType c_type;
  mblasDataType d_type;
  mblasDataType bias_type;
  bool operator==(const matmul_prec_type rhs) const {
    return compute == rhs.compute && scalar == rhs.scalar &&
           a_type == rhs.a_type && b_type == rhs.b_type &&
           c_type == rhs.c_type && d_type == rhs.d_type &&
           // Omitting bias type is acceptable
           (bias_type == rhs.bias_type ||
            rhs.bias_type == mblasDataType::MBLAS_ANY);
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
  void *scale_dev_a;
  void *scale_dev_b;
  void *scale_dev_c;
  void *scale_dev_d;
  cublasLtMatmulDesc_t desc_op;
  cublasLtMatrixLayout_t desc_a;
  cublasLtMatrixLayout_t desc_b;
  cublasLtMatrixLayout_t desc_c;
  cublasLtMatrixLayout_t desc_d;
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulHeuristicResult_t algo;
  void *devWork;
  long wSZ;
  cublaslt_gemm_inst(int devID) { devIDX = devID; }
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

class cublasLtGemm : public genericGemm {
 private:
  void *dataHost;
  void **ptr_host_a;
  void **ptr_host_b;
  void **ptr_host_c;
  void **ptr_host_d;

  void *scale_host_a;
  void *scale_host_b;
  void *scale_host_c;
  void *scale_host_d;

  void *alpha;
  void *beta;

  bool inplace = false;
  bool use_scaling = false;
  mblasCuOperation transA;
  mblasCuOperation transB;

  mblasCuDataType precision;
  mblasCuComputeType compute;
  mblasCuDataType scalar;
  mblasCuDataType a_type;
  mblasCuDataType b_type;
  mblasCuDataType c_type;
  mblasCuDataType d_type;

  mblasCuDataType a_scale_type;
  mblasCuDataType b_scale_type;
  mblasCuDataType c_scale_type;
  mblasCuDataType d_scale_type;
  mblasCuDataType bias_type;

  scale_size a_scale_size;
  scale_size b_scale_size;
  scale_size c_scale_size;
  scale_size d_scale_size;

#if (CUDART_VERSION >= 12080)
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

  int workspace_size = 64 * 1024 * 1024;

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
  void run_threaded(void (cublasLtGemm::*func)(cublaslt_gemm_inst *));
  std::tuple<double, double, double> calculate_figure_of_merit(double total_time_ms);
  void test_matmul(cublaslt_gemm_inst *mat);

 public:
  cublasLtGemm(cxxopts::ParseResult result);
  std::string prepare_array();
  double test();
  std::string get_result_string();
  virtual void free_mem();
};
