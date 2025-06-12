#pragma once
#include <hipblaslt/hipblaslt.h>
// #include <hip/hip_runtime.h>
#include <cxxabi.h>

#include <iostream>
#include <string>

#include "genericGemm.h"
#include "mblasHipDataType.h"
#include "mblasHipComputeType.h"
#include "mblasHipOperation.h"

struct matmul_prec_type {
  mblasHipComputeType compute;
  mblasHipDataType scalar;
  mblasHipDataType a_type;
  mblasHipDataType b_type;
  mblasHipDataType c_type;
  mblasHipDataType d_type;
  mblasHipDataType bias_type;
  bool operator==(const matmul_prec_type rhs) const {
    return rhs.compute == compute && rhs.scalar == scalar &&
           rhs.a_type == a_type && rhs.b_type == b_type &&
           rhs.c_type == c_type && rhs.d_type == d_type &&
           // Omitting bias type is acceptable
           (rhs.bias_type == bias_type ||
            rhs.bias_type == mblasHipDataType(MBLAS_ANY));
  }
};

struct matmulPrecTypeF8 {
  mblasHipDataType scalar;
  mblasHipDataType c_type;
  mblasHipDataType d_type;
  mblasHipDataType bias_type;
  bool operator==(const matmulPrecTypeF8 rhs) const {
    return rhs.scalar == scalar &&
           rhs.c_type == c_type && rhs.d_type == d_type &&
           // Omitting bias type is acceptable
           (rhs.bias_type == bias_type ||
            rhs.bias_type == mblasHipDataType(MBLAS_ANY));
  }
};

struct hipblasLtGemmInst {
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
  hipblasLtMatmulDesc_t desc_op;
  hipblasLtMatrixLayout_t desc_a;
  hipblasLtMatrixLayout_t desc_b;
  hipblasLtMatrixLayout_t desc_c;
  hipblasLtMatrixLayout_t desc_d;
  hipblasLtMatmulPreference_t pref;
  hipblasLtMatmulHeuristicResult_t algo;
  void *devWork;
  long wSZ;
  hipblasLtGemmInst(int devID) { devIDX = devID; }
};

class hipblasLtGemm : public genericGemm {
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
  mblasHipOperation transA;
  mblasHipOperation transB;

  mblasHipDataType precision;
  mblasHipComputeType compute;
  mblasHipDataType scalar;
  mblasHipDataType a_type;
  mblasHipDataType b_type;
  mblasHipDataType c_type;
  mblasHipDataType d_type;
  mblasHipDataType bias_type;

  int workspace_size = 64 * 1024 * 1024;

  static std::vector<matmul_prec_type> matmul_supported;
  static std::vector<matmulPrecTypeF8> matmulSupportedF8;
  std::vector<hipblasLtGemmInst> mat_ptrs;

 private:
  // mblasHipDataType precisionStringToHipblasDType(std::string stringPrecision);
  // void parse_problem_type(std::string a, std::string b, std::string c);
  void parse_problem_type(std::string computeTStr, std::string scalarTStr,
                  std::string aStr, std::string bStr, std::string cStr,
                  std::string dStr);
  void validate_parameters();
  void parse_dev_iters(std::string);
  void alloc_host();
  void alloc_dev(hipblasLtGemmInst *);
  void fill_host();
  void copy_host_to_dev(hipblasLtGemmInst *);
  void prepare_matrix(hipblasLtGemmInst *);
  void no_tuning(hipblasLtGemmInst *);
  void auto_tuning(hipblasLtGemmInst *);
  void run_threaded(void (hipblasLtGemm::*func)(hipblasLtGemmInst *));
  std::tuple<double, double, double> calculate_figure_of_merit(double totalTime_ms);
  void test_matmul(hipblasLtGemmInst *mat);

 public:
  hipblasLtGemm(cxxopts::ParseResult result);
  std::string prepare_array();
  double test();
  std::string get_result_string();
  virtual void free_mem();
};