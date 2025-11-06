#pragma once
#include <rocblas/rocblas.h>
//#include <hip/hip_runtime.h>
#include <cxxabi.h>

#include <iostream>
#include <vector>
#include <string>

#include "generic_gemm.h"
#include "mblas_rocblas_data_type.h"
#include "mblas_rocblas_compute_type.h"
#include "mblas_rocblas_operation.h"

struct gemmPrecTypeAMD {
  mblas_rocblas_compute_type compute;
  mblas_rocblas_data_type scalar;
  mblas_rocblas_data_type ab_type;
  mblas_rocblas_data_type c_type;
  bool operator==(const gemmPrecTypeAMD rhs) const {
    return rhs.compute == compute && rhs.scalar == scalar &&
           rhs.ab_type == ab_type && rhs.c_type == c_type;
  }
};
struct TgemmPrecTypeAMD {
  mblas_rocblas_data_type ab_type;
  mblas_rocblas_data_type c_type;
  bool operator==(const TgemmPrecTypeAMD rhs) const {
    return rhs.ab_type == ab_type && rhs.c_type == c_type;
  }
};

struct rocblas_gemm_inst {
  int devIDX;
  double gflops = 0;
  double gbytes = 0;
  double time_us = 0;
  void *alpha;
  void *beta;
  /*
    Double pointers
    Only used for Batched variant of gemms
    Unused for others
  */
  void ** ptr_dev_a;
  void ** ptr_dev_b;
  void ** ptr_dev_c;
  void ** ptr_dev_d;
  void *devWork;
  long wSZ;
  rocblas_gemm_inst(int devID) { 
    devIDX = devID;
  }
};

class rocblas_gemm : public generic_gemm {
 private:
  void **ptr_host_a;
  void **ptr_host_b;
  void **ptr_host_c;
  void **ptr_host_d;

  // // Device array.  These are where the memory is stored on GPU
  // void *devA;
  // void *devB;
  // void *devC;

  // /*
  //   Double pointers
  //   Only used for Batched variant of gemms
  //   Unused for others
  // */
  // void **ptr_dev_a;
  // void **ptr_dev_b;
  // void **ptr_dev_c;
  // void **ptr_host_a;
  // void **ptr_host_b;
  // void **ptr_host_c;

  void *alpha;
  void *beta;

  bool inplace = false;

  mblas_rocblas_operation transA;
  mblas_rocblas_operation transB;

  // rocblas_status stat;
  // rocblas_handle handle;
  mblas_rocblas_data_type precision;
  mblas_rocblas_compute_type compute;
  mblas_rocblas_data_type scalar;
  mblas_rocblas_data_type a_type;
  mblas_rocblas_data_type b_type;
  mblas_rocblas_data_type c_type;
  mblas_rocblas_data_type d_type;

  int workspace_size = 128 * 1024 * 1024;

  // std::map<std::string, rocblas_datatype> precDType;
  // std::map<std::string, rocblas_datatype> computeDType;
  // std::map<rocblas_datatype, rocblas_datatype> precToCompute;
  // static gemmPrecTypeAMD gemm_ex_supported[];

  static std::vector<gemmPrecTypeAMD> gemm_ex_supported;
  static std::vector<TgemmPrecTypeAMD> Tgemm_ex_supported;
  std::vector<rocblas_gemm_inst> mat_ptrs;
  std::vector<std::vector<hipEvent_t *> *> eventPtr;

  void init_prec_map();
  // rocblas_datatype precisionStringToRocblasDType(std::string stringPrecision);
  // void parse_problem_type(std::string a, std::string b, std::string c);
  void parse_problem_type(std::string computeTStr, std::string scalarTStr,
                  std::string aStr, std::string bStr, std::string cStr, std::string dStr);
  void parse_dev_iters(std::string);
  rocblas_operation set_op(std::string);
  void alloc_host();
  void alloc_dev(rocblas_gemm_inst *);
  void fill_host();
  void copy_host_to_dev(rocblas_gemm_inst *);
  void run_threaded(void (rocblas_gemm::*func)(rocblas_gemm_inst *));
  std::tuple<double, double, double> calculate_figure_of_merit(double totalTime_ms);



  template <typename T>
  void test_Tgemm(std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const*, int, T const*, int, T const*, T*, int)> func, rocblas_gemm_inst *mat);

  template <typename T>
  void test_Tgemm_batched(
      std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const* const*, int, T const* const*, int, T const*, T* const*, int, int)>
          func,
      rocblas_gemm_inst *mat);

  template <typename T>
  void test_Tgemm_strided_batched(
          std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const*, int, long, T const*, int, long, T const*, T*, int, long, int)>
          func, rocblas_gemm_inst *mat);


  void test_gemm_ex(rocblas_gemm_inst *mat);
  void test_gemm_batched_ex(rocblas_gemm_inst *mat);
  void test_gemm_strided_batched_ex(rocblas_gemm_inst *mat);

 public:
  rocblas_gemm(cxxopts::ParseResult result);
  std::string prepare_array();
  double test(const int &ith_solution);
  std::string get_result_string();
  virtual void free_mem();

};
