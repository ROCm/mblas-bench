#pragma once
#include <rocblas/rocblas.h>
//#include <hip/hip_runtime.h>
#include <cxxabi.h>

#include <iostream>
#include <vector>
#include <string>

#include "genericGemm.h"
#include "mblasRocDataType.h"
#include "mblasRocComputeType.h"
#include "mblasRocOperation.h"

struct gemmPrecTypeAMD {
  mblasRocComputeType compute;
  mblasRocDataType scalar;
  mblasRocDataType ab_type;
  mblasRocDataType c_type;
  bool operator==(const gemmPrecTypeAMD rhs) const {
    return rhs.compute == compute && rhs.scalar == scalar &&
           rhs.ab_type == ab_type && rhs.c_type == c_type;
  }
};
struct TgemmPrecTypeAMD {
  mblasRocDataType ab_type;
  mblasRocDataType c_type;
  bool operator==(const TgemmPrecTypeAMD rhs) const {
    return rhs.ab_type == ab_type && rhs.c_type == c_type;
  }
};

struct rocblasgemmInst {
  int devIDX;
  double gflops = 0;
  double gbytes = 0;
  double time_us = 0;
  void * devA;
  void * devB;
  void * devC;
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
  void ** ptr_host_a;
  void ** ptr_host_b;
  void ** ptr_host_c;
  void *devWork;
  long wSZ;
  rocblasgemmInst(int devID) { 
    devIDX = devID;
  }
};

class rocblasGemm : public genericGemm {
 private:
  void * host_a;
  void * host_b;
  void * host_c;

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

  mblasRocOperation transA;
  mblasRocOperation transB;

  // rocblas_status stat;
  // rocblas_handle handle;
  mblasRocDataType precision;
  mblasRocComputeType compute;
  mblasRocDataType scalar;
  mblasRocDataType a_type;
  mblasRocDataType b_type;
  mblasRocDataType c_type;

  int workspace_size = 128 * 1024 * 1024;

  // std::map<std::string, rocblas_datatype> precDType;
  // std::map<std::string, rocblas_datatype> computeDType;
  // std::map<rocblas_datatype, rocblas_datatype> precToCompute;
  // static gemmPrecTypeAMD gemmExSupported[];

  static std::vector<gemmPrecTypeAMD> gemmExSupported;
  static std::vector<TgemmPrecTypeAMD> TgemmExSupported;
  std::vector<rocblasgemmInst> mat_ptrs;
  std::vector<std::vector<hipEvent_t *> *> eventPtr;

  void init_prec_map();
  // rocblas_datatype precisionStringToRocblasDType(std::string stringPrecision);
  // void parse_problem_type(std::string a, std::string b, std::string c);
  void parse_problem_type(std::string computeTStr, std::string scalarTStr,
                  std::string aStr, std::string bStr, std::string cStr);
  void parse_dev_iters(std::string);
  rocblas_operation setOp(std::string);
  void alloc_host();
  void alloc_dev(rocblasgemmInst *);
  void fill_host();
  void copy_host_to_dev(rocblasgemmInst *);
  void run_threaded(void (rocblasGemm::*func)(rocblasgemmInst *));
  std::tuple<double, double, double> calculate_figure_of_merit(double totalTime_ms);



  template <typename T>
  void test_Tgemm(std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const*, int, T const*, int, T const*, T*, int)> func, rocblasgemmInst *mat);

  template <typename T>
  void test_Tgemm_batched(
      std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const* const*, int, T const* const*, int, T const*, T* const*, int, int)>
          func,
      rocblasgemmInst *mat);

  template <typename T>
  void test_Tgemm_strided_batched(
          std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const*, int, long, T const*, int, long, T const*, T*, int, long, int)>
          func, rocblasgemmInst *mat);


  void test_gemm_ex(rocblasgemmInst *mat);
  void test_gemm_batched_ex(rocblasgemmInst *mat);
  void test_gemm_strided_batched_ex(rocblasgemmInst *mat);

 public:
  rocblasGemm(cxxopts::ParseResult result);
  std::string prepare_array();
  double test();
  std::string get_result_string();
  virtual void free_mem();

};
