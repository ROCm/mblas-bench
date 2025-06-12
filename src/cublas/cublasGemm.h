#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cxxabi.h>

#include <iostream>
#include <string>

#include "genericGemm.h"
#include "mblasCuDataType.h"
#include "mblasCuComputeType.h"
#include "mblasCuOperation.h"

struct gemmPrecType {
  mblasComputeType compute;
  mblasDataType scalar;
  mblasDataType ab_type;
  mblasDataType c_type;
  bool operator==(const gemmPrecType rhs) const {
    return compute == rhs.compute && scalar == rhs.scalar &&
           ab_type == rhs.ab_type && c_type == rhs.c_type;
  }
};
struct TgemmPrecType {
  mblasDataType ab_type;
  mblasDataType c_type;
  bool operator==(const TgemmPrecType rhs) const {
    return  ab_type == rhs.ab_type &&
            c_type == rhs.c_type;
  }
};

struct cublasgemmInst {
  int devIDX;
  double gflops = 0;
  double gbytes = 0;
  double time_us = 0;
  void *devA;
  void *devB;
  void *devC;
  void *alpha;
  void *beta;
  /*
    Double pointers
    Only used for Batched variant of gemms
    Unused for others
  */
  void **ptr_dev_a;
  void **ptr_dev_b;
  void **ptr_dev_c;
  void **ptr_host_a;
  void **ptr_host_b;
  void **ptr_host_c;
  void *devWork;
  long wSZ;
  cublasgemmInst(int devID) { devIDX = devID; }
};

class cublasGemm : public genericGemm {
 private:
  void *host_a;
  void *host_b;
  void *host_c;

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

  mblasCuOperation transA;
  mblasCuOperation transB;

  // cublasStatus_t stat;
  // cublasHandle_t handle;
  mblasCuDataType precision;
  mblasCuComputeType compute;
  mblasCuDataType scalar;
  mblasCuDataType a_type;
  mblasCuDataType b_type;
  mblasCuDataType c_type;

  int workspace_size = 128 * 1024 * 1024;

  static std::vector<gemmPrecType> gemmExSupported;
  static std::vector<TgemmPrecType> TgemmExSupported;
  std::vector<cublasgemmInst> mat_ptrs;
  std::vector<std::vector<cudaEvent_t *> *> eventPtr;

 private:
  void init_prec_map();
  // cudaDataType_t precisionStringToDType(std::string stringPrecision);
  // void parse_problem_type(std::string a, std::string b, std::string c);
  void parse_problem_type(std::string computeTStr, std::string scalarTStr,
                  std::string aStr, std::string bStr, std::string cStr);
  void parse_dev_iters(std::string);
  cublasOperation_t setOp(std::string);
  void alloc_host();
  void alloc_dev(cublasgemmInst *);
  void fill_host();
  void copy_host_to_dev(cublasgemmInst *);
  void run_threaded(void (cublasGemm::*func)(cublasgemmInst *));
  std::tuple<double, double, double> calculate_figure_of_merit(double totalTime_ms);


  double testGemmExBatched();
  double testGemmExStridedBatched();

  // Parameter names are included in function definitions for refrence only
  template <typename T>
  void test_Tgemm(std::function<cublasStatus_t(
                     cublasHandle_t handle, cublasOperation_t transa,
                     cublasOperation_t transb, int m, int n, int k,
                     const T *alpha, const T *A, int lda, const T *B, int ldb,
                     const T *beta, T *C, int ldc)>
                     func,
                 cublasgemmInst *mat);

  template <typename T>
  void testTgemmBatched(
      std::function<cublasStatus_t(cublasContext *, cublasOperation_t,
                                   cublasOperation_t, int, int, int, T const *,
                                   T const *const *, int, T const *const *, int,
                                   T const *, T *const *, int, int)>
          func,
      cublasgemmInst *mat);

  template <typename T>
  void testTgemmStridedBatched(
      std::function<cublasStatus_t(
          cublasContext *, cublasOperation_t, cublasOperation_t, int, int, int,
          T const *, T const *, int, long long, T const *, int, long long,
          T const *, T *, int, long long, int)>
          func,
      cublasgemmInst *mat);

  template <typename T>
  void testTGemmEx(
      std::function<cublasStatus_t(
          cublasContext *, cublasOperation_t, cublasOperation_t, int, int, int,
          T const *, void const *, cudaDataType_t, int, void const *,
          cudaDataType_t, int, T const *, void *, cudaDataType_t, int)>
          func,
      cublasgemmInst *mat);

  void testGemmEx(cublasgemmInst *mat);
 public:
  cublasGemm(cxxopts::ParseResult result);
  std::string prepare_array();
  double test();
  std::string get_result_string();
  virtual void free_mem();
};
