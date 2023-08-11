#pragma once
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <cxxabi.h>
#include <third_party/barrier.h>

#include <iostream>
#include <string>

#include "genericGemm.h"

struct gemmPrecType {
  hipblasComputeType_t compute;
  hipDataType scalar;
  hipDataType ab_type;
  hipDataType c_type;
  bool operator==(const gemmPrecType rhs) const {
    return rhs.compute == compute && rhs.scalar == scalar &&
           rhs.ab_type == rhs.ab_type && rhs.c_type == c_type;
  }
};
struct TgemmPrecType {
  hipDataType ab_type;
  hipDataType c_type;
  bool operator==(const TgemmPrecType rhs) const {
    return rhs.ab_type == rhs.ab_type && rhs.c_type == c_type;
  }
};

struct hipblasgemmInst {
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
  void **ptrDevA;
  void **ptrDevB;
  void **ptrDevC;
  void **ptrHostA;
  void **ptrHostB;
  void **ptrHostC;
  void *devWork;
  long wSZ;
  hipblasgemmInst(int devID) { devIDX = devID; }
};

class hipblasGemm : public genericGemm {
 private:
  void *hostA;
  void *hostB;
  void *hostC;

  // // Device array.  These are where the memory is stored on GPU
  // void *devA;
  // void *devB;
  // void *devC;

  // /*
  //   Double pointers
  //   Only used for Batched variant of gemms
  //   Unused for others
  // */
  // void **ptrDevA;
  // void **ptrDevB;
  // void **ptrDevC;
  // void **ptrHostA;
  // void **ptrHostB;
  // void **ptrHostC;

  void *alpha;
  void *beta;

  hipblasOperation_t transA;
  hipblasOperation_t transB;

  // hipblasStatus_t stat;
  // hipblasHandle_t handle;
  hipDataType precision;
  hipblasComputeType_t compute;
  hipDataType scalar;
  hipDataType a_type;
  hipDataType b_type;
  hipDataType c_type;

  int workspaceSz = 128 * 1024 * 1024;

  // std::map<std::string, hipDataType> precDType;
  // std::map<std::string, hipblasComputeType_t> computeDType;
  // std::map<hipDataType, hipblasComputeType_t> precToCompute;
  // static gemmPrecType gemmExSupported[];

  static std::vector<gemmPrecType> gemmExSupported;
  static std::vector<TgemmPrecType> TgemmExSupported;
  std::vector<hipblasgemmInst> matPtrs;
  std::vector<std::vector<hipEvent_t *> *> eventPtr;

 public:
  hipblasGemm(cxxopts::ParseResult result);
  void initPrecMap();
  // hipDataType precisionStringToDType(std::string stringPrecision);
  // void parseMType(std::string a, std::string b, std::string c);
  void parseMType(std::string computeTStr, std::string scalarTStr,
                  std::string aStr, std::string bStr, std::string cStr);
  void parseDevIters(std::string);
  hipblasOperation_t setOp(std::string);
  std::string prepareArray();
  void allocHost();
  void allocDev(hipblasgemmInst *);
  void fillHost();
  void copyHostToDev(hipblasgemmInst *);
  void runThreaded(void (hipblasGemm::*func)(hipblasgemmInst *));
  std::tuple<double, double, double> calculateFOM(double totalTime_ms);

  virtual void freeMem();

  std::string getResultString();
  double test();
  double testGemmExBatched();
  double testGemmExStridedBatched();

  // Parameter names are included in function definitions for refrence only
  template <typename T>
  void testTgemm(std::function<hipblasStatus_t(
                     hipblasHandle_t handle, hipblasOperation_t transa,
                     hipblasOperation_t transb, int m, int n, int k,
                     const T *alpha, const T *A, int lda, const T *B, int ldb,
                     const T *beta, T *C, int ldc)>
                     func,
                 hipblasgemmInst *mat);

  template <typename T>
  void testTgemmBatched(
      std::function<hipblasStatus_t(hipblasContext *, hipblasOperation_t,
                                    hipblasOperation_t, int, int, int, T const *,
                                    T const *const *, int, T const *const *, int,
                                    T const *, T *const *, int, int)>
          func,
      hipblasgemmInst *mat);

  template <typename T>
  void testTgemmStridedBatched(
      std::function<hipblasStatus_t(
          hipblasContext *, hipblasOperation_t, hipblasOperation_t, int, int, int,
          T const *, T const *, int, long long, T const *, int, long long,
          T const *, T *, int, long long, int)>
          func,
      hipblasgemmInst *mat);

  void testGemmEx(hipblasgemmInst *mat);
};
