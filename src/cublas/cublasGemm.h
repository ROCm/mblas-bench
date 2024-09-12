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
  void **ptrDevA;
  void **ptrDevB;
  void **ptrDevC;
  void **ptrHostA;
  void **ptrHostB;
  void **ptrHostC;
  void *devWork;
  long wSZ;
  cublasgemmInst(int devID) { devIDX = devID; }
};

class cublasGemm : public genericGemm {
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

  int workspaceSz = 128 * 1024 * 1024;

  static std::vector<gemmPrecType> gemmExSupported;
  static std::vector<TgemmPrecType> TgemmExSupported;
  std::vector<cublasgemmInst> matPtrs;
  std::vector<std::vector<cudaEvent_t *> *> eventPtr;

 private:
  void initPrecMap();
  // cudaDataType_t precisionStringToDType(std::string stringPrecision);
  // void parseMType(std::string a, std::string b, std::string c);
  void parseMType(std::string computeTStr, std::string scalarTStr,
                  std::string aStr, std::string bStr, std::string cStr);
  void parseDevIters(std::string);
  cublasOperation_t setOp(std::string);
  void allocHost();
  void allocDev(cublasgemmInst *);
  void fillHost();
  void copyHostToDev(cublasgemmInst *);
  void runThreaded(void (cublasGemm::*func)(cublasgemmInst *));
  std::tuple<double, double, double> calculateFOM(double totalTime_ms);


  double testGemmExBatched();
  double testGemmExStridedBatched();

  // Parameter names are included in function definitions for refrence only
  template <typename T>
  void testTgemm(std::function<cublasStatus_t(
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
  std::string prepareArray();
  double test();
  std::string getResultString();
  virtual void freeMem();
};
