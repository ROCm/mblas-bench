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

struct matmulPrecType {
  mblasComputeType compute;
  mblasDataType scalar;
  mblasDataType a_type;
  mblasDataType b_type;
  mblasDataType c_type;
  mblasDataType d_type;
  mblasDataType bias_type;
  bool operator==(const matmulPrecType rhs) const {
    return compute == rhs.compute && scalar == rhs.scalar &&
           a_type == rhs.a_type && b_type == rhs.b_type &&
           c_type == rhs.c_type && d_type == rhs.d_type &&
           // Omitting bias type is acceptable
           (bias_type == rhs.bias_type ||
            rhs.bias_type == mblasDataType::MBLAS_ANY);
  }
};

struct cublasltgemmInst {
  int devIDX;
  double gflops = 0;
  double gbytes = 0;
  double time_us = 0;
  void *dataDev;
  //void **devA;
  //void **devB;
  //void **devC;
  //void **devD;
  void **ptrDevA;
  void **ptrDevB;
  void **ptrDevC;
  void **ptrDevD;
  cublasLtMatmulDesc_t descOP;
  cublasLtMatrixLayout_t descA;
  cublasLtMatrixLayout_t descB;
  cublasLtMatrixLayout_t descC;
  cublasLtMatrixLayout_t descD;
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulHeuristicResult_t algo;
  void *devWork;
  long wSZ;
  cublasltgemmInst(int devID) { devIDX = devID; }
};

class cublasLtGemm : public genericGemm {
 private:
  void *dataHost;
  void **ptrHostA;
  void **ptrHostB;
  void **ptrHostC;
  void **ptrHostD;

  void *alpha;
  void *beta;

  bool inplace = false;
  mblasCuOperation transA;
  mblasCuOperation transB;

  mblasCuDataType precision;
  mblasCuComputeType compute;
  mblasCuDataType scalar;
  mblasCuDataType a_type;
  mblasCuDataType b_type;
  mblasCuDataType c_type;
  mblasCuDataType d_type;
  mblasCuDataType bias_type;

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

  int workspaceSz = 64 * 1024 * 1024;

  static std::vector<matmulPrecType> matmulSupported;
  std::vector<cublasltgemmInst> matPtrs;

 private:
  // cudaDataType_t precisionStringToDType(std::string stringPrecision);
  // void parseMType(std::string a, std::string b, std::string c);
  void parseMType(std::string computeTStr, std::string scalarTStr,
                  std::string aStr, std::string bStr, std::string cStr,
                  std::string dStr);
  void validateParameters();
  void parseDevIters(std::string);
  void allocHost();
  void allocDev(cublasltgemmInst *);
  void fillHost();
  void copyHostToDev(cublasltgemmInst *);
  void prepareMatrix(cublasltgemmInst *);
  void noTuning(cublasltgemmInst *);
  void autoTuning(cublasltgemmInst *);
  void runThreaded(void (cublasLtGemm::*func)(cublasltgemmInst *));
  std::tuple<double, double, double> calculateFOM(double totalTime_ms);
  void testMatmul(cublasltgemmInst *mat);

 public:
  cublasLtGemm(cxxopts::ParseResult result);
  std::string prepareArray();
  double test();
  std::string getResultString();
  virtual void freeMem();
};
