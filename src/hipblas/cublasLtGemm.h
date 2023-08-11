#pragma once
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cxxabi.h>
#include <third_party/barrier.h>

#include <iostream>
#include <string>

#include "genericGemm.h"

struct matmulPrecType {
  cublasComputeType_t compute;
  cublasDataType_t scalar;
  cublasDataType_t a_type;
  cublasDataType_t b_type;
  cublasDataType_t c_type;
  cublasDataType_t d_type;
  cublasDataType_t bias_type;
  bool operator==(const matmulPrecType rhs) const {
    return rhs.compute == compute && rhs.scalar == scalar &&
           rhs.a_type == rhs.a_type && rhs.b_type == b_type &&
           rhs.c_type == c_type && rhs.d_type == d_type &&
           // Omitting bias type is acceptable
           (rhs.bias_type == bias_type ||
            rhs.bias_type == (cudaDataType_t)(-1));
  }
};

struct cublasltgemmInst {
  int devIDX;
  double gflops = 0;
  double gbytes = 0;
  double time_us = 0;
  void *devA;
  void *devB;
  void *devC;
  void *devD;
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
  void *hostA;
  void *hostB;
  void *hostC;

  void *alpha;
  void *beta;

  bool inplace = false;
  cublasOperation_t transA;
  cublasOperation_t transB;

  cudaDataType_t precision;
  cublasComputeType_t compute;
  cudaDataType_t scalar;
  cudaDataType_t a_type;
  cudaDataType_t b_type;
  cudaDataType_t c_type;
  cudaDataType_t d_type;
  cudaDataType_t bias_type;

  int workspaceSz = 64 * 1024 * 1024;

  static std::vector<matmulPrecType> matmulSupported;
  std::vector<cublasltgemmInst> matPtrs;

 public:
  cublasLtGemm(cxxopts::ParseResult result);
  // cudaDataType_t precisionStringToDType(std::string stringPrecision);
  // void parseMType(std::string a, std::string b, std::string c);
  void parseMType(std::string computeTStr, std::string scalarTStr,
                  std::string aStr, std::string bStr, std::string cStr,
                  std::string dStr);
  void validateParameters();
  void parseDevIters(std::string);
  std::string prepareArray();
  void allocHost();
  void allocDev(cublasltgemmInst *);
  void fillHost();
  void copyHostToDev(cublasltgemmInst *);
  void prepareMatrix(cublasltgemmInst *);
  void noTuning(cublasltgemmInst *);
  void autoTuning(cublasltgemmInst *);

  void runThreaded(void (cublasLtGemm::*func)(cublasltgemmInst *));
  std::tuple<double, double, double> calculateFOM(double totalTime_ms);

  virtual void freeMem();

  std::string getResultString();
  double test();

  void testMatmul(cublasltgemmInst *mat);
};
