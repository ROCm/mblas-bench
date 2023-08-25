#pragma once
#include <rocblas/rocblas.h>
//#include <hip/hip_runtime.h>
#include <cxxabi.h>

#include <iostream>
#include <vector>
#include <string>

#include "genericGemm.h"

struct gemmPrecTypeAMD {
  rocblas_datatype compute;
  rocblas_datatype scalar;
  rocblas_datatype ab_type;
  rocblas_datatype c_type;
  bool operator==(const gemmPrecTypeAMD rhs) const {
    return rhs.compute == compute && rhs.scalar == scalar &&
           rhs.ab_type == ab_type && rhs.c_type == c_type;
  }
};
struct TgemmPrecTypeAMD {
  rocblas_datatype ab_type;
  rocblas_datatype c_type;
  bool operator==(const TgemmPrecTypeAMD rhs) const {
    return rhs.ab_type == ab_type && rhs.c_type == c_type;
  }
};

struct rocblasgemmInst {
  int devIDX;
  double gflops = 0;
  double gbytes = 0;
  double time_us = 0;
  std::vector<void *> devA;
  std::vector<void *> devB;
  std::vector<void *> devC;
  void *alpha;
  void *beta;
  /*
    Double pointers
    Only used for Batched variant of gemms
    Unused for others
  */
  std::vector<void **> ptrDevA;
  std::vector<void **> ptrDevB;
  std::vector<void **> ptrDevC;
  std::vector<void **> ptrHostA;
  std::vector<void **> ptrHostB;
  std::vector<void **> ptrHostC;
  void *devWork;
  long wSZ;
  rocblasgemmInst(int devID, int nblocks) { 
    devIDX = devID;
    devA.resize(nblocks);
    devB.resize(nblocks);
    devC.resize(nblocks);
    ptrDevA.resize(nblocks);
    ptrDevB.resize(nblocks);
    ptrDevC.resize(nblocks);
    ptrHostA.resize(nblocks);
    ptrHostB.resize(nblocks);
    ptrHostC.resize(nblocks);
  }
};

class rocblasGemm : public genericGemm {
 private:
  std::vector<void *> hostA;
  std::vector<void *> hostB;
  std::vector<void *> hostC;

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

  rocblas_operation transA;
  rocblas_operation transB;

  // rocblas_status stat;
  // rocblas_handle handle;
  rocblas_datatype precision;
  rocblas_datatype compute;
  rocblas_datatype scalar;
  rocblas_datatype a_type;
  rocblas_datatype b_type;
  rocblas_datatype c_type;

  int workspaceSz = 128 * 1024 * 1024;

  // std::map<std::string, rocblas_datatype> precDType;
  // std::map<std::string, rocblas_datatype> computeDType;
  // std::map<rocblas_datatype, rocblas_datatype> precToCompute;
  // static gemmPrecTypeAMD gemmExSupported[];

  static std::vector<gemmPrecTypeAMD> gemmExSupported;
  static std::vector<TgemmPrecTypeAMD> TgemmExSupported;
  std::vector<rocblasgemmInst> matPtrs;
  std::vector<std::vector<hipEvent_t *> *> eventPtr;

 public:
  rocblasGemm(cxxopts::ParseResult result);
  void initPrecMap();
  // rocblas_datatype precisionStringToDType(std::string stringPrecision);
  // void parseMType(std::string a, std::string b, std::string c);
  void parseMType(std::string computeTStr, std::string scalarTStr,
                  std::string aStr, std::string bStr, std::string cStr);
  void parseDevIters(std::string);
  rocblas_operation setOp(std::string);
  std::string prepareArray();
  void allocHost();
  void allocDev(rocblasgemmInst *);
  void fillHost();
  void copyHostToDev(rocblasgemmInst *);
  void runThreaded(void (rocblasGemm::*func)(rocblasgemmInst *));
  std::tuple<double, double, double> calculateFOM(double totalTime_ms);

  virtual void freeMem();

  std::string getResultString();
  double test();
  double testGemmExBatched();
  double testGemmExStridedBatched();

  // Parameter names are included in function definitions for refrence only
  // template <typename T>
  // void testTgemm(std::function<rocblas_status(
  //                    rocblas_handle handle, rocblas_operation transa,
  //                    rocblas_operation transb, int m, int n, int k,
  //                    const T *alpha, const T *A, int lda, const T *B, int ldb,
  //                    const T *beta, T *C, int ldc)>
  //                    func,
  //                rocblasgemmInst *mat);

  // template <typename T>
  // void testTgemmBatched(
  //     std::function<rocblas_status(rocblas_handle, rocblas_operation,
  //                                   rocblas_operation, int, int, int, T const *,
  //                                   T const *const *, int, T const *const *, int,
  //                                   T const *, T *const *, int, int)>
  //         func,
  //     rocblasgemmInst *mat);

  // template <typename T>
  // void testTgemmStridedBatched(
  //     std::function<rocblas_status(
  //         rocblas_handle, rocblas_operation, rocblas_operation, int, int, int,
  //         T const *, T const *, int, long long, T const *, int, long long,
  //         T const *, T *, int, long long, int)>
  //         func,
  //     rocblasgemmInst *mat);

  void testGemmEx(rocblasgemmInst *mat);
};
