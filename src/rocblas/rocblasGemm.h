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
  void ** ptrDevA;
  void ** ptrDevB;
  void ** ptrDevC;
  void ** ptrHostA;
  void ** ptrHostB;
  void ** ptrHostC;
  void *devWork;
  long wSZ;
  rocblasgemmInst(int devID, int nblocks) { 
    devIDX = devID;
    //devA.resize(nblocks);
    //devB.resize(nblocks);
    //devC.resize(nblocks);
    //ptrDevA.resize(nblocks);
    //ptrDevB.resize(nblocks);
    //ptrDevC.resize(nblocks);
    //ptrHostA.resize(nblocks);
    //ptrHostB.resize(nblocks);
    //ptrHostC.resize(nblocks);
  }
};

class rocblasGemm : public genericGemm {
 private:
  void * hostA;
  void * hostB;
  void * hostC;

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

  inline void * getOffsetPtr(void * mat, long long int blockstride, int rep, int nblocks, rocblas_datatype type);
  template<typename T>
  inline T * getOffsetPtrNC(T * mat, long long int blockstride, int rep); 

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

  template <typename T>
  void testTgemm(std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const*, int, T const*, int, T const*, T*, int)> func, rocblasgemmInst *mat);

  template <typename T>
  void testTgemm_batched(
      std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const* const*, int, T const* const*, int, T const*, T* const*, int, int)>
          func,
      rocblasgemmInst *mat);

  template <typename T>
  void testTgemm_strided_batched(
          std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const*, int, long, T const*, int, long, T const*, T*, int, long, int)>
          func, rocblasgemmInst *mat);


  void test_gemm_ex(rocblasgemmInst *mat);
  void test_gemm_batched_ex(rocblasgemmInst *mat);
  void test_gemm_strided_batched_ex(rocblasgemmInst *mat);
};
