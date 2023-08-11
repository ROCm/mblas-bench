#include "hipblasGemm.h"

#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>

#include <bitset>
#include <future>
#include <iomanip>
#include <numeric>
#include <regex>
#include <string>
#include <thread>

#include "hipblasConvert.h"
#include "hipblasCreateAllocate.h"
#include "hipblasDtypeUtils.h"
#include "hipError.h"
#include "third_party/cxxopts.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::move;
using std::string;
using std::thread;
using std::vector;

// clang-format off
std::vector<gemmPrecType> hipblasGemm::gemmExSupported = {
    // Compute type                 Scale Type    A/B Type      C Type
    {HIPBLAS_COMPUTE_16F,           HIP_R_16F,    HIP_R_16F,    HIP_R_16F   },
    {HIPBLAS_COMPUTE_32F,           HIP_R_32F,    HIP_R_16F,    HIP_R_16F   },
    {HIPBLAS_COMPUTE_32F,           HIP_R_32F,    HIP_R_16F,    HIP_R_32F   },
    {HIPBLAS_COMPUTE_32F,           HIP_R_32F,    HIP_R_16BF,   HIP_R_16BF  },
    {HIPBLAS_COMPUTE_32F,           HIP_R_32F,    HIP_R_16BF,   HIP_R_32F   },
    {HIPBLAS_COMPUTE_32F,           HIP_R_32F,    HIP_R_32F,    HIP_R_32F   },
    {HIPBLAS_COMPUTE_64F,           HIP_R_64F,    HIP_R_64F,    HIP_R_64F   },
    {HIPBLAS_COMPUTE_32I,           HIP_R_32I,    HIP_R_8I,     HIP_R_32I   }, 
    {HIPBLAS_COMPUTE_32F,           HIP_R_32F,    HIP_C_32F,    HIP_C_32F   },
    {HIPBLAS_COMPUTE_64F,           HIP_R_64F,    HIP_C_64F,    HIP_C_64F   },
};
// clang-format on

std::vector<TgemmPrecType> hipblasGemm::TgemmExSupported = {};

void hipblasGemm::initPrecMap() {}

void hipblasGemm::parseDevIters(std::string deviceStr) {
  // Parse device
  std::stringstream ss(deviceStr);
  while (ss.good()) {
    string deviceSStr;
    getline(ss, deviceSStr, ',');
    int devInt = stoi(deviceSStr);
    hipblasgemmInst val = hipblasgemmInst(devInt);
    matPtrs.push_back(val);
  }
}

void hipblasGemm::parseMType(string computeTStr, string scalarTStr, string aStr,
                             string bStr, string cStr) {
  compute = selectCompute(computeTStr, precision);
  scalar = selectScalar(scalarTStr, precision, compute);

  if (aStr == "" || bStr == "" || cStr == "") {
    // Precision not completely specified, default to precision
    // cerr << "Precision incorrectly specified, setting precision to "
    //         "-r/--precision"
    //      << endl;
    a_type = precision;
    b_type = precision;
    c_type = precision;
    return;
  }
  // Parse each precision
  a_type = precisionStringToDType(aStr);
  b_type = precisionStringToDType(bStr);
  c_type = precisionStringToDType(cStr);

  // Validate against supported precision table (fun)
  if (a_type != b_type) {
    string errorString = "A Type must the same as B Type";
    throw std::invalid_argument(errorString);
  }
  if (function.find("GemmEx")) {
    /*
      Possible functions:
        hipblasGemmEx
        hipblasGemmExBatched
        hipblasGemmExStridedBatched
    */
    gemmPrecType selType = {compute, scalar, a_type, c_type};
    auto result =
        std::find(begin(gemmExSupported), end(gemmExSupported), selType);
    if (result == end(gemmExSupported)) {
      // Unable to find matching config, not supported
      string errorString =
          "Invalid GEMM specification for GemmEx.  Combination of parameters "
          "not supported"
          "\nCompute type: " +
          computeTStr + "\nScalar type: " + scalarTStr + "\nA type: " + aStr +
          "\nB type: " + bStr + "\nC type: " + cStr;
      throw std::invalid_argument(errorString);
    }
  } else if (function.find("gemmEx")) {
    TgemmPrecType selType = {a_type, c_type};
    auto result =
        std::find(begin(TgemmExSupported), end(TgemmExSupported), selType);
    if (result == end(TgemmExSupported)) {
      // Unable to find matching config, not supported
      string errorString =
          "Invalid GEMM specification for GemmEx.  Combination of parameters "
          "not supported"
          "\nA type: " +
          aStr + "\nB type: " + bStr + "\nC type: " + cStr;
      throw std::invalid_argument(errorString);
    }
  }
}

hipblasGemm::hipblasGemm(cxxopts::ParseResult result) : genericGemm(result) {
  // hipblasCreate(&handle);
  // checkHipblas(hipblasCreate(&handle));
  initPrecMap();
  // Grab precision from command line
  precision = precisionStringToDType(result["precision"].as<string>());
  // Grab compute type from command line
  string computeT = result["compute_type"].as<string>();
  string scalarT = result["scalar_type"].as<string>();
  string aT = result["a_type"].as<string>();
  string bT = result["b_type"].as<string>();
  string cT = result["c_type"].as<string>();
  parseMType(computeT, scalarT, aT, bT, cT);

  parseDevIters(result["device"].as<string>());
  std::string tA = result["transposeA"].as<std::string>();
  std::string tB = result["transposeB"].as<std::string>();
  transA = opStringToOp(result["transposeA"].as<std::string>());
  transB = opStringToOp(result["transposeB"].as<std::string>());

  // Pull in alpha and beta, alloc memory and save to pointers
  string salpha = result["alpha"].as<string>();
  string salphai = result["alphai"].as<string>();
  alpha =
      typeCallHost<allocSetScalar>(precision, salpha.c_str(), salphai.c_str());
  string sbeta = result["beta"].as<string>();
  string sbetai = result["betai"].as<string>();
  beta = typeCallHost<allocSetScalar>(precision, sbeta.c_str(), sbetai.c_str());
  // std::cout << *((float *)alpha) << std::endl;
  // std::cout << *((float *)beta) << std::endl;
}

string hipblasGemm::prepareArray() {
  // std::cout << "Pre Convert: " << *((float *)alpha) << std::endl;
  // alpha = convertScalar(scalar, alpha);
  // std::cout << "Post Convert: " << __half2float(*(__half *)alpha) <<
  // std::endl; beta = convertScalar(scalar, beta);
  this->allocHost();
  this->fillHost();

  int num_devices;
  hipGetDeviceCount(&num_devices);
  // Check range of devices here
  // This implementation may not work if
  // CUDA_VISIBLE_DEVICES is set to something weird
  for (auto &instance : matPtrs) {
    if (instance.devIDX >= num_devices) {
      string errorString =
          "Invalid device id"
          "\nNumber of detected devices: " +
          std::to_string(num_devices) +
          "\nDevice selection:           " + std::to_string(instance.devIDX);
      throw std::invalid_argument(errorString);
    }
  }
  // for (auto &instance : matPtrs) {
  //  this->allocDev(&instance);
  //  this->copyHostToDev(&instance);
  //}

  runThreaded(&hipblasGemm::allocDev);
  runThreaded(&hipblasGemm::copyHostToDev);
  std::ostringstream ossHeader;
  ossHeader << "transA_option,transB_option,M,N,K,lda,ldb,ldc,";
  if (batched) {
    ossHeader << "batch_count,";
  }
  ossHeader << "hipBLAS-Gflops,hipBLAS-GB/s,hipBLAS-us," << endl;
  return ossHeader.str();
}

void hipblasGemm::runThreaded(void (hipblasGemm::*func)(hipblasgemmInst *)) {
  vector<thread> threads;
  for (auto &instance : matPtrs) {
    threads.push_back(thread(func, this, &instance));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void hipblasGemm::allocHost() {
  // auto resultA = std::async(allocateHostArr, a_type, m, k, batchct);
  // auto resultB = std::async(allocateHostArr, b_type, k, n, batchct);
  // auto resultC = std::async(allocateHostArr, c_type, n, m, batchct);
  // hostA = resultA.get();
  // hostB = resultB.get();
  // hostC = resultC.get();
  hostA = allocateHostArr(a_type, m, k, batchct);
  hostB = allocateHostArr(b_type, k, n, batchct);
  hostC = allocateHostArr(c_type, m, n, batchct);
}

void hipblasGemm::allocDev(hipblasgemmInst *mat) {
  hipSetDevice(mat->devIDX);
  mat->devA = allocateDevArr(a_type, m, k, batchct);
  mat->devB = allocateDevArr(b_type, k, n, batchct);
  mat->devC = allocateDevArr(c_type, m, n, batchct);
  mat->wSZ = workspaceSz;
  hipMalloc(&mat->devWork, mat->wSZ);
}

void hipblasGemm::fillHost() {
  // Some random functions treat the matrix as a vectors, some require a matrix
  // vector<thread> threads;
  // threads.push_back(thread(initHostH, a_type, initialization, hostA, m, k,
  // lda,
  //                         batchct, stride_a, 2.f, false));
  // threads.push_back(thread(initHostH, b_type, initialization, hostB, k, n,
  // ldb,
  //                         batchct, stride_b, 3.f, true));
  // threads.push_back(thread(initHostH, c_type, initialization, hostC, m, n,
  // ldc,
  //                         batchct, stride_c, 0.f, false));
  // for (auto &thread : threads) {
  //  thread.join();
  //}
  typeCallHost<initHost>(a_type, initialization, hostA, rowsA, colsA, lda,
                         batchct, stride_a, controlA, constantA);
  typeCallHost<initHost>(b_type, initialization, hostB, rowsB, colsB, ldb,
                         batchct, stride_b, controlB, constantB);
  typeCallHost<initHost>(c_type, initialization, hostC, rowsC, colsC, ldc,
                         batchct, stride_c, controlC, constantC);
}

void hipblasGemm::copyHostToDev(hipblasgemmInst *mat) {
  hipSetDevice(mat->devIDX);
  copyAndConvert(a_type, hostA, mat->devA, m, k, batchct);
  copyAndConvert(b_type, hostB, mat->devB, k, n, batchct);
  copyAndConvert(c_type, hostC, mat->devC, n, m, batchct);
  if (batched && !strided) {
    // Perform some pointer arithmetic to calculate the arrays we pass to the
    // gpu
    mat->ptrHostA =
        (void **)malloc(batchct * typeCallHost<sizeofCUDTP>(a_type));
    mat->ptrHostB =
        (void **)malloc(batchct * typeCallHost<sizeofCUDTP>(b_type));
    mat->ptrHostC =
        (void **)malloc(batchct * typeCallHost<sizeofCUDTP>(c_type));
    checkHip(
        hipMalloc(&mat->ptrDevA, batchct * typeCallHost<sizeofCUDTP>(a_type)));
    checkHip(
        hipMalloc(&mat->ptrDevB, batchct * typeCallHost<sizeofCUDTP>(b_type)));
    checkHip(
        hipMalloc(&mat->ptrDevC, batchct * typeCallHost<sizeofCUDTP>(c_type)));
    typeCallDev<batchedPtrMagic>(a_type, mat->ptrHostA, mat->ptrDevA, mat->devA,
                                 batchct, m, k);
    typeCallDev<batchedPtrMagic>(b_type, mat->ptrHostB, mat->ptrDevB, mat->devB,
                                 batchct, k, n);
    typeCallDev<batchedPtrMagic>(c_type, mat->ptrHostC, mat->ptrDevC, mat->devC,
                                 batchct, n, m);
  }
}

void hipblasGemm::freeMem() {
  free(alpha);
  free(beta);
  free(hostA);
  free(hostB);
  free(hostC);
  for (auto mat : matPtrs) {
    hipFree(mat.devA);
    hipFree(mat.devB);
    hipFree(mat.devC);
    hipFree(mat.devWork);
    if (batched && !strided) {
      free(mat.ptrHostA);
      free(mat.ptrHostB);
      free(mat.ptrHostC);
      hipFree(mat.ptrDevA);
      hipFree(mat.ptrDevB);
      hipFree(mat.ptrDevC);
    }
  }
}

double hipblasGemm::test() {
  vector<thread> threads;
  double gflops = 0.0;
  for (auto &mat : matPtrs) {
    // Tgemm
    if ((function == "hipblasDgemm" && precision == HIP_R_64F) ||
        (function == "gemm" && precision == HIP_R_64F)) {
      std::function<decltype(hipblasDgemm)> dgemm_var = hipblasDgemm;
      threads.push_back(
          thread(&hipblasGemm::testTgemm<double>, this, dgemm_var, &mat));
    } else if ((function == "hipblasSgemm" && precision == HIP_R_32F) ||
               (function == "gemm" && precision == HIP_R_32F)) {
      std::function<decltype(hipblasSgemm)> sgemm_var = hipblasSgemm;
      threads.push_back(
          thread(&hipblasGemm::testTgemm<float>, this, sgemm_var, &mat));
    } else if ((function == "hipblasHgemm" && precision == HIP_R_16F) ||
               (function == "gemm" && precision == HIP_R_16F)) {
      std::function<decltype(hipblasHgemm)> hgemm_var = hipblasHgemm;
      threads.push_back(
          thread(&hipblasGemm::testTgemm<__half>, this, hgemm_var, &mat));
    } else if ((function == "hipblasZgemm" && precision == HIP_C_64F) ||
               (function == "gemm" && precision == HIP_C_64F)) {
      std::function<decltype(hipblasZgemm)> zgemm_var = hipblasZgemm;
      threads.push_back(thread(&hipblasGemm::testTgemm<hipDoubleComplex>, this,
                               zgemm_var, &mat));
    } else if ((function == "hipblasCgemm" && precision == HIP_C_32F) ||
               (function == "gemm" && precision == HIP_C_32F)) {
      std::function<decltype(hipblasCgemm)> cgemm_var = hipblasCgemm;
      threads.push_back(
          thread(&hipblasGemm::testTgemm<hipComplex>, this, cgemm_var, &mat));
    }
    // TgemmBatched
    else if (function == "hipblasDgemmBatched" && precision == HIP_R_64F) {
      std::function<decltype(hipblasDgemmBatched)> dgemm_var =
          hipblasDgemmBatched;
      threads.push_back(
          thread(&hipblasGemm::testTgemmBatched<double>, this, dgemm_var, &mat));
    } else if (function == "hipblasSgemmBatched" && precision == HIP_R_32F) {
      std::function<decltype(hipblasSgemmBatched)> sgemm_var =
          hipblasSgemmBatched;
      threads.push_back(
          thread(&hipblasGemm::testTgemmBatched<float>, this, sgemm_var, &mat));
    } else if (function == "hipblasHgemmBatched" && precision == HIP_R_16F) {
      std::function<decltype(hipblasHgemmBatched)> hgemm_var =
          hipblasHgemmBatched;
      threads.push_back(
          thread(&hipblasGemm::testTgemmBatched<__half>, this, hgemm_var, &mat));
    } else if (function == "hipblasZgemmBatched" && precision == HIP_C_64F) {
      std::function<decltype(hipblasZgemmBatched)> zgemm_var =
          hipblasZgemmBatched;
      threads.push_back(thread(&hipblasGemm::testTgemmBatched<hipDoubleComplex>,
                               this, zgemm_var, &mat));
    } else if (function == "hipblasCgemmBatched" && precision == HIP_C_32F) {
      std::function<decltype(hipblasCgemmBatched)> cgemm_var =
          hipblasCgemmBatched;
      threads.push_back(thread(&hipblasGemm::testTgemmBatched<hipComplex>, this,
                               cgemm_var, &mat));
    }
    // TgemmStridedBatched
    else if (function == "hipblasDgemmStridedBatched" &&
             precision == HIP_R_64F) {
      std::function<decltype(hipblasDgemmStridedBatched)> dgemm_var =
          hipblasDgemmStridedBatched;
      threads.push_back(thread(&hipblasGemm::testTgemmStridedBatched<double>,
                               this, dgemm_var, &mat));
    } else if (function == "hipblasSgemmStridedBatched" &&
               precision == HIP_R_32F) {
      std::function<decltype(hipblasSgemmStridedBatched)> sgemm_var =
          hipblasSgemmStridedBatched;
      threads.push_back(thread(&hipblasGemm::testTgemmStridedBatched<float>,
                               this, sgemm_var, &mat));
    } else if (function == "hipblasHgemmStridedBatched" &&
               precision == HIP_R_16F) {
      std::function<decltype(hipblasHgemmStridedBatched)> hgemm_var =
          hipblasHgemmStridedBatched;
      threads.push_back(thread(&hipblasGemm::testTgemmStridedBatched<__half>,
                               this, hgemm_var, &mat));
    } else if (function == "hipblasZgemmStridedBatched" &&
               precision == HIP_C_64F) {
      std::function<decltype(hipblasZgemmStridedBatched)> zgemm_var =
          hipblasZgemmStridedBatched;
      threads.push_back(
          thread(&hipblasGemm::testTgemmStridedBatched<hipDoubleComplex>, this,
                 zgemm_var, &mat));
    } else if (function == "hipblasCgemmStridedBatched" &&
               precision == HIP_C_32F) {
      std::function<decltype(hipblasCgemmStridedBatched)> cgemm_var =
          hipblasCgemmStridedBatched;
      threads.push_back(thread(&hipblasGemm::testTgemmStridedBatched<hipComplex>,
                               this, cgemm_var, &mat));
    }
    // gemmEx
    else if (strided && function == "hipblasGemmExStridedBatched") {
      // Call the Gemm strided batched deployment script
    } else if (batched && function == "hipblasGemmExBatched") {
      // Call the Gemm batched code
    } else if (function == "hipblasGemmEx" || function == "gemm_ex") {
      threads.push_back(thread(&hipblasGemm::testGemmEx, this, &mat));
    }
  }
  // Wait on running jobs
  for (auto &thread : threads) {
    thread.join();
  }

  // Sum all gflops
  gflop_per_second = std::accumulate(
      begin(matPtrs), end(matPtrs), 0.0,
      [](double i, const hipblasgemmInst &o) { return o.gflops + i; });

  gbyte_per_second = std::accumulate(
      begin(matPtrs), end(matPtrs), 0.0,
      [](double i, const hipblasgemmInst &o) { return o.gbytes + i; });

  iter_time_us = std::accumulate(begin(matPtrs), end(matPtrs), 0.0,
                                 [](double i, const hipblasgemmInst &o) {
                                   return o.time_us + i;
                                 }) /
                 matPtrs.size();

  return gflop_per_second;
}

std::string hipblasGemm::getResultString() {
  std::ostringstream ossValues;
  ossValues << std::setprecision(7);
  ossValues << opToString(transA) << ',' << opToString(transB) << ',' << m
            << ',' << n << ',' << k << ',' << lda << ',' << ldb << ',' << ldc
            << ',';
  if (batched) {
    ossValues << batchct << ',';
  }
  ossValues << gflop_per_second << ',';
  ossValues << gbyte_per_second << ',';
  ossValues << iter_time_us << ',';
  ossValues << endl;
  return ossValues.str();
}

std::tuple<double, double, double> hipblasGemm::calculateFOM(
    double totalTime_ms) {
  double avgTime_ms = totalTime_ms / iters;
  double avgTime_s = avgTime_ms / 1000.0f;
  double avgTime_us = avgTime_ms * 1000.0f;

  int a_sz = typeCallDev<sizeofCUDT>(a_type);
  int b_sz = typeCallDev<sizeofCUDT>(b_type);
  int c_sz = typeCallDev<sizeofCUDT>(c_type);

  int flopPerSize = 2;
  if (!isReal(precision)) {
    int flopPerSize = 8;
  }
  double gbytes = ((static_cast<double>(a_sz) * static_cast<double>(m) *
                    static_cast<double>(k)) +
                   (static_cast<double>(b_sz) * static_cast<double>(k) *
                    static_cast<double>(n)) +
                   (static_cast<double>(c_sz) * static_cast<double>(n) *
                    static_cast<double>(m))) /
                  1e9;
  double gflops = static_cast<double>(flopPerSize) *
                  (static_cast<double>(m) * static_cast<double>(n) *
                   static_cast<double>(k)) /
                  1e9;

  double gflopPerSec = gflops * static_cast<double>(batchct) / avgTime_s;
  double gbytePerSec = gbytes * batchct / avgTime_s;

  return std::tuple<double, double, double>(gflopPerSec, gbytePerSec,
                                            avgTime_us);
}

template <typename T>
void hipblasGemm::testTgemm(
    std::function<hipblasStatus_t(
        hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
        const T *, const T *, int, const T *, int, const T *, T *, int)>
        func,
    hipblasgemmInst *mat) {
  hipblasStatus_t stat;
  hipblasHandle_t handle;
  hipStream_t stream;
  checkHip(hipSetDevice(mat->devIDX));
  checkHipblas(hipblasCreate(&handle));
  checkHip(hipStreamCreate(&stream));
  checkHipblas(hipblasSetStream(handle, stream));
  // checkHipblas(hipblasSetWorkspace(handle, mat->devWork, mat->wSZ));

  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T *devAP = static_cast<T *>(mat->devA);
  T *devBP = static_cast<T *>(mat->devB);
  T *devCP = static_cast<T *>(mat->devC);

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = func(handle, transA, transB, m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc);

    // Check for errors during the gemm run
    checkHipblas(stat);
    checkHip(hipGetLastError());
  }
  hipStreamSynchronize(stream);

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  /*
    Run and time the performance test
  */
  hipEventRecord(start, stream);
  for (int rep = 0; rep < iters; rep++) {
    stat = func(handle, transA, transB, m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc);
  }
  hipEventRecord(stop, stream);
  hipEventSynchronize(stop);

  // Check for errors during the performance test
  checkHipblas(stat);
  checkHip(hipGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  hipEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculateFOM(static_cast<double>(elapsedTime_ms));
}

template <typename T>
void hipblasGemm::testTgemmBatched(
    std::function<hipblasStatus_t(hipblasContext *, hipblasOperation_t,
                                 hipblasOperation_t, int, int, int, T const *,
                                 T const *const *, int, T const *const *, int,
                                 T const *, T *const *, int, int)>
        func,
    hipblasgemmInst *mat) {
  hipblasStatus_t stat;
  hipblasHandle_t handle;
  hipStream_t stream;
  checkHip(hipSetDevice(mat->devIDX));
  checkHipblas(hipblasCreate(&handle));
  checkHip(hipStreamCreate(&stream));
  checkHipblas(hipblasSetStream(handle, stream));
  // checkHipblas(hipblasSetWorkspace(handle, mat->devWork, mat->wSZ));

  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T **devAP = reinterpret_cast<T **>(mat->ptrDevA);
  T **devBP = reinterpret_cast<T **>(mat->ptrDevB);
  T **devCP = reinterpret_cast<T **>(mat->ptrDevC);

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = func(handle, transA, transB, m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc, batchct);

    // Check for errors during the gemm run
    checkHipblas(stat);
    checkHip(hipGetLastError());
  }
  hipStreamSynchronize(stream);

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  /*
    Run and time the performance test
  */
  hipEventRecord(start, stream);
  for (int rep = 0; rep < iters; rep++) {
    stat = func(handle, transA, transB, m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc, batchct);
  }
  hipEventRecord(stop, stream);
  hipEventSynchronize(stop);

  // Check for errors during the performance test
  checkHipblas(stat);
  checkHip(hipGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  hipEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculateFOM(static_cast<double>(elapsedTime_ms));
}

template <typename T>
void hipblasGemm::testTgemmStridedBatched(
    std::function<hipblasStatus_t(
        hipblasContext *, hipblasOperation_t, hipblasOperation_t, int, int, int,
        T const *, T const *, int, long long, T const *, int, long long,
        T const *, T *, int, long long, int)>
        func,
    hipblasgemmInst *mat) {
  hipblasStatus_t stat;
  hipblasHandle_t handle;
  hipStream_t stream;
  checkHip(hipSetDevice(mat->devIDX));
  checkHipblas(hipblasCreate(&handle));
  checkHip(hipStreamCreate(&stream));
  checkHipblas(hipblasSetStream(handle, stream));
  // checkHipblas(hipblasSetWorkspace(handle, mat->devWork, mat->wSZ));

  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T *devAP = static_cast<T *>(mat->devA);
  T *devBP = static_cast<T *>(mat->devB);
  T *devCP = static_cast<T *>(mat->devC);

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = func(handle, transA, transB, m, n, k, alphaP, devAP, lda, stride_a,
                devBP, ldb, stride_b, betaP, devCP, ldc, stride_c, batchct);

    // Check for errors during the gemm run
    checkHipblas(stat);
    checkHip(hipGetLastError());
  }
  hipStreamSynchronize(stream);

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  /*
    Run and time the performance test
  */
  hipEventRecord(start, stream);
  for (int rep = 0; rep < iters; rep++) {
    stat = func(handle, transA, transB, m, n, k, alphaP, devAP, lda, stride_a,
                devBP, ldb, stride_b, betaP, devCP, ldc, stride_c, batchct);
  }
  hipEventRecord(stop, stream);
  hipEventSynchronize(stop);

  // Check for errors during the performance test
  checkHipblas(stat);
  checkHip(hipGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  hipEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculateFOM(static_cast<double>(elapsedTime_ms));
}

void hipblasGemm::testGemmEx(hipblasgemmInst *mat) {
  hipblasStatus_t stat;
  hipblasHandle_t handle;
  hipStream_t stream;
  checkHip(hipSetDevice(mat->devIDX));
  checkHipblas(hipblasCreate(&handle));
  checkHip(hipStreamCreate(&stream));
  checkHipblas(hipblasSetStream(handle, stream));
  checkHipblas(hipblasSetWorkspace(handle, mat->devWork, mat->wSZ));
  // cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH); // HIP_UNSUPPORTED
  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = hipblasGemmEx_v2(handle, transA, transB, m, n, k, alpha, mat->devA,
                            a_type, lda, mat->devB, b_type, ldb, beta, mat->devC,
                            c_type, ldc, compute, HIPBLAS_GEMM_DEFAULT);

    // Check for errors during the gemm run
    checkHipblas(stat);
    checkHip(hipGetLastError());
  }
  hipStreamSynchronize(stream);

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  /*
    Run and time the performance test
  */
  hipEventRecord(start, stream);
  for (int rep = 0; rep < iters; rep++) {
    stat = hipblasGemmEx_v2(handle, transA, transB, m, n, k, alpha, mat->devA,
                         a_type, lda, mat->devB, b_type, ldb, beta, mat->devC,
                         c_type, ldc, compute, HIPBLAS_GEMM_DEFAULT);
  }
  hipEventRecord(stop, stream);
  hipEventSynchronize(stop);

  // Check for errors during the performance test
  checkHipblas(stat);
  checkHip(hipGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  hipEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculateFOM(static_cast<double>(elapsedTime_ms));
}
