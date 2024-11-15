#include "hipblasLtGemm.h"

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

#include <future>
#include <iomanip>
#include <numeric>
#include <regex>
#include <string>
#include <thread>

#include "rocblasConvert.h"
#include "rocblasCreateAllocate.h"
#include "rocblasDtypeUtils.h"
#include "rocblasError.h"
#include "cxxopts.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::move;
using std::string;
using std::thread;
using std::vector;

// clang-format off
std::vector<matmulPrecType> hipblasLtGemm::matmulSupported = {
  // Compute type           Scale Type      A Type          B Type          C Type          D Type          Bias Type
  {MBLAS_COMPUTE_32F,   MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F},
  {MBLAS_COMPUTE_32F,   MBLAS_R_32F,  MBLAS_R_16F,  MBLAS_R_16F,  MBLAS_R_16F,  MBLAS_R_16F,  MBLAS_R_16F},
  {MBLAS_COMPUTE_32F,   MBLAS_R_32F,  MBLAS_R_16BF,  MBLAS_R_16BF,  MBLAS_R_16BF,  MBLAS_R_16BF,  MBLAS_R_16BF},
};
// clang-format on

void hipblasLtGemm::parseDevIters(std::string deviceStr) {
  // Parse device
  std::stringstream ss(deviceStr);
  while (ss.good()) {
    string deviceSStr;
    getline(ss, deviceSStr, ',');
    int devInt = stoi(deviceSStr);
    hipblasLtGemmInst val = hipblasLtGemmInst(devInt);
    matPtrs.push_back(val);
  }
}

void hipblasLtGemm::parseMType(string computeTStr, string scalarTStr,
                               string aStr, string bStr, string cStr,
                               string dStr) {
  compute.setCompute(computeTStr, precision);
  scalar.setScalar(scalarTStr, precision, compute);
  bool noParse = false;
  if (aStr == "" || bStr == "" || cStr == "") {
    // Precision not completely specified, default to precision
    // cerr << "Precision incorrectly specified, setting precision to "
    //         "-r/--precision"
    //      << endl;
    a_type = precision;
    b_type = precision;
    c_type = precision;
    d_type = precision;
    inplace = true;
    noParse = true;
  }

  if (dStr == "") {
    // Assume the user means C = D, so also establish that here
    dStr = cStr;
    inplace = true;
  }

  // Parse each precision
  if (!noParse) {
    a_type = mblasHipDataType(aStr);
    b_type = mblasHipDataType(bStr);
    c_type = mblasHipDataType(cStr);
    d_type = mblasHipDataType(dStr);
  }
}

void hipblasLtGemm::validateParameters() {
  // Validate that data types exist in table of supported configurations
  matmulPrecType selType = {
      compute, scalar, a_type, b_type, c_type, d_type, mblasHipDataType(MBLAS_ANY)};
  auto result =
      std::find(begin(matmulSupported), end(matmulSupported), selType);
  if (result == end(matmulSupported)) {
    // Unable to find matching config, not supported
    string errorString =
        "Invalid GEMM specification for MatMul.  Combination of parameters "
        "not supported"
        "\nCompute type: " +
        compute.toString() + "\nScalar type: " + scalar.toString() +
        "\nA type: " + a_type.toString() +
        "\nB type: " + b_type.toString() +
        "\nC type: " + c_type.toString() +
        "\nD type: " + d_type.toString();
    throw std::invalid_argument(errorString);
  }
  // Validate that FP8 kernels will use TN format only
  // GEMM fails if not
  // if ((isFp8(a_type) || isFp8(b_type) || isFp8(c_type) || isFp8(d_type)) &&
  //     (transA != CUBLAS_OP_T || transB != CUBLAS_OP_N)) {
  //   string errorString =
  //       "Transpose operation selection not supported"
  //       "\nOnly TN format is supported"
  //       "\nTransA: " +
  //       opToString(transA) + "\nTransB: " + opToString(transB);
  //   throw std::invalid_argument(errorString);
  // }
}

hipblasLtGemm::hipblasLtGemm(cxxopts::ParseResult result) : genericGemm(result) {
  // Grab precision from command line
  precision = mblasHipDataType(result["precision"].as<string>());
  // Grab compute type from command line
  string computeT = result["compute_type"].as<string>();
  string scalarT = result["scalar_type"].as<string>();
  string aT = result["a_type"].as<string>();
  string bT = result["b_type"].as<string>();
  string cT = result["c_type"].as<string>();
  string dT = result["d_type"].as<string>();
  parseMType(computeT, scalarT, aT, bT, cT, dT);

  parseDevIters(result["device"].as<string>());
  std::string tA = result["transposeA"].as<std::string>();
  std::string tB = result["transposeB"].as<std::string>();
  transA = mblasHipOperation(result["transposeA"].as<std::string>());
  transB = mblasHipOperation(result["transposeB"].as<std::string>());
  validateParameters();

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

string hipblasLtGemm::prepareArray() {
  alpha = convertScalar(scalar, alpha);
  beta = convertScalar(scalar, beta);
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
  runThreaded(&hipblasLtGemm::allocDev);
  runThreaded(&hipblasLtGemm::copyHostToDev);
  runThreaded(&hipblasLtGemm::prepareMatrix);
  // Enable tuning with a parameter later
  if (false) {
  } else {
    runThreaded(&hipblasLtGemm::noTuning);
  }
  std::ostringstream ossHeader;
  ossHeader << "transA_option,transB_option,M,N,K,lda,ldb,ldc,";
  if (batched) {
    ossHeader << "batch_count,";
  }
  ossHeader << "hipBLASLt-Gflops,hipBLASLt-GB/s,hipBLASLt-us," << endl;
  return ossHeader.str();
}

void hipblasLtGemm::runThreaded(void (hipblasLtGemm::*func)(hipblasLtGemmInst *)) {
  vector<thread> threads;
  for (auto &instance : matPtrs) {
    threads.push_back(thread(func, this, &instance));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void hipblasLtGemm::allocHost() {
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

void hipblasLtGemm::allocDev(hipblasLtGemmInst *mat) {
  hipSetDevice(mat->devIDX);
  mat->devA = allocateDevArr(a_type, m, k, batchct);
  mat->devB = allocateDevArr(b_type, k, n, batchct);
  mat->devC = allocateDevArr(c_type, m, n, batchct);
  if (!inplace) {
    mat->devD = allocateDevArr(d_type, n, m, batchct);
  } else {
    mat->devD = mat->devC;
  }
  mat->wSZ = workspaceSz;
  hipMalloc(&mat->devWork, mat->wSZ);
}

void hipblasLtGemm::fillHost() {
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
  //                         batchct, stride_c, 1.f, false));
  // for (auto &thread : threads) {
  //  thread.join();
  //}

  typeCallHost<initHost>(a_type, initialization, hostA, rowsA, colsA, lda,
                         batchct, stride_a, controlA, constantA, filenameA);
  typeCallHost<initHost>(b_type, initialization, hostB, rowsB, colsB, ldb,
                         batchct, stride_b, controlB, constantB, filenameB);
  typeCallHost<initHost>(c_type, initialization, hostC, rowsC, colsC, ldc,
                         batchct, stride_c, controlC, constantC, filenameC);
}

void hipblasLtGemm::copyHostToDev(hipblasLtGemmInst *mat) {
  hipSetDevice(mat->devIDX);
  copyAndConvert(a_type, hostA, mat->devA, m, k, batchct);
  copyAndConvert(b_type, hostB, mat->devB, k, n, batchct);
  copyAndConvert(c_type, hostC, mat->devC, n, m, batchct);
}

void hipblasLtGemm::prepareMatrix(hipblasLtGemmInst *mat) {
  checkHipblas(hipblasLtMatmulDescCreate(&mat->descOP, compute, scalar));
  // These values are read in with no type, so they need to be convirted first
  // Thanks for the wonderful standard Nvidia :D!
  hipblasOperation_t transA_local = transA.convertToHip();
  hipblasOperation_t transB_local = transB.convertToHip();
  checkHipblas(hipblasLtMatmulDescSetAttribute(
      mat->descOP, HIPBLASLT_MATMUL_DESC_TRANSA, &transA_local, sizeof(transA)));
  checkHipblas(hipblasLtMatmulDescSetAttribute(
      mat->descOP, HIPBLASLT_MATMUL_DESC_TRANSB, &transB_local, sizeof(transB)));

  checkHipblas(
      hipblasLtMatrixLayoutCreate(&mat->descA, a_type, rowsA, colsA, lda));
  checkHipblas(
      hipblasLtMatrixLayoutCreate(&mat->descB, b_type, rowsB, colsB, ldb));
  checkHipblas(
      hipblasLtMatrixLayoutCreate(&mat->descC, c_type, rowsC, colsC, ldc));
  if (!inplace) {
    checkHipblas(
        hipblasLtMatrixLayoutCreate(&mat->descD, d_type, rowsD, colsD, ldd));
  } else {
    mat->descD = mat->descC;
  }

  checkHipblas(hipblasLtMatmulPreferenceCreate(&mat->pref));
  checkHipblas(hipblasLtMatmulPreferenceSetAttribute(
      mat->pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &mat->wSZ,
      sizeof(mat->wSZ)));
  // if (isFp8(a_type) || isFp8(b_type) || isFp8(c_type) || isFp8(d_type)) {
  //   // Default is 0, enable for faster fp8 results
  //   int8_t fastAccuMode = 1;
  //   hipblasLtMatmulDescSetAttribute(mat->descOP, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
  //                                  &fastAccuMode, sizeof(fastAccuMode));
  // }
}

void hipblasLtGemm::noTuning(hipblasLtGemmInst *mat) {
  hipblasStatus_t stat;
  hipblasLtHandle_t handle;
  checkHip(hipSetDevice(mat->devIDX));
  checkHipblas(hipblasLtCreate(&handle));
  int retResults = 0;
  hipblasLtMatmulHeuristicResult_t heuristicResult = {0};

  checkHipblas(hipblasLtMatmulAlgoGetHeuristic(
      handle, mat->descOP, mat->descA, mat->descB, mat->descC, mat->descD,
      mat->pref, 1, &heuristicResult, &retResults));

  if (retResults == 0) {
    checkHipblas(HIPBLAS_STATUS_NOT_SUPPORTED);
  }
  mat->algo = heuristicResult;
}
void hipblasLtGemm::autoTuning(hipblasLtGemmInst *mat) {
  // Not currently implemented, using simple method
  noTuning(mat);
}

void hipblasLtGemm::freeMem() {
  free(alpha);
  free(beta);
  free(hostA);
  free(hostB);
  free(hostC);
  for (auto mat : matPtrs) {
    hipFree(mat.devA);
    hipFree(mat.devB);
    hipFree(mat.devC);
  }
}

double hipblasLtGemm::test() {
  vector<thread> threads;
  double gflops = 0.0;
  for (auto &mat : matPtrs) {
    threads.push_back(thread(&hipblasLtGemm::testMatmul, this, &mat));
  }
  // Wait on running jobs
  for (auto &thread : threads) {
    thread.join();
  }

  // Sum all gflops
  gflop_per_second = std::accumulate(
      begin(matPtrs), end(matPtrs), 0.0,
      [](double i, const hipblasLtGemmInst &o) { return o.gflops + i; });

  gbyte_per_second = std::accumulate(
      begin(matPtrs), end(matPtrs), 0.0,
      [](double i, const hipblasLtGemmInst &o) { return o.gbytes + i; });

  iter_time_us = std::accumulate(begin(matPtrs), end(matPtrs), 0.0,
                                 [](double i, const hipblasLtGemmInst &o) {
                                   return o.time_us + i;
                                 }) /
                 matPtrs.size();

  return gflop_per_second;
}

std::string hipblasLtGemm::getResultString() {
  std::ostringstream ossValues;
  ossValues << std::setprecision(7);
  ossValues << transA.toStringShort() << ',' << transB.toStringShort() << ',' << m
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

std::tuple<double, double, double> hipblasLtGemm::calculateFOM(
    double totalTime_ms) {
  double avgTime_ms = totalTime_ms / iters;
  double avgTime_s = avgTime_ms / 1000.0f;
  double avgTime_us = avgTime_ms * 1000.0f;

  int a_sz = typeCallDev<sizeofCUDT>(a_type);
  int b_sz = typeCallDev<sizeofCUDT>(b_type);
  int c_sz = typeCallDev<sizeofCUDT>(c_type);

  int flopPerSize = 2;
  if (!precision.isReal()) {
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

void hipblasLtGemm::testMatmul(hipblasLtGemmInst *mat) {
  hipblasStatus_t stat;
  hipblasLtHandle_t handle;
  hipStream_t stream;
  checkHip(hipSetDevice(mat->devIDX));
  checkHipblas(hipblasLtCreate(&handle));
  checkHip(hipStreamCreate(&stream));
  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = hipblasLtMatmul(handle, mat->descOP, alpha, mat->devA, mat->descA,
                          mat->devB, mat->descB, beta, mat->devC, mat->descC,
                          mat->devD, mat->descD, &mat->algo.algo, mat->devWork,
                          mat->wSZ, stream);
    // Check for errors during the gemm run
    checkHipblas(stat);
    checkHip(hipGetLastError());
  }
  hipStreamSynchronize(stream);

  hipEvent_t start, stop;
  checkHip(hipEventCreate(&start));
  checkHip(hipEventCreate(&stop));

  /*
    Run and time the performance test
  */
  hipEventRecord(start, stream);
  for (int rep = 0; rep < iters; rep++) {
    stat = hipblasLtMatmul(handle, mat->descOP, alpha, mat->devA, mat->descA,
                          mat->devB, mat->descB, beta, mat->devC, mat->descC,
                          mat->devD, mat->descD, &mat->algo.algo, mat->devWork,
                          mat->wSZ, stream);
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