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
  // Compute type                 Scale Type    A Type        B Type        C Type        D Type        Bias Type
  {MBLAS_COMPUTE_32F,             MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F},
  {MBLAS_COMPUTE_32F_FAST_TF32,   MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F},
  {MBLAS_COMPUTE_32F,             MBLAS_R_32F,  MBLAS_R_16F,  MBLAS_R_16F,  MBLAS_R_16F,  MBLAS_R_16F,  MBLAS_R_16F},
  {MBLAS_COMPUTE_32F,             MBLAS_R_32F,  MBLAS_R_16F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F},
  {MBLAS_COMPUTE_32F,             MBLAS_R_32F,  MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32I,             MBLAS_R_32I,  MBLAS_R_8I,   MBLAS_R_8I,   MBLAS_R_8I,   MBLAS_R_8I,   MBLAS_ANY},
};

std::vector<matmulPrecTypeF8> hipblasLtGemm::matmulSupportedF8 = {
  // Scale Type  C Type           D Type            Bias Type
  {MBLAS_R_32F,  MBLAS_R_16F,     MBLAS_R_16F,      MBLAS_R_16F },
  {MBLAS_R_32F,  MBLAS_R_16BF,    MBLAS_R_16BF,     MBLAS_R_16BF},
  {MBLAS_R_32F,  MBLAS_R_32F,     MBLAS_R_32F,      MBLAS_R_16BF},
  {MBLAS_R_32F,  MBLAS_R_8F_E4M3, MBLAS_R_8F_E4M3,  MBLAS_R_16F },
  {MBLAS_R_32F,  MBLAS_R_8F_E5M2, MBLAS_R_8F_E5M2,  MBLAS_R_16F },
  // FP32 bias variants
  // Scale Type  C Type           D Type            Bias Type
  {MBLAS_R_32F,  MBLAS_R_16F,     MBLAS_R_16F,      MBLAS_R_32F },
  {MBLAS_R_32F,  MBLAS_R_16BF,    MBLAS_R_16BF,     MBLAS_R_32F },
  {MBLAS_R_32F,  MBLAS_R_32F,     MBLAS_R_32F,      MBLAS_R_32F },
  {MBLAS_R_32F,  MBLAS_R_8F_E4M3, MBLAS_R_8F_E4M3,  MBLAS_R_32F },
  {MBLAS_R_32F,  MBLAS_R_8F_E5M2, MBLAS_R_8F_E5M2,  MBLAS_R_32F },
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
  if (result != end(matmulSupported)) {
    return;
  } else if (compute == mblasHipComputeType::MBLAS_COMPUTE_32F && a_type.isFp8() && b_type.isFp8()) {
    // Special FP8 type filtering
    matmulPrecTypeF8 selTypeF8 = {
        scalar, c_type, d_type, mblasHipDataType(MBLAS_ANY)};
    auto result = std::find(begin(matmulSupportedF8), end(matmulSupportedF8), selTypeF8);
    if (result != end(matmulSupportedF8)) {
      return;
    }
  }
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
  uint64_t a_offset, b_offset, c_offset, d_offset;
  set_flush_batch_count(a_offset, b_offset, c_offset, d_offset, 
      typeCallDev<sizeofCUDT>(a_type), typeCallDev<sizeofCUDT>(b_type), 
      typeCallDev<sizeofCUDT>(c_type), typeCallDev<sizeofCUDT>(d_type), 
      get_packing_count(a_type), 
      get_packing_count(b_type), 
      get_packing_count(c_type), 
      get_packing_count(d_type), 
      inplace);
}

string hipblasLtGemm::prepareArray() {
  alpha = convertScalar(scalar, alpha);
  beta = convertScalar(scalar, beta);
  this->alloc_host();
  this->fill_host();

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
  //  this->alloc_dev(&instance);
  //  this->copyHostToDev(&instance);
  //}
  runThreaded(&hipblasLtGemm::alloc_dev);
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

void hipblasLtGemm::alloc_host() {
  // auto resultA = std::async(allocateHostArr, a_type, m, k, batch_count);
  // auto resultB = std::async(allocateHostArr, b_type, k, n, batch_count);
  // auto resultC = std::async(allocateHostArr, c_type, n, m, batch_count);
  // hostA = resultA.get();
  // hostB = resultB.get();
  // hostC = resultC.get();
  ptr_host_a =
      (void **)malloc(flush_batch_count * typeCallHost<sizeofCUDTP>(a_type));
  ptr_host_b =
      (void **)malloc(flush_batch_count * typeCallHost<sizeofCUDTP>(b_type));
  ptr_host_c =
      (void **)malloc(flush_batch_count * typeCallHost<sizeofCUDTP>(c_type));
  if (!inplace) {
    ptr_host_d =
      (void **)malloc(flush_batch_count * typeCallHost<sizeofCUDTP>(d_type));
  } else {
    ptr_host_d = ptr_host_c;
  }


  for (int i = 0; i < flush_batch_count; i++) {
    ptr_host_a[i] = allocateHostArr(a_type, rows_mem_a, cols_mem_a, batch_count);
    ptr_host_b[i] = allocateHostArr(b_type, rows_mem_b, cols_mem_b, batch_count);
    ptr_host_c[i] = allocateHostArr(c_type, rows_mem_c, cols_mem_c, batch_count);
    if (!inplace) {
      ptr_host_d[i] = allocateHostArr(d_type, rows_mem_d, cols_mem_d, batch_count);
    }
  }
  //hostA = allocateHostArr(a_type, m, k, batch_count);
  //hostB = allocateHostArr(b_type, k, n, batch_count);
  //hostC = allocateHostArr(c_type, m, n, batch_count);
}

void hipblasLtGemm::alloc_dev(hipblasLtGemmInst *mat) {
  hipSetDevice(mat->devIDX);

  mat->ptrDevA =
      (void **)malloc(batch_count * flush_batch_count * typeCallDev<sizeofCUDTP>(a_type));
  mat->ptrDevB =
      (void **)malloc(batch_count * flush_batch_count * typeCallDev<sizeofCUDTP>(b_type));
  mat->ptrDevC =
      (void **)malloc(batch_count * flush_batch_count * typeCallDev<sizeofCUDTP>(c_type));
  if (!inplace) {
    mat->ptrDevD =
        (void **)malloc(batch_count * flush_batch_count * typeCallDev<sizeofCUDTP>(d_type));
  } else {
    mat->ptrDevD = mat->ptrDevC;
  }

  for (int i = 0; i < flush_batch_count; i++) {
    mat->ptrDevA[i] = allocateDevArr(a_type, rows_mem_a, cols_mem_a, batch_count);
    mat->ptrDevB[i] = allocateDevArr(b_type, rows_mem_b, cols_mem_b, batch_count);
    mat->ptrDevC[i] = allocateDevArr(c_type, rows_mem_c, cols_mem_c, batch_count);
    if (!inplace) {
      mat->ptrDevD[i] = allocateDevArr(d_type, rows_mem_d, cols_mem_d, batch_count);
    }
  }
  mat->wSZ = workspaceSz;
  hipMalloc(&mat->devWork, mat->wSZ);
}

void hipblasLtGemm::fill_host() {
  // Some random functions treat the matrix as a vectors, some require a matrix
  // vector<thread> threads;
  // threads.push_back(thread(initHostH, a_type, initialization, hostA, m, k,
  // lda,
  //                         batch_count, stride_a, 2.f, false));
  // threads.push_back(thread(initHostH, b_type, initialization, hostB, k, n,
  // ldb,
  //                         batch_count, stride_b, 3.f, true));
  // threads.push_back(thread(initHostH, c_type, initialization, hostC, m, n,
  // ldc,
  //                         batch_count, stride_c, 1.f, false));
  // for (auto &thread : threads) {
  //  thread.join();
  //}

  for (int i = 0; i < flush_batch_count; i++){
    typeCallHost<initHost>(a_type, initialization, ptr_host_a[i], rows_a, cols_a, lda,
                           batch_count, stride_a, controlA, constantA, filenameA);
    typeCallHost<initHost>(b_type, initialization, ptr_host_b[i], rows_b, cols_b, ldb,
                           batch_count, stride_b, controlB, constantB, filenameB);
    typeCallHost<initHost>(c_type, initialization, ptr_host_c[i], rows_c, cols_c, ldc,
                           batch_count, stride_c, controlC, constantC, filenameC);
  }
}

void hipblasLtGemm::copyHostToDev(hipblasLtGemmInst *mat) {
  hipSetDevice(mat->devIDX);
  for (int i = 0; i < flush_batch_count; i++) {
    copyAndConvert(a_type, ptr_host_a[i], mat->ptrDevA[i], m, k, batch_count);
    copyAndConvert(b_type, ptr_host_b[i], mat->ptrDevB[i], k, n, batch_count);
    copyAndConvert(c_type, ptr_host_c[i], mat->ptrDevC[i], n, m, batch_count);
  }
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
      hipblasLtMatrixLayoutCreate(&mat->descA, a_type, rows_a, cols_a, lda));
  checkHipblas(
      hipblasLtMatrixLayoutCreate(&mat->descB, b_type, rows_b, cols_b, ldb));
  checkHipblas(
      hipblasLtMatrixLayoutCreate(&mat->descC, c_type, rows_c, cols_c, ldc));
  if (!inplace) {
    checkHipblas(
        hipblasLtMatrixLayoutCreate(&mat->descD, d_type, rows_d, cold_d, ldd));
  } else {
    mat->descD = mat->descC;
  }
  if (batch_count > 1) {
    checkHipblas(hipblasLtMatrixLayoutSetAttribute(mat->descA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    checkHipblas(hipblasLtMatrixLayoutSetAttribute(mat->descB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    checkHipblas(hipblasLtMatrixLayoutSetAttribute(mat->descC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    checkHipblas(hipblasLtMatrixLayoutSetAttribute(mat->descD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

    checkHipblas(hipblasLtMatrixLayoutSetAttribute(mat->descA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
    checkHipblas(hipblasLtMatrixLayoutSetAttribute(mat->descB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
    checkHipblas(hipblasLtMatrixLayoutSetAttribute(mat->descC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));
    checkHipblas(hipblasLtMatrixLayoutSetAttribute(mat->descD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_d, sizeof(stride_d)));
  }

  checkHipblas(hipblasLtMatmulPreferenceCreate(&mat->pref));
  checkHipblas(hipblasLtMatmulPreferenceSetAttribute(
      mat->pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &mat->wSZ,
      sizeof(mat->wSZ)));
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
  //free(hostA);
  //free(hostB);
  //free(hostC);
  for (auto mat : matPtrs) {
    hipFree(mat.ptrDevA);
    hipFree(mat.ptrDevB);
    hipFree(mat.ptrDevC);
    hipFree(mat.ptrDevD);
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
    ossValues << batch_count << ',';
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

  double gflopPerSec = gflops * static_cast<double>(batch_count) / avgTime_s;
  double gbytePerSec = gbytes * batch_count / avgTime_s;

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
    int flush_index = rep % flush_batch_count;
    stat = hipblasLtMatmul(handle, mat->descOP, alpha, mat->ptrDevA[flush_index], mat->descA,
                          mat->ptrDevB[flush_index], mat->descB, beta, mat->ptrDevC[flush_index], mat->descC,
                          mat->ptrDevD[flush_index], mat->descD, &mat->algo.algo, mat->devWork,
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
    int flush_index = rep % flush_batch_count;
    stat = hipblasLtMatmul(handle, mat->descOP, alpha, mat->ptrDevA[flush_index], mat->descA,
                          mat->ptrDevB[flush_index], mat->descB, beta, mat->ptrDevC[flush_index], mat->descC,
                          mat->ptrDevD[flush_index], mat->descD, &mat->algo.algo, mat->devWork,
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