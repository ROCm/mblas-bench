#include "cublasLtGemm.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <future>
#include <iomanip>
#include <numeric>
#include <regex>
#include <string>
#include <thread>

#include "cublasConvert.h"
#include "cublasCreateAllocate.h"
#include "cublasDtypeUtils.h"
#include "cudaError.h"
#include "cxxopts.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::move;
using std::string;
using std::thread;
using std::vector;

// clang-format off
std::vector<matmulPrecType> cublasLtGemm::matmulSupported = {
  // Compute type                   Scale Type    A Type            B Type            C Type        D Type            Bias Type
  {CUBLAS_COMPUTE_16F,              CUDA_R_16F,   CUDA_R_16F,       CUDA_R_16F,       CUDA_R_16F,   CUDA_R_16F,       CUDA_R_16F},
  {CUBLAS_COMPUTE_16F_PEDANTIC,     CUDA_R_16F,   CUDA_R_16F,       CUDA_R_16F,       CUDA_R_16F,   CUDA_R_16F,       CUDA_R_16F},
  {CUBLAS_COMPUTE_32I,              CUDA_R_32I,   CUDA_R_8I,        CUDA_R_8I,        CUDA_R_32I,   CUDA_R_32I,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32I_PEDANTIC,     CUDA_R_32I,   CUDA_R_8I,        CUDA_R_8I,        CUDA_R_32I,   CUDA_R_32I,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32I,              CUDA_R_32F,   CUDA_R_8I,        CUDA_R_8I,        CUDA_R_8I,    CUDA_R_8I,        (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32I_PEDANTIC,     CUDA_R_32F,   CUDA_R_8I,        CUDA_R_8I,        CUDA_R_8I,    CUDA_R_8I,        (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32F ,             CUDA_R_32F,   CUDA_R_16BF,      CUDA_R_16BF,      CUDA_R_16BF,  CUDA_R_16BF,      CUDA_R_16BF},
  {CUBLAS_COMPUTE_32F_PEDANTIC,     CUDA_R_32F,   CUDA_R_16BF,      CUDA_R_16BF,      CUDA_R_16BF,  CUDA_R_16BF,      CUDA_R_16BF},
  {CUBLAS_COMPUTE_32F ,             CUDA_R_32F,   CUDA_R_16F,       CUDA_R_16F,       CUDA_R_16F,   CUDA_R_16F,       CUDA_R_16F},
  {CUBLAS_COMPUTE_32F_PEDANTIC,     CUDA_R_32F,   CUDA_R_16F,       CUDA_R_16F,       CUDA_R_16F,   CUDA_R_16F,       CUDA_R_16F},
  {CUBLAS_COMPUTE_32F ,             CUDA_R_32F,   CUDA_R_8I,        CUDA_R_8I,        CUDA_R_32F,   CUDA_R_32F,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32F_PEDANTIC,     CUDA_R_32F,   CUDA_R_8I,        CUDA_R_8I,        CUDA_R_32F,   CUDA_R_32F,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32F ,             CUDA_R_32F,   CUDA_R_16BF,      CUDA_R_16BF,      CUDA_R_32F,   CUDA_R_32F,       CUDA_R_32F},
  {CUBLAS_COMPUTE_32F_PEDANTIC,     CUDA_R_32F,   CUDA_R_16BF,      CUDA_R_16BF,      CUDA_R_32F,   CUDA_R_32F,       CUDA_R_32F},
  {CUBLAS_COMPUTE_32F ,             CUDA_R_32F,   CUDA_R_16F,       CUDA_R_16F,       CUDA_R_32F,   CUDA_R_32F,       CUDA_R_32F},
  {CUBLAS_COMPUTE_32F_PEDANTIC,     CUDA_R_32F,   CUDA_R_16F,       CUDA_R_16F,       CUDA_R_32F,   CUDA_R_32F,       CUDA_R_32F},
  {CUBLAS_COMPUTE_32F ,             CUDA_R_32F,   CUDA_R_32F,       CUDA_R_32F,       CUDA_R_32F,   CUDA_R_32F,       CUDA_R_32F},
  {CUBLAS_COMPUTE_32F_PEDANTIC,     CUDA_R_32F,   CUDA_R_32F,       CUDA_R_32F,       CUDA_R_32F,   CUDA_R_32F,       CUDA_R_32F},
  {CUBLAS_COMPUTE_32F ,             CUDA_C_32F,   CUDA_C_8I,        CUDA_C_8I,        CUDA_C_32F,   CUDA_C_32F,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32F_PEDANTIC,     CUDA_C_32F,   CUDA_C_8I,        CUDA_C_8I,        CUDA_C_32F,   CUDA_C_32F,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32F ,             CUDA_C_32F,   CUDA_C_32F,       CUDA_C_32F,       CUDA_C_32F,   CUDA_C_32F,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32F_PEDANTIC,     CUDA_C_32F,   CUDA_C_32F,       CUDA_C_32F,       CUDA_C_32F,   CUDA_C_32F,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32F_FAST_16F,     CUDA_R_32F,   CUDA_R_32F,       CUDA_R_32F,       CUDA_R_32F,   CUDA_R_32F,       CUDA_R_32F},
  {CUBLAS_COMPUTE_32F_FAST_16BF,    CUDA_R_32F,   CUDA_R_32F,       CUDA_R_32F,       CUDA_R_32F,   CUDA_R_32F,       CUDA_R_32F},
  {CUBLAS_COMPUTE_32F_FAST_TF32,    CUDA_R_32F,   CUDA_R_32F,       CUDA_R_32F,       CUDA_R_32F,   CUDA_R_32F,       CUDA_R_32F},
  {CUBLAS_COMPUTE_32F_FAST_16F,     CUDA_C_32F,   CUDA_C_32F,       CUDA_C_32F,       CUDA_C_32F,   CUDA_C_32F,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32F_FAST_16BF,    CUDA_C_32F,   CUDA_C_32F,       CUDA_C_32F,       CUDA_C_32F,   CUDA_C_32F,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32F_FAST_TF32,    CUDA_C_32F,   CUDA_C_32F,       CUDA_C_32F,       CUDA_C_32F,   CUDA_C_32F,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_64F,              CUDA_R_64F,   CUDA_R_64F,       CUDA_R_64F,       CUDA_R_64F,   CUDA_R_64F,       CUDA_R_64F},
  {CUBLAS_COMPUTE_64F_PEDANTIC,     CUDA_R_64F,   CUDA_R_64F,       CUDA_R_64F,       CUDA_R_64F,   CUDA_R_64F,       CUDA_R_64F},
  {CUBLAS_COMPUTE_64F,              CUDA_C_64F,   CUDA_C_64F,       CUDA_C_64F,       CUDA_C_64F,   CUDA_C_64F,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_64F_PEDANTIC,     CUDA_C_64F,   CUDA_C_64F,       CUDA_C_64F,       CUDA_C_64F,   CUDA_C_64F,       (cudaDataType_t)(-1) },
  // IMMA kernels
  // Compute type                   Scale Type    A Type            B Type            C Type        D Type            Bias Type
  {CUBLAS_COMPUTE_32I,              CUDA_R_32I,   CUDA_R_8I,        CUDA_R_8I,        CUDA_R_32I,   CUDA_R_32I,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32I_PEDANTIC,     CUDA_R_32I,   CUDA_R_8I,        CUDA_R_8I,        CUDA_R_32I,   CUDA_R_32I,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32I,              CUDA_R_32F,   CUDA_R_8I,        CUDA_R_8I,        CUDA_R_8I,    CUDA_R_8I,        (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32I_PEDANTIC,     CUDA_R_32F,   CUDA_R_8I,        CUDA_R_8I,        CUDA_R_8I,    CUDA_R_8I,        (cudaDataType_t)(-1)},
  // FP8 kernels
  // Compute type                   Scale Type    A Type            B Type            C Type        D Type            Bias Type
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E4M3,   CUDA_R_8F_E4M3,   CUDA_R_16BF,  CUDA_R_16BF,      CUDA_R_16BF},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E4M3,   CUDA_R_8F_E4M3,   CUDA_R_16BF,  CUDA_R_8F_E4M3,   CUDA_R_16BF},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E4M3,   CUDA_R_8F_E4M3,   CUDA_R_16F,   CUDA_R_8F_E4M3,   CUDA_R_16F},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E4M3,   CUDA_R_8F_E4M3,   CUDA_R_16F,   CUDA_R_16F,       CUDA_R_16F},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E4M3,   CUDA_R_8F_E4M3,   CUDA_R_32F,   CUDA_R_32F,       CUDA_R_16BF},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E4M3,   CUDA_R_8F_E5M2,   CUDA_R_16BF,  CUDA_R_16BF,      CUDA_R_16BF},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E4M3,   CUDA_R_8F_E5M2,   CUDA_R_16BF,  CUDA_R_8F_E4M3,   CUDA_R_16BF},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E4M3,   CUDA_R_8F_E5M2,   CUDA_R_16BF,  CUDA_R_8F_E5M2,   CUDA_R_16BF},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E4M3,   CUDA_R_8F_E5M2,   CUDA_R_16F,   CUDA_R_8F_E4M3,   CUDA_R_16F},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E4M3,   CUDA_R_8F_E5M2,   CUDA_R_16F,   CUDA_R_8F_E5M2,   CUDA_R_16F},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E4M3,   CUDA_R_8F_E5M2,   CUDA_R_16F,   CUDA_R_16F,       CUDA_R_16F},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E4M3,   CUDA_R_8F_E5M2,   CUDA_R_32F,   CUDA_R_32F,       CUDA_R_16BF},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E5M2,   CUDA_R_8F_E4M3,   CUDA_R_16BF,  CUDA_R_16BF,      CUDA_R_16BF},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E5M2,   CUDA_R_8F_E4M3,   CUDA_R_16BF,  CUDA_R_8F_E4M3,   CUDA_R_16BF},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E5M2,   CUDA_R_8F_E4M3,   CUDA_R_16BF,  CUDA_R_8F_E5M2,   CUDA_R_16BF},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E5M2,   CUDA_R_8F_E4M3,   CUDA_R_16F,   CUDA_R_8F_E4M3,   CUDA_R_16F},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E5M2,   CUDA_R_8F_E4M3,   CUDA_R_16F,   CUDA_R_8F_E5M2,   CUDA_R_16F},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E5M2,   CUDA_R_8F_E4M3,   CUDA_R_16F,   CUDA_R_16F,       CUDA_R_16F},
  {CUBLAS_COMPUTE_32F,              CUDA_R_32F,   CUDA_R_8F_E5M2,   CUDA_R_8F_E4M3,   CUDA_R_32F,   CUDA_R_32F,       CUDA_R_16BF},
  // Mixed precision complex kernels
  // Compute type                   Scale Type    A Type            B Type            C Type        D Type            Bias Type
  {CUBLAS_COMPUTE_32F ,             CUDA_C_32F,   CUDA_C_16F,       CUDA_C_16F,       CUDA_C_16F,   CUDA_C_16F,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32F ,             CUDA_C_32F,   CUDA_C_16F,       CUDA_C_16F,       CUDA_C_32F,   CUDA_C_32F,       (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32F ,             CUDA_C_32F,   CUDA_C_16BF,      CUDA_C_16BF,      CUDA_C_16BF,  CUDA_C_16BF,      (cudaDataType_t)(-1)},
  {CUBLAS_COMPUTE_32F ,             CUDA_C_32F,   CUDA_C_16BF,      CUDA_C_16BF,      CUDA_C_32F,   CUDA_C_32F,       (cudaDataType_t)(-1)},
};
// clang-format on

void cublasLtGemm::parseDevIters(std::string deviceStr) {
  // Parse device
  std::stringstream ss(deviceStr);
  while (ss.good()) {
    string deviceSStr;
    getline(ss, deviceSStr, ',');
    int devInt = stoi(deviceSStr);
    cublasltgemmInst val = cublasltgemmInst(devInt);
    matPtrs.push_back(val);
  }
}

void cublasLtGemm::parseMType(string computeTStr, string scalarTStr,
                              string aStr, string bStr, string cStr,
                              string dStr) {
  compute = selectCompute(computeTStr, precision);
  scalar = selectScalar(scalarTStr, precision, compute);
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
    a_type = precisionStringToDType(aStr);
    b_type = precisionStringToDType(bStr);
    c_type = precisionStringToDType(cStr);
    d_type = precisionStringToDType(dStr);
  }
}

void cublasLtGemm::validateParameters() {
  // Validate that data types exist in table of supported configurations
  matmulPrecType selType = {
      compute, scalar, a_type, b_type, c_type, d_type, (cudaDataType_t)(-1)};
  auto result =
      std::find(begin(matmulSupported), end(matmulSupported), selType);
  if (result == end(matmulSupported)) {
    // Unable to find matching config, not supported
    string errorString =
        "Invalid GEMM specification for MatMul.  Combination of parameters "
        "not supported"
        "\nCompute type: " +
        computeToString(compute) + "\nScalar type: " + precToString(scalar) +
        "\nA type: " + precToString(a_type) +
        "\nB type: " + precToString(b_type) +
        "\nC type: " + precToString(c_type) +
        "\nD type: " + precToString(d_type);
    throw std::invalid_argument(errorString);
  }
  // Validate that FP8 kernels will use TN format only
  // GEMM fails if not
  if ((isFp8(a_type) || isFp8(b_type) || isFp8(c_type) || isFp8(d_type)) &&
      (transA != CUBLAS_OP_T || transB != CUBLAS_OP_N)) {
    string errorString =
        "Transpose operation selection not supported"
        "\nOnly TN format is supported"
        "\nTransA: " +
        opToString(transA) + "\nTransB: " + opToString(transB);
    throw std::invalid_argument(errorString);
  }
}

cublasLtGemm::cublasLtGemm(cxxopts::ParseResult result) : genericGemm(result) {
  // Grab precision from command line
  precision = precisionStringToDType(result["precision"].as<string>());
  // Grab compute type from command line
  string computeT = result["compute_type"].as<string>();
  string scalarT = result["scalar_type"].as<string>();
  string aT = result["a_type"].as<string>();
  string bT = result["b_type"].as<string>();
  string cT = result["c_type"].as<string>();
  string dT = result["d_type"].as<string>();
  string compcomputeT = result["composite_compute_type"].as<string>();
  if (compcomputeT == "f32") {
    // Feature from rocBLAS, set the original compute type
    computeT = "CUBLAS_COMPUTE_32F";
  }
  parseMType(computeT, scalarT, aT, bT, cT, dT);

  parseDevIters(result["device"].as<string>());
  std::string tA = result["transposeA"].as<std::string>();
  std::string tB = result["transposeB"].as<std::string>();
  transA = opStringToOp(result["transposeA"].as<std::string>());
  transB = opStringToOp(result["transposeB"].as<std::string>());
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

string cublasLtGemm::prepareArray() {
  alpha = convertScalar(scalar, alpha);
  beta = convertScalar(scalar, beta);
  this->allocHost();
  this->fillHost();

  int num_devices;
  cudaGetDeviceCount(&num_devices);
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
  runThreaded(&cublasLtGemm::allocDev);
  runThreaded(&cublasLtGemm::copyHostToDev);
  runThreaded(&cublasLtGemm::prepareMatrix);
  // Enable tuning with a parameter later
  if (false) {
  } else {
    runThreaded(&cublasLtGemm::noTuning);
  }
  std::ostringstream ossHeader;
  ossHeader << "transA_option,transB_option,M,N,K,lda,ldb,ldc,";
  if (batched) {
    ossHeader << "batch_count,";
  }
  ossHeader << "cuBLAS-Gflops,cuBLAS-GB/s,cuBLAS-us," << endl;
  return ossHeader.str();
}

void cublasLtGemm::runThreaded(void (cublasLtGemm::*func)(cublasltgemmInst *)) {
  vector<thread> threads;
  for (auto &instance : matPtrs) {
    threads.push_back(thread(func, this, &instance));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void cublasLtGemm::allocHost() {
  // auto resultA = std::async(allocateHostArr, a_type, m, k, batchct);
  // auto resultB = std::async(allocateHostArr, b_type, k, n, batchct);
  // auto resultC = std::async(allocateHostArr, c_type, n, m, batchct);
  // hostA = resultA.get();
  // hostB = resultB.get();
  // hostC = resultC.get();
  hostA = allocateHostArr(a_type, rowsMemA, colsMemA, batchct);
  hostB = allocateHostArr(b_type, rowsMemB, colsMemB, batchct);
  hostC = allocateHostArr(c_type, rowsMemC, colsMemC, batchct);
}

void cublasLtGemm::allocDev(cublasltgemmInst *mat) {
  cudaSetDevice(mat->devIDX);
  mat->devA = allocateDevArr(a_type, rowsMemA, colsMemA, batchct);
  mat->devB = allocateDevArr(b_type, rowsMemB, colsMemB, batchct);
  mat->devC = allocateDevArr(c_type, rowsMemC, colsMemC, batchct);
  if (!inplace) {
    mat->devD = allocateDevArr(d_type, rowsMemD, colsMemD, batchct);
  } else {
    mat->devD = mat->devC;
  }
  mat->wSZ = workspaceSz;
  cudaMalloc(&mat->devWork, mat->wSZ);
}

void cublasLtGemm::fillHost() {
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

void cublasLtGemm::copyHostToDev(cublasltgemmInst *mat) {
  cudaSetDevice(mat->devIDX);
  copyAndConvert(a_type, hostA, mat->devA, rowsMemA, colsMemA, batchct);
  copyAndConvert(b_type, hostB, mat->devB, rowsMemB, colsMemB, batchct);
  copyAndConvert(c_type, hostC, mat->devC, rowsMemC, colsMemC, batchct);
}

void cublasLtGemm::prepareMatrix(cublasltgemmInst *mat) {
  checkCublas(cublasLtMatmulDescCreate(&mat->descOP, compute, scalar));
  checkCublas(cublasLtMatmulDescSetAttribute(
      mat->descOP, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
  checkCublas(cublasLtMatmulDescSetAttribute(
      mat->descOP, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

  checkCublas(
      cublasLtMatrixLayoutCreate(&mat->descA, a_type, rowsA, colsA, lda));
  checkCublas(
      cublasLtMatrixLayoutCreate(&mat->descB, b_type, rowsB, colsB, ldb));
  checkCublas(
      cublasLtMatrixLayoutCreate(&mat->descC, c_type, rowsC, colsC, ldc));
  if (!inplace) {
    checkCublas(
        cublasLtMatrixLayoutCreate(&mat->descD, d_type, rowsD, colsD, ldd));
  } else {
    mat->descD = mat->descC;
  }

  checkCublas(cublasLtMatmulPreferenceCreate(&mat->pref));
  checkCublas(cublasLtMatmulPreferenceSetAttribute(
      mat->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &mat->wSZ,
      sizeof(mat->wSZ)));
  if (isFp8(a_type) || isFp8(b_type) || isFp8(c_type) || isFp8(d_type)) {
    // Default is 0, enable for faster fp8 results
    int8_t fastAccuMode = 1;
    cublasLtMatmulDescSetAttribute(mat->descOP, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                   &fastAccuMode, sizeof(fastAccuMode));
  }
}

void cublasLtGemm::noTuning(cublasltgemmInst *mat) {
  cublasStatus_t stat;
  cublasLtHandle_t handle;
  checkCuda(cudaSetDevice(mat->devIDX));
  checkCublas(cublasLtCreate(&handle));
  int retResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {0};

  checkCublas(cublasLtMatmulAlgoGetHeuristic(
      handle, mat->descOP, mat->descA, mat->descB, mat->descC, mat->descD,
      mat->pref, 1, &heuristicResult, &retResults));

  if (retResults == 0) {
    checkCublas(CUBLAS_STATUS_NOT_SUPPORTED);
  }
  mat->algo = heuristicResult;
}
void cublasLtGemm::autoTuning(cublasltgemmInst *mat) {
  // Not currently implemented, using simple method
  noTuning(mat);
}

void cublasLtGemm::freeMem() {
  free(alpha);
  free(beta);
  free(hostA);
  free(hostB);
  free(hostC);
  for (auto mat : matPtrs) {
    cudaFree(mat.devA);
    cudaFree(mat.devB);
    cudaFree(mat.devC);
  }
}

double cublasLtGemm::test() {
  vector<thread> threads;
  double gflops = 0.0;
  for (auto &mat : matPtrs) {
    threads.push_back(thread(&cublasLtGemm::testMatmul, this, &mat));
  }
  // Wait on running jobs
  for (auto &thread : threads) {
    thread.join();
  }

  // Sum all gflops
  gflop_per_second = std::accumulate(
      begin(matPtrs), end(matPtrs), 0.0,
      [](double i, const cublasltgemmInst &o) { return o.gflops + i; });

  gbyte_per_second = std::accumulate(
      begin(matPtrs), end(matPtrs), 0.0,
      [](double i, const cublasltgemmInst &o) { return o.gbytes + i; });

  iter_time_us = std::accumulate(begin(matPtrs), end(matPtrs), 0.0,
                                 [](double i, const cublasltgemmInst &o) {
                                   return o.time_us + i;
                                 }) /
                 matPtrs.size();

  return gflop_per_second;
}

std::string cublasLtGemm::getResultString() {
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

std::tuple<double, double, double> cublasLtGemm::calculateFOM(
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

void cublasLtGemm::testMatmul(cublasltgemmInst *mat) {
  cublasStatus_t stat;
  cublasLtHandle_t handle;
  cudaStream_t stream;
  checkCuda(cudaSetDevice(mat->devIDX));
  checkCublas(cublasLtCreate(&handle));
  checkCuda(cudaStreamCreate(&stream));
  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = cublasLtMatmul(handle, mat->descOP, alpha, mat->devA, mat->descA,
                          mat->devB, mat->descB, beta, mat->devC, mat->descC,
                          mat->devD, mat->descD, &mat->algo.algo, mat->devWork,
                          mat->wSZ, stream);
    // Check for errors during the gemm run
    checkCublas(stat);
    checkCuda(cudaGetLastError());
  }
  cudaStreamSynchronize(stream);

  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));

  /*
    Run and time the performance test
  */
  cudaEventRecord(start, stream);
  for (int rep = 0; rep < iters; rep++) {
    stat = cublasLtMatmul(handle, mat->descOP, alpha, mat->devA, mat->descA,
                          mat->devB, mat->descB, beta, mat->devC, mat->descC,
                          mat->devD, mat->descD, &mat->algo.algo, mat->devWork,
                          mat->wSZ, stream);
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  // Check for errors during the performance test
  checkCublas(stat);
  checkCuda(cudaGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  cudaEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculateFOM(static_cast<double>(elapsedTime_ms));
}
