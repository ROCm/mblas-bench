#include "cublasGemm.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <bitset>
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
#include "genericInit.h"
#include "mblasCuDataType.h"

using std::cerr;
using std::cout;
using std::endl;
using std::move;
using std::string;
using std::thread;
using std::vector;

// clang-format off
std::vector<gemmPrecType> cublasGemm::gemmExSupported = {
    // Compute type                 Scale Type    A/B Type      C Type
    {MBLAS_COMPUTE_16F,            MBLAS_R_16F,   MBLAS_R_16F,   MBLAS_R_16F  },
    {MBLAS_COMPUTE_16F_PEDANTIC,   MBLAS_R_16F,   MBLAS_R_16F,   MBLAS_R_16F  },
    {MBLAS_COMPUTE_32I,            MBLAS_R_32I,   MBLAS_R_8I,    MBLAS_R_32I  },
    {MBLAS_COMPUTE_32I_PEDANTIC,   MBLAS_R_32I,   MBLAS_R_8I,    MBLAS_R_32I  },
    // Compute type                 Scale Type    A/B Type      C Type
    {MBLAS_COMPUTE_32F,            MBLAS_R_32F,   MBLAS_R_16BF,  MBLAS_R_16BF },
    {MBLAS_COMPUTE_32F_PEDANTIC,   MBLAS_R_32F,   MBLAS_R_16BF,  MBLAS_R_16BF },
    {MBLAS_COMPUTE_32F,            MBLAS_R_32F,   MBLAS_R_16F,   MBLAS_R_16F  },
    {MBLAS_COMPUTE_32F_PEDANTIC,   MBLAS_R_32F,   MBLAS_R_16F,   MBLAS_R_16F  }, 
    {MBLAS_COMPUTE_32F,            MBLAS_R_32F,   MBLAS_R_8I,    MBLAS_R_32F  },
    {MBLAS_COMPUTE_32F_PEDANTIC,   MBLAS_R_32F,   MBLAS_R_8I,    MBLAS_R_32F  },
    {MBLAS_COMPUTE_32F,            MBLAS_R_32F,   MBLAS_R_16BF,  MBLAS_R_32F  },
    {MBLAS_COMPUTE_32F_PEDANTIC,   MBLAS_R_32F,   MBLAS_R_16BF,  MBLAS_R_32F  },
    {MBLAS_COMPUTE_32F,            MBLAS_R_32F,   MBLAS_R_16F,   MBLAS_R_32F  },
    {MBLAS_COMPUTE_32F_PEDANTIC,   MBLAS_R_32F,   MBLAS_R_16F,   MBLAS_R_32F  },
    {MBLAS_COMPUTE_32F,            MBLAS_R_32F,   MBLAS_R_32F,   MBLAS_R_32F  },
    {MBLAS_COMPUTE_32F_PEDANTIC,   MBLAS_R_32F,   MBLAS_R_32F,   MBLAS_R_32F  },
    // Compute type                 Scale Type    A/B Type      C Type
    {MBLAS_COMPUTE_32F,            MBLAS_C_32F,   MBLAS_C_8I,    MBLAS_C_32F  },
    {MBLAS_COMPUTE_32F_PEDANTIC,   MBLAS_C_32F,   MBLAS_C_8I,    MBLAS_C_32F  },
    {MBLAS_COMPUTE_32F,            MBLAS_C_32F,   MBLAS_C_32F,   MBLAS_C_32F  },
    {MBLAS_COMPUTE_32F_PEDANTIC,   MBLAS_C_32F,   MBLAS_C_32F,   MBLAS_C_32F  },
    // Compute type                 Scale Type    A/B Type      C Type
    {MBLAS_COMPUTE_32F_FAST_16F,   MBLAS_R_32F,   MBLAS_R_32F,   MBLAS_R_32F  },
    {MBLAS_COMPUTE_32F_FAST_16BF,  MBLAS_R_32F,   MBLAS_R_32F,   MBLAS_R_32F  },
    {MBLAS_COMPUTE_32F_FAST_TF32,  MBLAS_R_32F,   MBLAS_R_32F,   MBLAS_R_32F  },
    {MBLAS_COMPUTE_32F_FAST_16F,   MBLAS_C_32F,   MBLAS_C_32F,   MBLAS_C_32F  },
    {MBLAS_COMPUTE_32F_FAST_16BF,  MBLAS_C_32F,   MBLAS_C_32F,   MBLAS_C_32F  },
    {MBLAS_COMPUTE_32F_FAST_TF32,  MBLAS_C_32F,   MBLAS_C_32F,   MBLAS_C_32F  },
    // Compute type                 Scale Type    A/B Type      C Type
    {MBLAS_COMPUTE_64F,            MBLAS_R_64F,   MBLAS_R_64F,   MBLAS_R_64F  },
    {MBLAS_COMPUTE_64F_PEDANTIC,   MBLAS_R_64F,   MBLAS_R_64F,   MBLAS_R_64F  },
    {MBLAS_COMPUTE_64F,            MBLAS_C_64F,   MBLAS_C_64F,   MBLAS_C_64F  },
    {MBLAS_COMPUTE_64F_PEDANTIC,   MBLAS_C_64F,   MBLAS_C_64F,   MBLAS_C_64F  },
};
// clang-format on

std::vector<TgemmPrecType> cublasGemm::TgemmExSupported = {
    {mblasCuDataType::MBLAS_R_16BF, mblasCuDataType::MBLAS_R_16BF}, {mblasCuDataType::MBLAS_R_16F, mblasCuDataType::MBLAS_R_16F},
    {mblasCuDataType::MBLAS_R_8I, mblasCuDataType::MBLAS_R_32F},    {mblasCuDataType::MBLAS_R_16BF, mblasCuDataType::MBLAS_R_32F},
    {mblasCuDataType::MBLAS_R_16F, mblasCuDataType::MBLAS_R_32F},   {mblasCuDataType::MBLAS_R_32F, mblasCuDataType::MBLAS_R_32F},
    {mblasCuDataType::MBLAS_C_8I, mblasCuDataType::MBLAS_C_32F},    {mblasCuDataType::MBLAS_C_32F, mblasCuDataType::MBLAS_C_32F},

};

void cublasGemm::initPrecMap() {}

void cublasGemm::parseDevIters(std::string deviceStr) {
  // Parse device
  std::stringstream ss(deviceStr);
  while (ss.good()) {
    string deviceSStr;
    getline(ss, deviceSStr, ',');
    int devInt = stoi(deviceSStr);
    cublasgemmInst val = cublasgemmInst(devInt);
    matPtrs.push_back(val);
  }
}

void cublasGemm::parseMType(string computeTStr, string scalarTStr, string aStr,
                            string bStr, string cStr) {
  compute.set_compute(computeTStr, precision);
  scalar.set_scalar(scalarTStr, precision, compute);

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
  a_type = mblasCuDataType(aStr);
  b_type = mblasCuDataType(bStr);
  c_type = mblasCuDataType(cStr);

  // Validate against supported precision table (fun)
  if (a_type != b_type) {
    string errorString = "A Type must the same as B Type";
    throw std::invalid_argument(errorString);
  }
  if (function.find("GemmEx") || function.find("gemm_ex")) {
    /*
      Possible functions:
        cublasGemmEx
        cublasGemmExBatched
        cublasGemmExStridedBatched
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

cublasGemm::cublasGemm(cxxopts::ParseResult result) : genericGemm(result) {
  // cublasCreate(&handle);
  // checkCublas(cublasCreate(&handle));
  initPrecMap();
  // Grab precision from command line
  //precision = mblasCuDataType(result["precision"].as<string>());
  precision = mblasCuDataType(result["precision"].as<string>());
  // Grab compute type from command line
  string computeT = result["compute_type"].as<string>();
  string scalarT = result["scalar_type"].as<string>();
  string aT = result["a_type"].as<string>();
  string bT = result["b_type"].as<string>();
  string cT = result["c_type"].as<string>();
  string compcomputeT = result["composite_compute_type"].as<string>();
  if (compcomputeT == "f32") {
    // Feature from rocBLAS, set the original compute type
    computeT = "CUBLAS_COMPUTE_32F";
  }
  parseMType(computeT, scalarT, aT, bT, cT);

  parseDevIters(result["device"].as<string>());
  std::string tA = result["transposeA"].as<std::string>();
  std::string tB = result["transposeB"].as<std::string>();
  transA = mblasCuOperation(result["transposeA"].as<std::string>());
  transB = mblasCuOperation(result["transposeB"].as<std::string>());

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

string cublasGemm::prepareArray() {
  alpha = convert_scalar(scalar, alpha);
  beta = convert_scalar(scalar, beta);
  this->alloc_host();
  this->fill_host();

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
  //  this->alloc_dev(&instance);
  //  this->copyHostToDev(&instance);
  //}

  runThreaded(&cublasGemm::alloc_dev);
  runThreaded(&cublasGemm::copyHostToDev);
  std::ostringstream ossHeader;
  ossHeader << "transA_option,transB_option,M,N,K,lda,ldb,ldc,";
  if (batched) {
    ossHeader << "batch_count,";
  }
  ossHeader << "cuBLAS-Gflops,cuBLAS-GB/s,cuBLAS-us," << endl;
  return ossHeader.str();
}

void cublasGemm::runThreaded(void (cublasGemm::*func)(cublasgemmInst *)) {
  vector<thread> threads;
  for (auto &instance : matPtrs) {
    threads.push_back(thread(func, this, &instance));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void cublasGemm::alloc_host() {
  // auto resultA = std::async(allocateHostArr, a_type, m, k, batch_count);
  // auto resultB = std::async(allocateHostArr, b_type, k, n, batch_count);
  // auto resultC = std::async(allocateHostArr, c_type, n, m, batch_count);
  // hostA = resultA.get();
  // hostB = resultB.get();
  // hostC = resultC.get();
  hostA = allocateHostArr(a_type, rows_mem_a, cols_mem_a, batch_count);
  hostB = allocateHostArr(b_type, rows_mem_b, cols_mem_b, batch_count);
  hostC = allocateHostArr(c_type, rows_mem_c, cols_mem_c, batch_count);
}

void cublasGemm::alloc_dev(cublasgemmInst *mat) {
  cudaSetDevice(mat->devIDX);
  mat->devA = allocateDevArr(a_type, rows_mem_a, cols_mem_a, batch_count);
  mat->devB = allocateDevArr(b_type, rows_mem_b, cols_mem_b, batch_count);
  mat->devC = allocateDevArr(c_type, rows_mem_c, cols_mem_c, batch_count);
  mat->wSZ = workspaceSz;
  cudaMalloc(&mat->devWork, mat->wSZ);
}

void cublasGemm::fill_host() {
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
  //                         batch_count, stride_c, 0.f, false));
  // for (auto &thread : threads) {
  //  thread.join();
  //}
  typeCallHost<initHost>(a_type, initialization, hostA, rows_a, cols_a, lda,
                         batch_count, stride_a, control_a, constant_a, filename_a);
  typeCallHost<initHost>(b_type, initialization, hostB, rows_b, cols_b, ldb,
                         batch_count, stride_b, control_b, constant_b, filename_b);
  typeCallHost<initHost>(c_type, initialization, hostC, rows_c, cols_c, ldc,
                         batch_count, stride_c, control_c, constant_c, filename_c);
  // initHostH(a_type, initialization, hostA, rows_a, cols_a, lda,
  //           batch_count, stride_a, control_a, constant_a, filename_a);
  // initHostH(b_type, initialization, hostB, rows_b, cols_b, ldb,
  //           batch_count, stride_b, control_b, constant_b, filename_b);
  // initHostH(c_type, initialization, hostC, rows_c, cols_c, ldc,
  //                        batch_count, stride_c, control_c, constant_c, filename_c);
}

void cublasGemm::copyHostToDev(cublasgemmInst *mat) {
  cudaSetDevice(mat->devIDX);
  copy_and_convert(a_type, hostA, mat->devA, rows_mem_a, cols_mem_a, batch_count);
  copy_and_convert(b_type, hostB, mat->devB, rows_mem_b, cols_mem_b, batch_count);
  copy_and_convert(c_type, hostC, mat->devC, rows_mem_c, cols_mem_c, batch_count);
  if (batched && !strided) {
    // Perform some pointer arithmetic to calculate the arrays we pass to the
    // gpu
    mat->ptr_host_a =
        (void **)malloc(batch_count * typeCallHost<sizeofCUDTP>(a_type));
    mat->ptr_host_b =
        (void **)malloc(batch_count * typeCallHost<sizeofCUDTP>(b_type));
    mat->ptr_host_c =
        (void **)malloc(batch_count * typeCallHost<sizeofCUDTP>(c_type));
    checkCuda(
        cudaMalloc(&mat->ptrDevA, batch_count * typeCallHost<sizeofCUDTP>(a_type)));
    checkCuda(
        cudaMalloc(&mat->ptrDevB, batch_count * typeCallHost<sizeofCUDTP>(b_type)));
    checkCuda(
        cudaMalloc(&mat->ptrDevC, batch_count * typeCallHost<sizeofCUDTP>(c_type)));
    //typeCallDev<batchedPtrMagic>(a_type, mat->ptr_host_a, mat->ptrDevA, mat->devA,
    //                            batch_count, rows_mem_a, cols_mem_a);
    //typeCallDev<batchedPtrMagic>(b_type, mat->ptr_host_b, mat->ptrDevB, mat->devB,
    //                            batch_count, rows_mem_b, cols_mem_b);
    //typeCallDev<batchedPtrMagic>(c_type, mat->ptr_host_c, mat->ptrDevC, mat->devC,
    //                            batch_count, rows_mem_c, cols_mem_c);
  }
}

void cublasGemm::freeMem() {
  free(alpha);
  free(beta);
  free(hostA);
  free(hostB);
  free(hostC);
  for (auto mat : matPtrs) {
    cudaFree(mat.devA);
    cudaFree(mat.devB);
    cudaFree(mat.devC);
    cudaFree(mat.devWork);
    if (batched && !strided) {
      free(mat.ptr_host_a);
      free(mat.ptr_host_b);
      free(mat.ptr_host_c);
      cudaFree(mat.ptrDevA);
      cudaFree(mat.ptrDevB);
      cudaFree(mat.ptrDevC);
    }
  }
}

double cublasGemm::test() {
  vector<thread> threads;
  double gflops = 0.0;
  for (auto &mat : matPtrs) {
    // TgemmBatched
    if (matchGemmType(precision, function, mblasDataType::MBLAS_R_64F, {"cublasDgemm", "dgemm", "gemm"})) {
      std::function<decltype(cublasDgemm)> dgemm_var = cublasDgemm;
      threads.push_back(
          thread(&cublasGemm::testTgemm<double>, this, dgemm_var, &mat));
    } else if (matchGemmType(precision, function, mblasDataType::MBLAS_R_32F, {"cublasSgemm", "sgemm", "gemm"})) {
      std::function<decltype(cublasSgemm)> sgemm_var = cublasSgemm;
      threads.push_back(
          thread(&cublasGemm::testTgemm<float>, this, sgemm_var, &mat));
    } else if (matchGemmType(precision, function, mblasDataType::MBLAS_R_16F, {"cublasHgemm", "hgemm", "gemm"})) {
      std::function<decltype(cublasHgemm)> hgemm_var = cublasHgemm;
      threads.push_back(
          thread(&cublasGemm::testTgemm<__half>, this, hgemm_var, &mat));
    } else if (matchGemmType(precision, function, mblasDataType::MBLAS_C_64F, {"cublasZgemm", "zgemm", "gemm"})) {
      std::function<decltype(cublasZgemm)> zgemm_var = cublasZgemm;
      threads.push_back(thread(&cublasGemm::testTgemm<cuDoubleComplex>, this,
                               zgemm_var, &mat));
    } else if (matchGemmType(precision, function, mblasDataType::MBLAS_C_32F, {"cublasCgemm", "cgemm", "gemm"})) {
      std::function<decltype(cublasCgemm)> cgemm_var = cublasCgemm;
      threads.push_back(
          thread(&cublasGemm::testTgemm<cuComplex>, this, cgemm_var, &mat));
    } else if (matchGemmType(precision, function, mblasDataType::MBLAS_C_64F, {"cublasZgemm3m", "zgemm3m", "gemm3m"})) {
      std::function<decltype(cublasZgemm3m)> zgemm3m_var = cublasZgemm3m;
      threads.push_back(thread(&cublasGemm::testTgemm<cuDoubleComplex>, this,
                               zgemm3m_var, &mat));
    } else if (matchGemmType(precision, function, mblasDataType::MBLAS_C_32F, {"cublasCgemm3m", "cgemm3m", "gemm3m"})) {
      std::function<decltype(cublasCgemm3m)> cgemm3m_var = cublasCgemm3m;
      threads.push_back(
          thread(&cublasGemm::testTgemm<cuComplex>, this, cgemm3m_var, &mat));
    }
    // TgemmBatched
    else if (function == "cublasDgemmBatched" && precision == mblasDataType::MBLAS_R_64F) {
      std::function<decltype(cublasDgemmBatched)> dgemm_var =
          cublasDgemmBatched;
      threads.push_back(
          thread(&cublasGemm::testTgemmBatched<double>, this, dgemm_var, &mat));
    } else if (function == "cublasSgemmBatched" && precision == mblasDataType::MBLAS_R_32F) {
      std::function<decltype(cublasSgemmBatched)> sgemm_var =
          cublasSgemmBatched;
      threads.push_back(
          thread(&cublasGemm::testTgemmBatched<float>, this, sgemm_var, &mat));
    } else if (function == "cublasHgemmBatched" && precision == mblasDataType::MBLAS_R_16F) {
      std::function<decltype(cublasHgemmBatched)> hgemm_var =
          cublasHgemmBatched;
      threads.push_back(
          thread(&cublasGemm::testTgemmBatched<__half>, this, hgemm_var, &mat));
    } else if (function == "cublasZgemmBatched" && precision == mblasDataType::MBLAS_C_64F) {
      std::function<decltype(cublasZgemmBatched)> zgemm_var =
          cublasZgemmBatched;
      threads.push_back(thread(&cublasGemm::testTgemmBatched<cuDoubleComplex>,
                               this, zgemm_var, &mat));
    } else if (function == "cublasCgemmBatched" && precision == mblasDataType::MBLAS_C_32F) {
      std::function<decltype(cublasCgemmBatched)> cgemm_var =
          cublasCgemmBatched;
      threads.push_back(thread(&cublasGemm::testTgemmBatched<cuComplex>, this,
                               cgemm_var, &mat));
    }
    // TgemmStridedBatched
    else if (function == "cublasDgemmStridedBatched" &&
             precision == mblasDataType::MBLAS_R_64F) {
      std::function<decltype(cublasDgemmStridedBatched)> dgemm_var =
          cublasDgemmStridedBatched;
      threads.push_back(thread(&cublasGemm::testTgemmStridedBatched<double>,
                               this, dgemm_var, &mat));
    } else if (function == "cublasSgemmStridedBatched" &&
               precision == mblasDataType::MBLAS_R_32F) {
      std::function<decltype(cublasSgemmStridedBatched)> sgemm_var =
          cublasSgemmStridedBatched;
      threads.push_back(thread(&cublasGemm::testTgemmStridedBatched<float>,
                               this, sgemm_var, &mat));
    } else if (function == "cublasHgemmStridedBatched" &&
               precision == mblasDataType::MBLAS_R_16F) {
      std::function<decltype(cublasHgemmStridedBatched)> hgemm_var =
          cublasHgemmStridedBatched;
      threads.push_back(thread(&cublasGemm::testTgemmStridedBatched<__half>,
                               this, hgemm_var, &mat));
    } else if (function == "cublasZgemmStridedBatched" &&
               precision == mblasDataType::MBLAS_C_64F) {
      std::function<decltype(cublasZgemmStridedBatched)> zgemm_var =
          cublasZgemmStridedBatched;
      threads.push_back(
          thread(&cublasGemm::testTgemmStridedBatched<cuDoubleComplex>, this,
                 zgemm_var, &mat));
    } else if (function == "cublasCgemmStridedBatched" &&
               precision == mblasDataType::MBLAS_C_32F) {
      std::function<decltype(cublasCgemmStridedBatched)> cgemm_var =
          cublasCgemmStridedBatched;
      threads.push_back(thread(&cublasGemm::testTgemmStridedBatched<cuComplex>,
                               this, cgemm_var, &mat));
    } else if (function == "cublasCgemm3mStridedBatched" &&
               precision == mblasDataType::MBLAS_C_32F) {
      std::function<decltype(cublasCgemm3mStridedBatched)> cgemm_var =
          cublasCgemm3mStridedBatched;
      threads.push_back(thread(&cublasGemm::testTgemmStridedBatched<cuComplex>,
                               this, cgemm_var, &mat));
    }
    // TgemmEx
    else if (function == "cublasSgemmEx") {
      std::function<decltype(cublasSgemmEx)> sgemm_var = cublasSgemmEx;
      threads.push_back(
          thread(&cublasGemm::testTGemmEx<float>, this, sgemm_var, &mat));
    } else if (function == "cublasCgemmEx") {
      std::function<decltype(cublasCgemmEx)> cgemm_var = cublasCgemmEx;
      threads.push_back(
          thread(&cublasGemm::testTGemmEx<cuComplex>, this, cgemm_var, &mat));
    }
    // gemmEx
    else if (strided && function == "cublasGemmExStridedBatched") {
      // Call the Gemm strided batched deployment script
    } else if (batched && function == "cublasGemmExBatched") {
      // Call the Gemm batched code
    } else if (function == "cublasGemmEx" || function == "gemm_ex" || function == "gemm_ex3") {
      threads.push_back(thread(&cublasGemm::testGemmEx, this, &mat));
    }
  }
  // Wait on running jobs
  for (auto &thread : threads) {
    thread.join();
  }

  // Sum all gflops
  gflop_per_second = std::accumulate(
      begin(matPtrs), end(matPtrs), 0.0,
      [](double i, const cublasgemmInst &o) { return o.gflops + i; });

  gbyte_per_second = std::accumulate(
      begin(matPtrs), end(matPtrs), 0.0,
      [](double i, const cublasgemmInst &o) { return o.gbytes + i; });

  iter_time_us = std::accumulate(begin(matPtrs), end(matPtrs), 0.0,
                                 [](double i, const cublasgemmInst &o) {
                                   return o.time_us + i;
                                 }) /
                 matPtrs.size();

  return gflop_per_second;
}

std::string cublasGemm::getResultString() {
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

std::tuple<double, double, double> cublasGemm::calculateFOM(
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

template <typename T>
void cublasGemm::testTgemm(
    std::function<cublasStatus_t(
        cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
        const T *, const T *, int, const T *, int, const T *, T *, int)>
        func,
    cublasgemmInst *mat) {
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;
  checkCuda(cudaSetDevice(mat->devIDX));
  checkCublas(cublasCreate(&handle));
  checkCuda(cudaStreamCreate(&stream));
  checkCublas(cublasSetStream(handle, stream));
  // checkCublas(cublasSetWorkspace(handle, mat->devWork, mat->wSZ));

  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T *devAP = static_cast<T *>(mat->devA);
  T *devBP = static_cast<T *>(mat->devB);
  T *devCP = static_cast<T *>(mat->devC);

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = func(handle, transA.convertToCuda(), transB.convertToCuda(), m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc);

    // Check for errors during the gemm run
    checkCublas(stat);
    checkCuda(cudaGetLastError());
  }
  cudaStreamSynchronize(stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /*
    Run and time the performance test
  */
  cudaEventRecord(start, stream);
  for (int rep = 0; rep < iters; rep++) {
    stat = func(handle, transA.convertToCuda(), transB.convertToCuda(), m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc);
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

template <typename T>
void cublasGemm::testTgemmBatched(
    std::function<cublasStatus_t(cublasContext *, cublasOperation_t,
                                 cublasOperation_t, int, int, int, T const *,
                                 T const *const *, int, T const *const *, int,
                                 T const *, T *const *, int, int)>
        func,
    cublasgemmInst *mat) {
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;
  checkCuda(cudaSetDevice(mat->devIDX));
  checkCublas(cublasCreate(&handle));
  checkCuda(cudaStreamCreate(&stream));
  checkCublas(cublasSetStream(handle, stream));
  // checkCublas(cublasSetWorkspace(handle, mat->devWork, mat->wSZ));

  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T **devAP = reinterpret_cast<T **>(mat->ptrDevA);
  T **devBP = reinterpret_cast<T **>(mat->ptrDevB);
  T **devCP = reinterpret_cast<T **>(mat->ptrDevC);

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = func(handle, transA.convertToCuda(), transB.convertToCuda(), m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc, batch_count);

    // Check for errors during the gemm run
    checkCublas(stat);
    checkCuda(cudaGetLastError());
  }
  cudaStreamSynchronize(stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /*
    Run and time the performance test
  */
  cudaEventRecord(start, stream);
  for (int rep = 0; rep < iters; rep++) {
    stat = func(handle, transA.convertToCuda(), transB.convertToCuda(), m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc, batch_count);
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

template <typename T>
void cublasGemm::testTgemmStridedBatched(
    std::function<cublasStatus_t(
        cublasContext *, cublasOperation_t, cublasOperation_t, int, int, int,
        T const *, T const *, int, long long, T const *, int, long long,
        T const *, T *, int, long long, int)>
        func,
    cublasgemmInst *mat) {
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;
  checkCuda(cudaSetDevice(mat->devIDX));
  checkCublas(cublasCreate(&handle));
  checkCuda(cudaStreamCreate(&stream));
  checkCublas(cublasSetStream(handle, stream));
  // checkCublas(cublasSetWorkspace(handle, mat->devWork, mat->wSZ));

  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T *devAP = static_cast<T *>(mat->devA);
  T *devBP = static_cast<T *>(mat->devB);
  T *devCP = static_cast<T *>(mat->devC);

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = func(handle, transA.convertToCuda(), transB.convertToCuda(), m, n, k, alphaP, devAP, lda, stride_a,
                devBP, ldb, stride_b, betaP, devCP, ldc, stride_c, batch_count);

    // Check for errors during the gemm run
    checkCublas(stat);
    checkCuda(cudaGetLastError());
  }
  cudaStreamSynchronize(stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /*
    Run and time the performance test
  */
  cudaEventRecord(start, stream);
  for (int rep = 0; rep < iters; rep++) {
    stat = func(handle, transA.convertToCuda(), transB.convertToCuda(), m, n, k, alphaP, devAP, lda, stride_a,
                devBP, ldb, stride_b, betaP, devCP, ldc, stride_c, batch_count);
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

template <typename T>
void cublasGemm::testTGemmEx(
    std::function<cublasStatus_t(
        cublasContext *, cublasOperation_t, cublasOperation_t, int, int, int,
        T const *, void const *, cudaDataType_t, int, void const *,
        cudaDataType_t, int, T const *, void *, cudaDataType_t, int)>
        func,
    cublasgemmInst *mat) {
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;
  checkCuda(cudaSetDevice(mat->devIDX));
  checkCublas(cublasCreate(&handle));
  checkCuda(cudaStreamCreate(&stream));
  checkCublas(cublasSetStream(handle, stream));
  // checkCublas(cublasSetWorkspace(handle, mat->devWork, mat->wSZ));

  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  // T *devAP = static_cast<T *>(mat->devA);
  // T *devBP = static_cast<T *>(mat->devB);
  // T *devCP = static_cast<T *>(mat->devC);

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = func(handle, transA.convertToCuda(), transB.convertToCuda(), m, n, k, alphaP, mat->devA, a_type, lda,
                mat->devB, b_type, ldb, betaP, mat->devC, c_type, ldc);

    // Check for errors during the gemm run
    checkCublas(stat);
    checkCuda(cudaGetLastError());
  }
  cudaStreamSynchronize(stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /*
    Run and time the performance test
  */
  cudaEventRecord(start, stream);
  for (int rep = 0; rep < iters; rep++) {
    stat = func(handle, transA.convertToCuda(), transB.convertToCuda(), m, n, k, alphaP, mat->devA, a_type, lda,
                mat->devB, b_type, ldb, betaP, mat->devC, c_type, ldc);
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

void cublasGemm::testGemmEx(cublasgemmInst *mat) {
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;
  checkCuda(cudaSetDevice(mat->devIDX));
  checkCublas(cublasCreate(&handle));
  checkCuda(cudaStreamCreate(&stream));
  checkCublas(cublasSetStream(handle, stream));
  checkCublas(cublasSetWorkspace(handle, mat->devWork, mat->wSZ));
  // cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = cublasGemmEx(handle, transA.convertToCuda(), transB.convertToCuda(), m, n, k, alpha, mat->devA,
                        a_type, lda, mat->devB, b_type, ldb, beta, mat->devC,
                        c_type, ldc, compute, CUBLAS_GEMM_DEFAULT);

    // Check for errors during the gemm run
    checkCublas(stat);
    checkCuda(cudaGetLastError());
  }
  cudaStreamSynchronize(stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /*
    Run and time the performance test
  */
  cudaEventRecord(start, stream);
  for (int rep = 0; rep < iters; rep++) {
    stat = cublasGemmEx(handle, transA.convertToCuda(), transB.convertToCuda(), m, n, k, alpha, mat->devA,
                        a_type, lda, mat->devB, b_type, ldb, beta, mat->devC,
                        c_type, ldc, compute, CUBLAS_GEMM_DEFAULT);
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
