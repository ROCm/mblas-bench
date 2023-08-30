#include "rocblasGemm.h"

#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>

#include <bitset>
#include <future>
#include <iomanip>
#include <numeric>
#include <regex>
#include <string>
#include <thread>

#include <iostream>

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
std::vector<gemmPrecTypeAMD> rocblasGemm::gemmExSupported = {
    // Compute type             Scale Type                A/B Type                  C Type
    {rocblas_datatype_f64_r,    rocblas_datatype_f64_r,   rocblas_datatype_f64_r,   rocblas_datatype_f64_r  },
    {rocblas_datatype_f32_r,    rocblas_datatype_f32_r,   rocblas_datatype_f32_r,   rocblas_datatype_f32_r  },
    {rocblas_datatype_f16_r,    rocblas_datatype_f16_r,   rocblas_datatype_f16_r,   rocblas_datatype_f16_r  },
    {rocblas_datatype_f32_r,    rocblas_datatype_f32_r,   rocblas_datatype_f16_r,   rocblas_datatype_f16_r  },
    {rocblas_datatype_f32_r,    rocblas_datatype_f32_r,   rocblas_datatype_f16_r,   rocblas_datatype_f32_r  },
    {rocblas_datatype_f32_r,    rocblas_datatype_f32_r,   rocblas_datatype_bf16_r,  rocblas_datatype_bf16_r },
    {rocblas_datatype_f32_r,    rocblas_datatype_f32_r,   rocblas_datatype_bf16_r,  rocblas_datatype_f32_r  },
    {rocblas_datatype_i32_r,    rocblas_datatype_i32_r,   rocblas_datatype_i8_r,    rocblas_datatype_i32_r  }, 
    {rocblas_datatype_f32_c,    rocblas_datatype_f32_c,   rocblas_datatype_f32_c,   rocblas_datatype_f32_c  },
    {rocblas_datatype_f64_c,    rocblas_datatype_f64_c,   rocblas_datatype_f64_c,   rocblas_datatype_f64_c  },
};
// clang-format on

std::vector<TgemmPrecTypeAMD> rocblasGemm::TgemmExSupported = {};

void rocblasGemm::initPrecMap() {}

void rocblasGemm::parseDevIters(std::string deviceStr) {
  // Parse device
  std::stringstream ss(deviceStr);
  while (ss.good()) {
    string deviceSStr;
    getline(ss, deviceSStr, ',');
    int devInt = stoi(deviceSStr);
    rocblasgemmInst val = rocblasgemmInst(devInt, blockct);
    matPtrs.push_back(val);
  }
}

void rocblasGemm::parseMType(string computeTStr, string scalarTStr, string aStr,
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
  if (function.find("gemm_ex")) {
    /*
      Possible functions:
        rocblasgemm_ex
        rocblasgemm_exBatched
        rocblasgemm_exStridedBatched
    */
    gemmPrecTypeAMD selType = {compute, scalar, a_type, c_type};
    auto result =
        std::find(begin(gemmExSupported), end(gemmExSupported), selType);
    if (result == end(gemmExSupported)) {
      // Unable to find matching config, not supported
      string errorString =
          "Invalid GEMM specification for gemm_ex.  Combination of parameters "
          "not supported"
          "\nCompute type: " +
          computeTStr + "\nScalar type: " + scalarTStr + "\nA type: " + aStr +
          "\nB type: " + bStr + "\nC type: " + cStr;
      throw std::invalid_argument(errorString);
    }
   
  }
}

rocblasGemm::rocblasGemm(cxxopts::ParseResult result) : genericGemm(result) {
  // rocblas_create_handle(&handle);
  // checkRocblas(rocblas_create_handle(&handle));
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

string rocblasGemm::prepareArray() {
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

  runThreaded(&rocblasGemm::allocDev);
  runThreaded(&rocblasGemm::copyHostToDev);
  std::ostringstream ossHeader;
  ossHeader << "transA_option,transB_option,M,N,K,lda,ldb,ldc,";
  if (batched) {
    ossHeader << "batch_count,";
  }
  ossHeader << "rocBLAS-Gflops,rocBLAS-GB/s,rocBLAS-us," << endl;
  return ossHeader.str();
}

void rocblasGemm::runThreaded(void (rocblasGemm::*func)(rocblasgemmInst *)) {
  vector<thread> threads;
  for (auto &instance : matPtrs) {
    threads.push_back(thread(func, this, &instance));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void rocblasGemm::allocHost() {
  // auto resultA = std::async(allocateHostArr, a_type, m, k, batchct);
  // auto resultB = std::async(allocateHostArr, b_type, k, n, batchct);
  // auto resultC = std::async(allocateHostArr, c_type, n, m, batchct);
  // hostA = resultA.get();
  // hostB = resultB.get();
  // hostC = resultC.get();
  hostA = allocateHostArr(a_type, m, k, batchct, blockct);
  hostB = allocateHostArr(b_type, k, n, batchct, blockct);
  hostC = allocateHostArr(c_type, m, n, batchct, blockct);
}

void rocblasGemm::allocDev(rocblasgemmInst *mat) {
  hipSetDevice(mat->devIDX);
  mat->devA = allocateDevArr(a_type, m, k, batchct, blockct);
  mat->devB = allocateDevArr(b_type, k, n, batchct, blockct);
  mat->devC = allocateDevArr(c_type, m, n, batchct, blockct);
  mat->wSZ = workspaceSz;
  hipMalloc(&mat->devWork, mat->wSZ);
}

void rocblasGemm::fillHost() {
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
                         batchct, stride_a, blockct, controlA, constantA, filenameA);
  typeCallHost<initHost>(b_type, initialization, hostB, rowsB, colsB, ldb,
                         batchct, stride_b, blockct, controlB, constantB, filenameB);
  typeCallHost<initHost>(c_type, initialization, hostC, rowsC, colsC, ldc,
                         batchct, stride_c, blockct, controlC, constantC, filenameC);
}

void rocblasGemm::copyHostToDev(rocblasgemmInst *mat) {
  hipSetDevice(mat->devIDX);
  copyAndConvert(a_type, hostA, mat->devA, m, k, batchct, blockct);
  copyAndConvert(b_type, hostB, mat->devB, k, n, batchct, blockct);
  copyAndConvert(c_type, hostC, mat->devC, n, m, batchct, blockct);
  if (batched && !strided) {
    // Perform some pointer arithmetic to calculate the arrays we pass to the
    // gpu
    mat->ptrHostA =
        (void **)malloc(batchct * blockct * typeCallHost<sizeofCUDTP>(a_type));
    mat->ptrHostB =
        (void **)malloc(batchct * blockct * typeCallHost<sizeofCUDTP>(b_type));
    mat->ptrHostC =
        (void **)malloc(batchct * blockct * typeCallHost<sizeofCUDTP>(c_type));
    checkHip(
        hipMalloc(&mat->ptrDevA, batchct * blockct * typeCallHost<sizeofCUDTP>(a_type)));
    checkHip(
        hipMalloc(&mat->ptrDevB, batchct * blockct * typeCallHost<sizeofCUDTP>(b_type)));
    checkHip(
        hipMalloc(&mat->ptrDevC, batchct * blockct * typeCallHost<sizeofCUDTP>(c_type)));
    typeCallDev<batchedPtrMagic>(a_type, mat->ptrHostA, mat->ptrDevA, mat->devA,
                                batchct, m, k, blockct);
    typeCallDev<batchedPtrMagic>(b_type, mat->ptrHostB, mat->ptrDevB, mat->devB,
                                batchct, k, n, blockct);
    typeCallDev<batchedPtrMagic>(c_type, mat->ptrHostC, mat->ptrDevC, mat->devC,
                                batchct, n, m, blockct);
  }
}

void rocblasGemm::freeMem() {
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

double rocblasGemm::test() {
  vector<thread> threads;
  double gflops = 0.0;
  for (auto &mat : matPtrs) {
    // Tgemm
    if ((function == "rocblas_dgemm" && precision == rocblas_datatype_f64_r) ||
        (function == "gemm" && precision == rocblas_datatype_f64_r)) {
      std::function<decltype(rocblas_dgemm)> dgemm_var = rocblas_dgemm;
      threads.push_back(
          thread(&rocblasGemm::testTgemm<double>, this, dgemm_var, &mat));
    } else if ((function == "rocblas_sgemm" && precision == rocblas_datatype_f32_r) ||
               (function == "gemm" && precision == rocblas_datatype_f32_r)) {
      std::function<decltype(rocblas_sgemm)> sgemm_var = rocblas_sgemm;
      threads.push_back(
          thread(&rocblasGemm::testTgemm<float>, this, sgemm_var, &mat));
    } else if ((function == "rocblas_hgemm" && precision == rocblas_datatype_f16_r) ||
               (function == "gemm" && precision == rocblas_datatype_f16_r)) {
      std::function<decltype(rocblas_hgemm)> hgemm_var = rocblas_hgemm;
      threads.push_back(
          thread(&rocblasGemm::testTgemm<rocblas_half>, this, hgemm_var, &mat));
    } else if ((function == "rocblas_zgemm" && precision == rocblas_datatype_f64_c) ||
               (function == "gemm" && precision == rocblas_datatype_f64_c)) {
      std::function<decltype(rocblas_zgemm)> zgemm_var = rocblas_zgemm;
      threads.push_back(thread(&rocblasGemm::testTgemm<rocblas_complex_num<double>>, this,
                               zgemm_var, &mat));
    } else if ((function == "rocblas_cgemm" && precision == rocblas_datatype_f32_c) ||
               (function == "gemm" && precision == rocblas_datatype_f32_c)) {
      std::function<decltype(rocblas_cgemm)> cgemm_var = rocblas_cgemm;
      threads.push_back(
          thread(&rocblasGemm::testTgemm<rocblas_complex_num<float>>, this, cgemm_var, &mat));
    }
    // TgemmBatched
    else if (function == "rocblas_dgemm_batched" && precision == rocblas_datatype_f64_r) {
      std::function<decltype(rocblas_dgemm_batched)> dgemm_var =
          rocblas_dgemm_batched;
      threads.push_back(
          thread(&rocblasGemm::testTgemm_batched<double>, this, dgemm_var, &mat));
    } else if (function == "rocblas_sgemm_batched" && precision == rocblas_datatype_f32_r) {
      std::function<decltype(rocblas_sgemm_batched)> sgemm_var =
          rocblas_sgemm_batched;
      threads.push_back(
          thread(&rocblasGemm::testTgemm_batched<float>, this, sgemm_var, &mat));
    } else if (function == "rocblas_hgemm_batched" && precision == rocblas_datatype_f16_r) {
      std::function<decltype(rocblas_hgemm_batched)> hgemm_var =
          rocblas_hgemm_batched;
      threads.push_back(
          thread(&rocblasGemm::testTgemm_batched<rocblas_half>, this, hgemm_var, &mat));
    } else if (function == "rocblas_zgemm_batched" && precision == rocblas_datatype_f64_c) {
      std::function<decltype(rocblas_zgemm_batched)> zgemm_var =
          rocblas_zgemm_batched;
      threads.push_back(thread(&rocblasGemm::testTgemm_batched<rocblas_complex_num<double>>,
                               this, zgemm_var, &mat));
    } else if (function == "rocblas_cgemm_batched" && precision == rocblas_datatype_f32_c) {
      std::function<decltype(rocblas_cgemm_batched)> cgemm_var =
          rocblas_cgemm_batched;
      threads.push_back(thread(&rocblasGemm::testTgemm_batched<rocblas_complex_num<float>>, this,
                               cgemm_var, &mat));
    }
    // TgemmStridedBatched
    else if (function == "rocblas_dgemm_strided_batched" &&
             precision == rocblas_datatype_f64_r) {
      std::function<decltype(rocblas_dgemm_strided_batched)> dgemm_var =
          rocblas_dgemm_strided_batched;
      threads.push_back(thread(&rocblasGemm::testTgemm_strided_batched<double>,
                               this, dgemm_var, &mat));
    } else if (function == "rocblas_sgemm_strided_batched" &&
               precision == rocblas_datatype_f32_r) {
      std::function<decltype(rocblas_sgemm_strided_batched)> sgemm_var =
          rocblas_sgemm_strided_batched;
      threads.push_back(thread(&rocblasGemm::testTgemm_strided_batched<float>,
                               this, sgemm_var, &mat));
    } else if (function == "rocblas_hgemm_strided_batched" &&
               precision == rocblas_datatype_f16_r) {
      std::function<decltype(rocblas_hgemm_strided_batched)> hgemm_var =
          rocblas_hgemm_strided_batched;
      threads.push_back(thread(&rocblasGemm::testTgemm_strided_batched<rocblas_half>,
                               this, hgemm_var, &mat));
    } else if (function == "rocblas_zgemm_strided_batched" &&
               precision == rocblas_datatype_f64_c) {
      std::function<decltype(rocblas_zgemm_strided_batched)> zgemm_var =
          rocblas_zgemm_strided_batched;
      threads.push_back(
          thread(&rocblasGemm::testTgemm_strided_batched<rocblas_complex_num<double>>, this,
                 zgemm_var, &mat));
    } else if (function == "rocblas_cgemm_strided_batched" &&
               precision == rocblas_datatype_f32_c) {
      std::function<decltype(rocblas_cgemm_strided_batched)> cgemm_var =
          rocblas_cgemm_strided_batched;
      threads.push_back(thread(&rocblasGemm::testTgemm_strided_batched<rocblas_complex_num<float>>,
                               this, cgemm_var, &mat));
    }
    
    // gemmEx
    // else if (strided && function == "rocblas_gemm_strided_batched_ex") {
    else if (strided && function == "rocblas_gemm_strided_batched_ex") {
      // Call the Gemm strided batched deployment script
    } else if (batched && function == "rocblas_gemm_batched_ex") {
      // Call the Gemm batched code
    } else if (function == "rocblas_gemm_ex" || function == "gemm_ex") {
      threads.push_back(thread(&rocblasGemm::test_gemm_ex, this, &mat));
    }
  }
  // Wait on running jobs
  for (auto &thread : threads) {
    thread.join();
  }

  // Sum all gflops
  gflop_per_second = std::accumulate(
      begin(matPtrs), end(matPtrs), 0.0,
      [](double i, const rocblasgemmInst &o) { return o.gflops + i; });

  gbyte_per_second = std::accumulate(
      begin(matPtrs), end(matPtrs), 0.0,
      [](double i, const rocblasgemmInst &o) { return o.gbytes + i; });

  iter_time_us = std::accumulate(begin(matPtrs), end(matPtrs), 0.0,
                                 [](double i, const rocblasgemmInst &o) {
                                   return o.time_us + i;
                                 }) /
                 matPtrs.size();

  return gflop_per_second;
}

std::string rocblasGemm::getResultString() {
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

std::tuple<double, double, double> rocblasGemm::calculateFOM(
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
void rocblasGemm::testTgemm(std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const*, int, T const*, int, T const*, T*, int)> func, rocblasgemmInst *mat) {
  rocblas_status stat;
  rocblas_handle handle;
  hipStream_t stream;
  checkHip(hipSetDevice(mat->devIDX));
  checkRocblas(rocblas_create_handle(&handle));
  checkHip(hipStreamCreate(&stream));
  checkRocblas(rocblas_set_stream(handle, stream));
  // checkRocblas(rocblas_set_workspace(handle, mat->devWork, mat->wSZ));

  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T *devAP = static_cast<T *>(mat->devA);
  T *devBP = static_cast<T *>(mat->devB);
  T *devCP = static_cast<T *>(mat->devC);

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    // clang-format off
    stat = func(handle, transA, transB, m, n, k, alphaP, 
               devAP, lda, 
               devBP, ldb, betaP, 
               devCP, ldc);
    // clang-format on
    // Check for errors during the gemm run
    checkRocblas(stat);
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
    // clang-format off
    stat = func(handle, transA, transB, m, n, k, alphaP, 
               devAP, lda, 
               devBP, ldb, betaP, 
               devCP, ldc);
    // clang-format on
  }
  hipEventRecord(stop, stream);
  hipEventSynchronize(stop);

  // Check for errors during the performance test
  checkRocblas(stat);
  checkHip(hipGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  hipEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculateFOM(static_cast<double>(elapsedTime_ms));
}

template <typename T>
void rocblasGemm::testTgemm_batched(std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const* const*, int, T const* const*, int, T const*, T* const*, int, int)> func, rocblasgemmInst *mat) {
  rocblas_status stat;
  rocblas_handle handle;
  hipStream_t stream;
  checkHip(hipSetDevice(mat->devIDX));
  checkRocblas(rocblas_create_handle(&handle));
  checkHip(hipStreamCreate(&stream));
  checkRocblas(rocblas_set_stream(handle, stream));
  // checkRocblas(rocblas_set_workspace(handle, mat->devWork, mat->wSZ));

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
    checkRocblas(stat);
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
  checkRocblas(stat);
  checkHip(hipGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  hipEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculateFOM(static_cast<double>(elapsedTime_ms));
}
template<typename T>
inline T * rocblasGemm::getOffsetPtrNC(T * mat, long long int blockstride, int rep) {
  return (mat + (rep%blockct)*blockstride);
}

template <typename T>
void rocblasGemm::testTgemm_strided_batched(
          std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const*, int, long, T const*, int, long, T const*, T*, int, long, int)>
          func, rocblasgemmInst *mat) {
  rocblas_status stat;
  rocblas_handle handle;
  hipStream_t stream;
  checkHip(hipSetDevice(mat->devIDX));
  checkRocblas(rocblas_create_handle(&handle));
  checkHip(hipStreamCreate(&stream));
  checkRocblas(rocblas_set_stream(handle, stream));
  // checkRocblas(rocblas_set_workspace(handle, mat->devWork, mat->wSZ));
  int blockstrideA = stride_a*blockct;
  int blockstrideB = stride_b*blockct;
  int blockstrideC = stride_c*blockct;

  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T *devAP = static_cast<T *>(mat->devA);
  T *devBP = static_cast<T *>(mat->devB);
  T *devCP = static_cast<T *>(mat->devC);
  T * matA = devAP;
  T * matB = devBP;
  T * matC = devCP;

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    // clang-format off
    stat = func(handle, transA, transB, m, n, k, alphaP, 
                matA, lda, stride_a,
                matB, ldb, stride_b, betaP, 
                matC, ldc, stride_c, batchct);
    // clang-format on
    matA = getOffsetPtrNC(devAP, blockstrideA, rep);
    matB = getOffsetPtrNC(devBP, blockstrideB, rep);
    matC = getOffsetPtrNC(devCP, blockstrideC, rep);
    // Check for errors during the gemm run
    checkRocblas(stat);
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
    // clang-format off
    stat = func(handle, transA, transB, m, n, k, alphaP, 
                matA, lda, stride_a,
                matB, ldb, stride_b, betaP, 
                matC, ldc, stride_c, batchct);
    // clang-format on
    matA = getOffsetPtrNC(devAP, blockstrideA, rep);
    matB = getOffsetPtrNC(devBP, blockstrideB, rep);
    matC = getOffsetPtrNC(devCP, blockstrideC, rep);
  }
  hipEventRecord(stop, stream);
  hipEventSynchronize(stop);

  // Check for errors during the performance test
  checkRocblas(stat);
  checkHip(hipGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  hipEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculateFOM(static_cast<double>(elapsedTime_ms));
}


inline void * rocblasGemm::getOffsetPtr(void * mat, long long int blockstride, int rep, int blockct, rocblas_datatype type) {
  return (void*)(((char*)mat) + (rep%blockct)*blockstride*typeCallDev<sizeofCUDT>(type));
}

void rocblasGemm::test_gemm_ex(rocblasgemmInst *mat) {
  rocblas_status stat;
  rocblas_handle handle;
  hipStream_t stream;
  checkHip(hipSetDevice(mat->devIDX));
  checkRocblas(rocblas_create_handle(&handle));
  checkHip(hipStreamCreate(&stream));
  checkRocblas(rocblas_set_stream(handle, stream));
  checkRocblas(rocblas_set_workspace(handle, mat->devWork, mat->wSZ));
  int blockstrideA = m * n;
  int blockstrideB = n * k;
  int blockstrideC = m * k;
  void * matA = mat->devA;
  void * matB = mat->devB;
  void * matC = mat->devC;
  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    // clang-format off
    stat = rocblas_gemm_ex(handle, transA, transB, m, n, k, alpha, 
                           matA, a_type, lda, 
                           matB, b_type, ldb, beta, 
                           matC, c_type, ldc, 
                           matC, c_type, ldc, compute,
                           rocblas_gemm_algo_standard, 0, 0);
    // clang-format on
    matA = getOffsetPtr(mat->devA, blockstrideA, rep, blockct, a_type);
    matB = getOffsetPtr(mat->devB, blockstrideB, rep, blockct, b_type);
    matC = getOffsetPtr(mat->devC, blockstrideC, rep, blockct, c_type);
    // Check for errors during the gemm run
    checkRocblas(stat);
    checkHip(hipGetLastError());
  }
  hipStreamSynchronize(stream);

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  /*
    Run and time the performance test
  */
  matA = mat->devA;
  matB = mat->devB;
  matC = mat->devC;
  hipEventRecord(start, stream);
  for (int rep = 0; rep < iters; rep++) {
    // clang-format off
    stat = rocblas_gemm_ex(handle, transA, transB, m, n, k, alpha, 
                           matA, a_type, lda, 
                           matB, b_type, ldb, beta, 
                           matC, c_type, ldc, 
                           matC, c_type, ldc, compute,
                           rocblas_gemm_algo_standard, 0, 0);
    // clang-format on
    matA = getOffsetPtr(mat->devA, blockstrideA, rep, blockct, a_type);
    matB = getOffsetPtr(mat->devB, blockstrideB, rep, blockct, b_type);
    matC = getOffsetPtr(mat->devC, blockstrideC, rep, blockct, c_type);
  }  
  hipEventRecord(stop, stream);
  hipEventSynchronize(stop);

  // Check for errors during the performance test
  checkRocblas(stat);
  checkHip(hipGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  hipEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculateFOM(static_cast<double>(elapsedTime_ms));
}


