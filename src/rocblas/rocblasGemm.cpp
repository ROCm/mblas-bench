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
    // {rocblas_datatype_f64_r,    rocblas_datatype_f64_r,   rocblas_datatype_f64_r,   rocblas_datatype_f64_r  },
    // {rocblas_datatype_f32_r,    rocblas_datatype_f32_r,   rocblas_datatype_f32_r,   rocblas_datatype_f32_r  },
    // {rocblas_datatype_f16_r,    rocblas_datatype_f16_r,   rocblas_datatype_f16_r,   rocblas_datatype_f16_r  },
    // {rocblas_datatype_f32_r,    rocblas_datatype_f32_r,   rocblas_datatype_f16_r,   rocblas_datatype_f16_r  },
    // {rocblas_datatype_f32_r,    rocblas_datatype_f32_r,   rocblas_datatype_f16_r,   rocblas_datatype_f32_r  },
    // {rocblas_datatype_f32_r,    rocblas_datatype_f32_r,   rocblas_datatype_bf16_r,  rocblas_datatype_bf16_r },
    // {rocblas_datatype_f32_r,    rocblas_datatype_f32_r,   rocblas_datatype_bf16_r,  rocblas_datatype_f32_r  },
    // {rocblas_datatype_i32_r,    rocblas_datatype_i32_r,   rocblas_datatype_i8_r,    rocblas_datatype_i32_r  }, 
    // {rocblas_datatype_f32_c,    rocblas_datatype_f32_c,   rocblas_datatype_f32_c,   rocblas_datatype_f32_c  },
    // {rocblas_datatype_f64_c,    rocblas_datatype_f64_c,   rocblas_datatype_f64_c,   rocblas_datatype_f64_c  },
    // Compute/Scale Type     A/B Type    C Type
    {MBLAS_COMPUTE_64F,   MBLAS_R_64F,  MBLAS_R_64F,  MBLAS_R_64F },
    {MBLAS_COMPUTE_64F,   MBLAS_C_64F,  MBLAS_C_64F,  MBLAS_C_64F },
    {MBLAS_COMPUTE_32F,   MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F },
    {MBLAS_COMPUTE_32F,   MBLAS_C_32F,  MBLAS_C_32F,  MBLAS_C_32F },
    {MBLAS_COMPUTE_16F,   MBLAS_R_16F,  MBLAS_R_16F,  MBLAS_R_16F },
    {MBLAS_COMPUTE_32F,   MBLAS_R_32F,  MBLAS_R_16F,  MBLAS_R_32F },
    {MBLAS_COMPUTE_32F,   MBLAS_R_32F,  MBLAS_R_16F,  MBLAS_R_16F },
    {MBLAS_COMPUTE_32F,   MBLAS_R_32F,  MBLAS_R_16BF, MBLAS_R_32F },
    {MBLAS_COMPUTE_32F,   MBLAS_R_32F,  MBLAS_R_16BF, MBLAS_R_16BF},
    {MBLAS_COMPUTE_32I,   MBLAS_R_32I,  MBLAS_R_8I,   MBLAS_R_32I },

};
// clang-format on

std::vector<TgemmPrecTypeAMD> rocblasGemm::TgemmExSupported = {};

void rocblasGemm::init_prec_map() {}

void rocblasGemm::parse_dev_iters(std::string deviceStr) {
  // Parse device
  std::stringstream ss(deviceStr);
  while (ss.good()) {
    string deviceSStr;
    getline(ss, deviceSStr, ',');
    int devInt = stoi(deviceSStr);
    rocblasgemmInst val = rocblasgemmInst(devInt);
    mat_ptrs.push_back(val);
  }
}

void rocblasGemm::parse_problem_type(string computeTStr, string scalarTStr, string aStr,
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
  a_type = mblasRocDataType(aStr);
  b_type = mblasRocDataType(bStr);
  c_type = mblasRocDataType(cStr);

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
  // check_rocblas(rocblas_create_handle(&handle));
  init_prec_map();
  // Grab precision from command line
  precision = mblasRocDataType(result["precision"].as<string>());
  // Grab compute type from command line
  string computeT = result["compute_type"].as<string>();
  string scalarT = result["scalar_type"].as<string>();
  string aT = result["a_type"].as<string>();
  string bT = result["b_type"].as<string>();
  string cT = result["c_type"].as<string>();
  parse_problem_type(computeT, scalarT, aT, bT, cT);

  parse_dev_iters(result["device"].as<string>());
  std::string tA = result["transposeA"].as<std::string>();
  std::string tB = result["transposeB"].as<std::string>();
  transA = mblasRocOperation(result["transposeA"].as<std::string>());
  transB = mblasRocOperation(result["transposeB"].as<std::string>());

  // Pull in alpha and beta, alloc memory and save to pointers
  string salpha = result["alpha"].as<string>();
  string salphai = result["alphai"].as<string>();
  alpha =
      type_call_host<allocSetScalar>(precision, salpha.c_str(), salphai.c_str());
  string sbeta = result["beta"].as<string>();
  string sbetai = result["betai"].as<string>();
  beta = type_call_host<allocSetScalar>(precision, sbeta.c_str(), sbetai.c_str());

  // std::cout << *((float *)alpha) << std::endl;
  // std::cout << *((float *)beta) << std::endl;
}

string rocblasGemm::prepare_array() {
  // std::cout << "Pre Convert: " << *((float *)alpha) << std::endl;
  // alpha = convert_scalar(scalar, alpha);
  // std::cout << "Post Convert: " << __half2float(*(__half *)alpha) <<
  // std::endl; beta = convert_scalar(scalar, beta);
  this->alloc_host();
  this->fill_host();

  int num_devices;
  hipGetDeviceCount(&num_devices);
  // Check range of devices here
  // This implementation may not work if
  // CUDA_VISIBLE_DEVICES is set to something weird
  for (auto &instance : mat_ptrs) {
    if (instance.devIDX >= num_devices) {
      string errorString =
          "Invalid device id"
          "\nNumber of detected devices: " +
          std::to_string(num_devices) +
          "\nDevice selection:           " + std::to_string(instance.devIDX);
      throw std::invalid_argument(errorString);
    }
  }
  // for (auto &instance : mat_ptrs) {
  //  this->alloc_dev(&instance);
  //  this->copy_host_to_dev(&instance);
  //}

  run_threaded(&rocblasGemm::alloc_dev);
  run_threaded(&rocblasGemm::copy_host_to_dev);
  std::ostringstream ossHeader;
  ossHeader << "transA_option,transB_option,M,N,K,lda,ldb,ldc,";
  if (batched) {
    ossHeader << "batch_count,";
  }
  ossHeader << "rocBLAS-Gflops,rocBLAS-GB/s,rocBLAS-us," << endl;
  return ossHeader.str();
}

void rocblasGemm::run_threaded(void (rocblasGemm::*func)(rocblasgemmInst *)) {
  vector<thread> threads;
  for (auto &instance : mat_ptrs) {
    threads.push_back(thread(func, this, &instance));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void rocblasGemm::alloc_host() {
  // auto resultA = std::async(allocate_host_array, a_type, m, k, batch_count);
  // auto resultB = std::async(allocate_host_array, b_type, k, n, batch_count);
  // auto resultC = std::async(allocate_host_array, c_type, n, m, batch_count);
  // host_a = resultA.get();
  // host_b = resultB.get();
  // host_c = resultC.get();
  host_a = allocate_host_array(a_type, rows_mem_a, cols_mem_a, batch_count);
  host_b = allocate_host_array(b_type, rows_mem_b, cols_mem_b, batch_count);
  host_c = allocate_host_array(c_type, rows_mem_c, cols_mem_c, batch_count);
}

void rocblasGemm::alloc_dev(rocblasgemmInst *mat) {
  hipSetDevice(mat->devIDX);
  mat->devA = allocate_dev_array(a_type, rows_mem_a, cols_mem_a, batch_count);
  mat->devB = allocate_dev_array(b_type, rows_mem_b, cols_mem_b, batch_count);
  mat->devC = allocate_dev_array(c_type, rows_mem_c, cols_mem_c, batch_count);
  mat->wSZ = workspace_size;
  hipMalloc(&mat->devWork, mat->wSZ);
}

void rocblasGemm::fill_host() {
  // Some random functions treat the matrix as a vectors, some require a matrix
  // vector<thread> threads;
  // threads.push_back(thread(initHostH, a_type, initialization, host_a, m, k,
  // lda,
  //                         batch_count, stride_a, 2.f, false));
  // threads.push_back(thread(initHostH, b_type, initialization, host_b, k, n,
  // ldb,
  //                         batch_count, stride_b, 3.f, true));
  // threads.push_back(thread(initHostH, c_type, initialization, host_c, m, n,
  // ldc,
  //                         batch_count, stride_c, 0.f, false));
  // for (auto &thread : threads) {
  //  thread.join();
  //}

  type_call_host<initHost>(a_type, initialization, host_a, rows_a, cols_a, lda,
                         batch_count, stride_a, control_a, constant_a, filename_a);
  type_call_host<initHost>(b_type, initialization, host_b, rows_b, cols_b, ldb,
                         batch_count, stride_b, control_b, constant_b, filename_b);
  type_call_host<initHost>(c_type, initialization, host_c, rows_c, cols_c, ldc,
                         batch_count, stride_c, control_c, constant_c, filename_c);
}

void rocblasGemm::copy_host_to_dev(rocblasgemmInst *mat) {
  hipSetDevice(mat->devIDX);
  copy_and_convert(a_type, host_a, mat->devA, rows_mem_a, cols_mem_a, batch_count);
  copy_and_convert(b_type, host_b, mat->devB, rows_mem_b, cols_mem_b, batch_count);
  copy_and_convert(c_type, host_c, mat->devC, rows_mem_c, cols_mem_c, batch_count);
  if (batched && !strided) {
    // Perform some pointer arithmetic to calculate the arrays we pass to the
    // gpu
    mat->ptr_host_a =
        (void **)malloc(batch_count * type_call_host<sizeofCUDTP>(a_type));
    mat->ptr_host_b =
        (void **)malloc(batch_count * type_call_host<sizeofCUDTP>(b_type));
    mat->ptr_host_c =
        (void **)malloc(batch_count * type_call_host<sizeofCUDTP>(c_type));
    check_hip(
        hipMalloc(&mat->ptr_dev_a, batch_count * type_call_host<sizeofCUDTP>(a_type)));
    check_hip(
        hipMalloc(&mat->ptr_dev_b, batch_count * type_call_host<sizeofCUDTP>(b_type)));
    check_hip(
        hipMalloc(&mat->ptr_dev_c, batch_count * type_call_host<sizeofCUDTP>(c_type)));
    type_call_dev<batchedPtrMagic>(a_type, mat->ptr_host_a, mat->ptr_dev_a, mat->devA,
                                batch_count, rows_mem_a, cols_mem_a);
    type_call_dev<batchedPtrMagic>(b_type, mat->ptr_host_b, mat->ptr_dev_b, mat->devB,
                                batch_count, rows_mem_b, cols_mem_b);
    type_call_dev<batchedPtrMagic>(c_type, mat->ptr_host_c, mat->ptr_dev_c, mat->devC,
                                batch_count, rows_mem_c, cols_mem_c);
  }
}

void rocblasGemm::free_mem() {
  free(alpha);
  free(beta);
  free(host_a);
  free(host_b);
  free(host_c);
  for (auto mat : mat_ptrs) {
    hipFree(mat.devA);
    hipFree(mat.devB);
    hipFree(mat.devC);
    hipFree(mat.devWork);
    if (batched && !strided) {
      free(mat.ptr_host_a);
      free(mat.ptr_host_b);
      free(mat.ptr_host_c);
      hipFree(mat.ptr_dev_a);
      hipFree(mat.ptr_dev_b);
      hipFree(mat.ptr_dev_c);
    }
  }
}

double rocblasGemm::test() {
  vector<thread> threads;
  double gflops = 0.0;
  for (auto &mat : mat_ptrs) {
    // Tgemm
    if ((function == "rocblas_dgemm" && precision == rocblas_datatype_f64_r) ||
        (function == "gemm" && precision == rocblas_datatype_f64_r)) {
      std::function<decltype(rocblas_dgemm)> dgemm_var = rocblas_dgemm;
      threads.push_back(
          thread(&rocblasGemm::test_Tgemm<double>, this, dgemm_var, &mat));
    } else if ((function == "rocblas_sgemm" && precision == rocblas_datatype_f32_r) ||
               (function == "gemm" && precision == rocblas_datatype_f32_r)) {
      std::function<decltype(rocblas_sgemm)> sgemm_var = rocblas_sgemm;
      threads.push_back(
          thread(&rocblasGemm::test_Tgemm<float>, this, sgemm_var, &mat));
    } else if ((function == "rocblas_hgemm" && precision == rocblas_datatype_f16_r) ||
               (function == "gemm" && precision == rocblas_datatype_f16_r)) {
      std::function<decltype(rocblas_hgemm)> hgemm_var = rocblas_hgemm;
      threads.push_back(
          thread(&rocblasGemm::test_Tgemm<rocblas_half>, this, hgemm_var, &mat));
    } else if ((function == "rocblas_zgemm" && precision == rocblas_datatype_f64_c) ||
               (function == "gemm" && precision == rocblas_datatype_f64_c)) {
      std::function<decltype(rocblas_zgemm)> zgemm_var = rocblas_zgemm;
      threads.push_back(thread(&rocblasGemm::test_Tgemm<rocblas_complex_num<double>>, this,
                               zgemm_var, &mat));
    } else if ((function == "rocblas_cgemm" && precision == rocblas_datatype_f32_c) ||
               (function == "gemm" && precision == rocblas_datatype_f32_c)) {
      std::function<decltype(rocblas_cgemm)> cgemm_var = rocblas_cgemm;
      threads.push_back(
          thread(&rocblasGemm::test_Tgemm<rocblas_complex_num<float>>, this, cgemm_var, &mat));
    }
    // TgemmBatched
    else if (function == "rocblas_dgemm_batched" && precision == rocblas_datatype_f64_r) {
      std::function<decltype(rocblas_dgemm_batched)> dgemm_var =
          rocblas_dgemm_batched;
      threads.push_back(
          thread(&rocblasGemm::test_Tgemm_batched<double>, this, dgemm_var, &mat));
    } else if (function == "rocblas_sgemm_batched" && precision == rocblas_datatype_f32_r) {
      std::function<decltype(rocblas_sgemm_batched)> sgemm_var =
          rocblas_sgemm_batched;
      threads.push_back(
          thread(&rocblasGemm::test_Tgemm_batched<float>, this, sgemm_var, &mat));
    } else if (function == "rocblas_hgemm_batched" && precision == rocblas_datatype_f16_r) {
      std::function<decltype(rocblas_hgemm_batched)> hgemm_var =
          rocblas_hgemm_batched;
      threads.push_back(
          thread(&rocblasGemm::test_Tgemm_batched<rocblas_half>, this, hgemm_var, &mat));
    } else if (function == "rocblas_zgemm_batched" && precision == rocblas_datatype_f64_c) {
      std::function<decltype(rocblas_zgemm_batched)> zgemm_var =
          rocblas_zgemm_batched;
      threads.push_back(thread(&rocblasGemm::test_Tgemm_batched<rocblas_complex_num<double>>,
                               this, zgemm_var, &mat));
    } else if (function == "rocblas_cgemm_batched" && precision == rocblas_datatype_f32_c) {
      std::function<decltype(rocblas_cgemm_batched)> cgemm_var =
          rocblas_cgemm_batched;
      threads.push_back(thread(&rocblasGemm::test_Tgemm_batched<rocblas_complex_num<float>>, this,
                               cgemm_var, &mat));
    }
    // TgemmStridedBatched
    else if (function == "rocblas_dgemm_strided_batched" &&
             precision == rocblas_datatype_f64_r) {
      std::function<decltype(rocblas_dgemm_strided_batched)> dgemm_var =
          rocblas_dgemm_strided_batched;
      threads.push_back(thread(&rocblasGemm::test_Tgemm_strided_batched<double>,
                               this, dgemm_var, &mat));
    } else if (function == "rocblas_sgemm_strided_batched" &&
               precision == rocblas_datatype_f32_r) {
      std::function<decltype(rocblas_sgemm_strided_batched)> sgemm_var =
          rocblas_sgemm_strided_batched;
      threads.push_back(thread(&rocblasGemm::test_Tgemm_strided_batched<float>,
                               this, sgemm_var, &mat));
    } else if (function == "rocblas_hgemm_strided_batched" &&
               precision == rocblas_datatype_f16_r) {
      std::function<decltype(rocblas_hgemm_strided_batched)> hgemm_var =
          rocblas_hgemm_strided_batched;
      threads.push_back(thread(&rocblasGemm::test_Tgemm_strided_batched<rocblas_half>,
                               this, hgemm_var, &mat));
    } else if (function == "rocblas_zgemm_strided_batched" &&
               precision == rocblas_datatype_f64_c) {
      std::function<decltype(rocblas_zgemm_strided_batched)> zgemm_var =
          rocblas_zgemm_strided_batched;
      threads.push_back(
          thread(&rocblasGemm::test_Tgemm_strided_batched<rocblas_complex_num<double>>, this,
                 zgemm_var, &mat));
    } else if (function == "rocblas_cgemm_strided_batched" &&
               precision == rocblas_datatype_f32_c) {
      std::function<decltype(rocblas_cgemm_strided_batched)> cgemm_var =
          rocblas_cgemm_strided_batched;
      threads.push_back(thread(&rocblasGemm::test_Tgemm_strided_batched<rocblas_complex_num<float>>,
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
      begin(mat_ptrs), end(mat_ptrs), 0.0,
      [](double i, const rocblasgemmInst &o) { return o.gflops + i; });

  gbyte_per_second = std::accumulate(
      begin(mat_ptrs), end(mat_ptrs), 0.0,
      [](double i, const rocblasgemmInst &o) { return o.gbytes + i; });

  iter_time_us = std::accumulate(begin(mat_ptrs), end(mat_ptrs), 0.0,
                                 [](double i, const rocblasgemmInst &o) {
                                   return o.time_us + i;
                                 }) /
                 mat_ptrs.size();

  return gflop_per_second;
}

std::string rocblasGemm::get_result_string() {
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

std::tuple<double, double, double> rocblasGemm::calculate_figure_of_merit(
    double totalTime_ms) {
  double avgTime_ms = totalTime_ms / iters;
  double avgTime_s = avgTime_ms / 1000.0f;
  double avgTime_us = avgTime_ms * 1000.0f;

  int a_sz = type_call_dev<sizeofCUDT>(a_type);
  int b_sz = type_call_dev<sizeofCUDT>(b_type);
  int c_sz = type_call_dev<sizeofCUDT>(c_type);

  int flopPerSize = 2;
  if (precision.isReal()) {
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
void rocblasGemm::test_Tgemm(std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const*, int, T const*, int, T const*, T*, int)> func, rocblasgemmInst *mat) {
  rocblas_status stat;
  rocblas_handle handle;
  hipStream_t stream;
  check_hip(hipSetDevice(mat->devIDX));
  check_rocblas(rocblas_create_handle(&handle));
  check_hip(hipStreamCreate(&stream));
  check_rocblas(rocblas_set_stream(handle, stream));
  // check_rocblas(rocblas_set_workspace(handle, mat->devWork, mat->wSZ));

  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T *devAP = static_cast<T *>(mat->devA);
  T *devBP = static_cast<T *>(mat->devB);
  T *devCP = static_cast<T *>(mat->devC);

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    // clang-format off
    stat = func(handle, transA.convertToRocm(), transB.convertToRocm(), m, n, k, alphaP, 
               devAP, lda, 
               devBP, ldb, betaP, 
               devCP, ldc);
    // clang-format on
    // Check for errors during the gemm run
    check_rocblas(stat);
    check_hip(hipGetLastError());
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
    stat = func(handle, transA.convertToRocm(), transB.convertToRocm(), m, n, k, alphaP, 
               devAP, lda, 
               devBP, ldb, betaP, 
               devCP, ldc);
    // clang-format on
  }
  hipEventRecord(stop, stream);
  hipEventSynchronize(stop);

  // Check for errors during the performance test
  check_rocblas(stat);
  check_hip(hipGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  hipEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculate_figure_of_merit(static_cast<double>(elapsedTime_ms));
}

template <typename T>
void rocblasGemm::test_Tgemm_batched(std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const* const*, int, T const* const*, int, T const*, T* const*, int, int)> func, rocblasgemmInst *mat) {
  rocblas_status stat;
  rocblas_handle handle;
  hipStream_t stream;
  check_hip(hipSetDevice(mat->devIDX));
  check_rocblas(rocblas_create_handle(&handle));
  check_hip(hipStreamCreate(&stream));
  check_rocblas(rocblas_set_stream(handle, stream));
  // check_rocblas(rocblas_set_workspace(handle, mat->devWork, mat->wSZ));

  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T **devAP = reinterpret_cast<T **>(mat->ptr_dev_a);
  T **devBP = reinterpret_cast<T **>(mat->ptr_dev_b);
  T **devCP = reinterpret_cast<T **>(mat->ptr_dev_c);

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    stat = func(handle, transA.convertToRocm(), transB.convertToRocm(), m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc, batch_count);

    // Check for errors during the gemm run
    check_rocblas(stat);
    check_hip(hipGetLastError());
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
    stat = func(handle, transA.convertToRocm(), transB.convertToRocm(), m, n, k, alphaP, devAP, lda, devBP, ldb,
                betaP, devCP, ldc, batch_count);
  }
  hipEventRecord(stop, stream);
  hipEventSynchronize(stop);

  // Check for errors during the performance test
  check_rocblas(stat);
  check_hip(hipGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  hipEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculate_figure_of_merit(static_cast<double>(elapsedTime_ms));
}

template <typename T>
void rocblasGemm::test_Tgemm_strided_batched(
          std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const*, int, long, T const*, int, long, T const*, T*, int, long, int)>
          func, rocblasgemmInst *mat) {
  rocblas_status stat;
  rocblas_handle handle;
  hipStream_t stream;
  check_hip(hipSetDevice(mat->devIDX));
  check_rocblas(rocblas_create_handle(&handle));
  check_hip(hipStreamCreate(&stream));
  check_rocblas(rocblas_set_stream(handle, stream));
  // check_rocblas(rocblas_set_workspace(handle, mat->devWork, mat->wSZ));

  T *alphaP = static_cast<T *>(alpha);
  T *betaP = static_cast<T *>(beta);
  T *devAP = static_cast<T *>(mat->devA);
  T *devBP = static_cast<T *>(mat->devB);
  T *devCP = static_cast<T *>(mat->devC);

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    // clang-format off
    stat = func(handle, transA.convertToRocm(), transB.convertToRocm(), m, n, k, alphaP, 
                devAP, lda, stride_a,
                devBP, ldb, stride_b, betaP, 
                devCP, ldc, stride_c, batch_count);
    // clang-format on
    // Check for errors during the gemm run
    check_rocblas(stat);
    check_hip(hipGetLastError());
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
    stat = func(handle, transA.convertToRocm(), transB.convertToRocm(), m, n, k, alphaP, 
                devAP, lda, stride_a,
                devBP, ldb, stride_b, betaP, 
                devCP, ldc, stride_c, batch_count);
    // clang-format on
  }
  hipEventRecord(stop, stream);
  hipEventSynchronize(stop);

  // Check for errors during the performance test
  check_rocblas(stat);
  check_hip(hipGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  hipEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculate_figure_of_merit(static_cast<double>(elapsedTime_ms));
}

void rocblasGemm::test_gemm_ex(rocblasgemmInst *mat) {
  rocblas_status stat;
  rocblas_handle handle;
  hipStream_t stream;
  check_hip(hipSetDevice(mat->devIDX));
  check_rocblas(rocblas_create_handle(&handle));
  check_hip(hipStreamCreate(&stream));
  check_rocblas(rocblas_set_stream(handle, stream));
  check_rocblas(rocblas_set_workspace(handle, mat->devWork, mat->wSZ));

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    // clang-format off
    stat = rocblas_gemm_ex(handle, transA.convertToRocm(), transB.convertToRocm(), m, n, k, alpha, 
                           mat->devA, a_type, lda, 
                           mat->devB, b_type, ldb, beta, 
                           mat->devC, c_type, ldc, 
                           mat->devC, c_type, ldc, compute,
                           rocblas_gemm_algo_standard, 0, 0);
    // clang-format on
    // Check for errors during the gemm run
    check_rocblas(stat);
    check_hip(hipGetLastError());
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
    stat = rocblas_gemm_ex(handle, transA.convertToRocm(), transB.convertToRocm(), m, n, k, alpha, 
                           mat->devA, a_type, lda, 
                           mat->devB, b_type, ldb, beta, 
                           mat->devC, c_type, ldc, 
                           mat->devC, c_type, ldc, compute,
                           rocblas_gemm_algo_standard, 0, 0);
    // clang-format on
  }  
  hipEventRecord(stop, stream);
  hipEventSynchronize(stop);

  // Check for errors during the performance test
  check_rocblas(stat);
  check_hip(hipGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  hipEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculate_figure_of_merit(static_cast<double>(elapsedTime_ms));
}


