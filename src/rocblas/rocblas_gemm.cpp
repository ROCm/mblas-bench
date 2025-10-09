#include "rocblas_gemm.h"

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

#include "hip_convert.h"
#include "hip_create_allocate.h"
#include "hip_datatype_utils.h"
#include "hip_error.h"
#include "cxxopts.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::move;
using std::string;
using std::thread;
using std::vector;

// clang-format off
std::vector<gemmPrecTypeAMD> rocblas_gemm::gemm_ex_supported = {
    // Compute            Scale Type    A/B Type      C/D Type
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

std::vector<TgemmPrecTypeAMD> rocblas_gemm::Tgemm_ex_supported = {};

void rocblas_gemm::init_prec_map() {}

void rocblas_gemm::parse_dev_iters(std::string deviceStr) {
  // Parse device
  std::stringstream ss(deviceStr);
  while (ss.good()) {
    string deviceSStr;
    getline(ss, deviceSStr, ',');
    int devInt = stoi(deviceSStr);
    rocblas_gemm_inst val = rocblas_gemm_inst(devInt);
    mat_ptrs.push_back(val);
  }
}

void rocblas_gemm::parse_problem_type(string computeTStr, string scalarTStr, string aStr,
                             string bStr, string cStr, string dStr) {
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
    inplace = true;
    return;
  }

  if (dStr == "") {
    // Assume the user means C = D, so also establish that here
    dStr = cStr;
    inplace = true;
  }
  // Parse each precision
  a_type = mblas_rocblas_data_type(aStr);
  b_type = mblas_rocblas_data_type(bStr);
  c_type = mblas_rocblas_data_type(cStr);
  d_type = mblas_rocblas_data_type(dStr);

  // Validate against supported precision table (fun)
  if (a_type != b_type) {
    string errorString = "A Type must the same as B Type";
    throw std::invalid_argument(errorString);
  }
  if (c_type != d_type) {
    string errorString = "C Type must the same as D Type";
    throw std::invalid_argument(errorString);
  }
  if (function.find("gemm_ex") || function.find("gemm_batched_ex") || function.find("gemm_strided_batched_ex")) {
    /*
      Possible functions:
        rocblas_gemm_ex
        rocblas_gemm_batched_ex
        rocblas_gemm_strided_batched_ex
    */

    gemmPrecTypeAMD selType = {compute, scalar, a_type, c_type};
    auto result =
        std::find(begin(gemm_ex_supported), end(gemm_ex_supported), selType);
    if (result == end(gemm_ex_supported)) {
      // Unable to find matching config, not supported
      string errorString =
          "Invalid GEMM specification for gemm_ex.  Combination of parameters "
          "not supported"
          "\nCompute type: " +
          computeTStr + "\nScalar type: " + scalarTStr + "\nA type: " + aStr +
          "\nB type: " + bStr + "\nC type: " + cStr;
      throw std::invalid_argument(errorString);
    }
   
  } else {
    // All other rocblas gemm functions use inplace
    inplace = true;
  }
}

rocblas_gemm::rocblas_gemm(cxxopts::ParseResult result) : generic_gemm(result) {
  // rocblas_create_handle(&handle);
  // check_rocblas(rocblas_create_handle(&handle));
  init_prec_map();
  // Grab precision from command line
  precision = mblas_rocblas_data_type(result["precision"].as<string>());
  // Grab compute type from command line
  string computeT = result["compute_type"].as<string>();
  string scalarT = result["scalar_type"].as<string>();
  string aT = result["a_type"].as<string>();
  string bT = result["b_type"].as<string>();
  string cT = result["c_type"].as<string>();
  string dT = result["c_type"].as<string>();
  parse_problem_type(computeT, scalarT, aT, bT, cT, dT);

  parse_dev_iters(result["device"].as<string>());
  std::string tA = result["transposeA"].as<std::string>();
  std::string tB = result["transposeB"].as<std::string>();
  transA = mblas_rocblas_operation(result["transposeA"].as<std::string>());
  transB = mblas_rocblas_operation(result["transposeB"].as<std::string>());

  // Pull in alpha and beta, alloc memory and save to pointers
  string salpha = result["alpha"].as<string>();
  string salphai = result["alphai"].as<string>();
  alpha =
      type_call_host<allocSetScalar>(precision, salpha.c_str(), salphai.c_str());
  string sbeta = result["beta"].as<string>();
  string sbetai = result["betai"].as<string>();
  beta = type_call_host<allocSetScalar>(precision, sbeta.c_str(), sbetai.c_str());

  set_flush_batch_count( 
      type_call_dev<sizeofCUDT>(a_type), type_call_dev<sizeofCUDT>(b_type), 
      type_call_dev<sizeofCUDT>(c_type), type_call_dev<sizeofCUDT>(d_type), 
      get_packing_count(a_type), 
      get_packing_count(b_type), 
      get_packing_count(c_type), 
      get_packing_count(d_type), 
      inplace);
  // TODO: does rocblas_gemm return multiple algorithms like cublaslt_gemm?
  requested_solution_count = result["requested_solution_num"].as<int>();
}

string rocblas_gemm::prepare_array() {
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

  run_threaded(&rocblas_gemm::alloc_dev);
  run_threaded(&rocblas_gemm::copy_host_to_dev);
  returned_algo_count = 1; // rocblas only returns one solution
  std::ostringstream ossHeader;
  ossHeader << "transA_option,transB_option,M,N,K,lda,ldb,ldc,";
  if (batched) {
    ossHeader << "batch_count,";
  }
  ossHeader << "rocBLAS-Gflops,rocBLAS-GB/s,rocBLAS-us," << endl;
  return ossHeader.str();
}

void rocblas_gemm::run_threaded(void (rocblas_gemm::*func)(rocblas_gemm_inst *)) {
  vector<thread> threads;
  for (auto &instance : mat_ptrs) {
    threads.push_back(thread(func, this, &instance));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void rocblas_gemm::alloc_host() {
  ptr_host_a =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(a_type));
  ptr_host_b =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(b_type));
  ptr_host_c =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(c_type));
  ptr_host_d =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(d_type));

  for (int i = 0; i < flush_batch_count; i++) {
    ptr_host_a[i] = allocate_host_array(a_type, rows_mem_a, cols_mem_a, batch_count);
    ptr_host_b[i] = allocate_host_array(b_type, rows_mem_b, cols_mem_b, batch_count);
    ptr_host_c[i] = allocate_host_array(c_type, rows_mem_c, cols_mem_c, batch_count);
    ptr_host_d[i] = allocate_host_array(d_type, rows_mem_d, cols_mem_d, batch_count);
  }
}

void rocblas_gemm::alloc_dev(rocblas_gemm_inst *mat) {
  hipSetDevice(mat->devIDX);

  mat->ptr_dev_a =
      (void **)malloc(flush_batch_count * type_call_dev<sizeofCUDTP>(a_type));
  mat->ptr_dev_b =
      (void **)malloc(flush_batch_count * type_call_dev<sizeofCUDTP>(b_type));
  mat->ptr_dev_c =
      (void **)malloc(flush_batch_count * type_call_dev<sizeofCUDTP>(c_type));
  if (!inplace) {
    mat->ptr_dev_d =
        (void **)malloc(flush_batch_count * type_call_dev<sizeofCUDTP>(d_type));
  } else {
    mat->ptr_dev_d = mat->ptr_dev_c;
  }

  for (int i = 0; i < flush_batch_count; i++) {
    mat->ptr_dev_a[i] = allocate_dev_array(a_type, rows_mem_a, cols_mem_a, batch_count);
    mat->ptr_dev_b[i] = allocate_dev_array(b_type, rows_mem_b, cols_mem_b, batch_count);
    mat->ptr_dev_c[i] = allocate_dev_array(c_type, rows_mem_c, cols_mem_c, batch_count);
    mat->ptr_dev_d[i] = allocate_dev_array(d_type, rows_mem_d, cols_mem_d, batch_count);
  }

  mat->wSZ = workspace_size;
  hipMalloc(&mat->devWork, mat->wSZ);
}

void rocblas_gemm::fill_host() {
  for (int i = 0; i < flush_batch_count; i++){
    type_call_host<initHost>(a_type, initialization, ptr_host_a[i], rows_a, cols_a, lda,
                           batch_count, stride_a, control_a, constant_a, filename_a);
    type_call_host<initHost>(b_type, initialization, ptr_host_b[i], rows_b, cols_b, ldb,
                           batch_count, stride_b, control_b, constant_b, filename_b);
    type_call_host<initHost>(c_type, initialization, ptr_host_c[i], rows_c, cols_c, ldc,
                           batch_count, stride_c, control_c, constant_c, filename_c);
    // D is just output, don't need to init
  }
}

void rocblas_gemm::copy_host_to_dev(rocblas_gemm_inst *mat) {
  hipSetDevice(mat->devIDX);
  for (int i = 0; i < flush_batch_count; i++) {
    copy_and_convert(a_type, ptr_host_a[i], mat->ptr_dev_a[i], rows_mem_a, cols_mem_a, batch_count);
    copy_and_convert(b_type, ptr_host_b[i], mat->ptr_dev_b[i], rows_mem_b, cols_mem_b, batch_count);
    copy_and_convert(c_type, ptr_host_c[i], mat->ptr_dev_c[i], rows_mem_c, cols_mem_c, batch_count);
  }
}

void rocblas_gemm::free_mem() {
  free(alpha);
  free(beta);
  free(ptr_host_a);
  free(ptr_host_b);
  free(ptr_host_c);
  if (!inplace) {
    free(ptr_host_d);
  }
  for (auto mat : mat_ptrs) {
    hipFree(mat.ptr_dev_a);
    hipFree(mat.ptr_dev_b);
    hipFree(mat.ptr_dev_c);
    hipFree(mat.ptr_dev_d);
    hipFree(mat.devWork);
    // if (batched && !strided) {
    //   free(mat.ptr_host_a);
    //   free(mat.ptr_host_b);
    //   free(mat.ptr_host_c);
    //   hipFree(mat.ptr_dev_a);
    //   hipFree(mat.ptr_dev_b);
    //   hipFree(mat.ptr_dev_c);
    // }
  }
}

double rocblas_gemm::test(const int &ith_solution) {
  vector<thread> threads;
  double gflops = 0.0;
  for (auto &mat : mat_ptrs) {
    // Tgemm
    if ((function == "rocblas_dgemm" && precision == rocblas_datatype_f64_r) ||
        (function == "gemm" && precision == rocblas_datatype_f64_r)) {
      std::function<decltype(rocblas_dgemm)> dgemm_var = rocblas_dgemm;
      threads.push_back(
          thread(&rocblas_gemm::test_Tgemm<double>, this, dgemm_var, &mat));
    } else if ((function == "rocblas_sgemm" && precision == rocblas_datatype_f32_r) ||
               (function == "gemm" && precision == rocblas_datatype_f32_r)) {
      std::function<decltype(rocblas_sgemm)> sgemm_var = rocblas_sgemm;
      threads.push_back(
          thread(&rocblas_gemm::test_Tgemm<float>, this, sgemm_var, &mat));
    } else if ((function == "rocblas_hgemm" && precision == rocblas_datatype_f16_r) ||
               (function == "gemm" && precision == rocblas_datatype_f16_r)) {
      std::function<decltype(rocblas_hgemm)> hgemm_var = rocblas_hgemm;
      threads.push_back(
          thread(&rocblas_gemm::test_Tgemm<rocblas_half>, this, hgemm_var, &mat));
    } else if ((function == "rocblas_zgemm" && precision == rocblas_datatype_f64_c) ||
               (function == "gemm" && precision == rocblas_datatype_f64_c)) {
      std::function<decltype(rocblas_zgemm)> zgemm_var = rocblas_zgemm;
      threads.push_back(thread(&rocblas_gemm::test_Tgemm<rocblas_complex_num<double>>, this,
                               zgemm_var, &mat));
    } else if ((function == "rocblas_cgemm" && precision == rocblas_datatype_f32_c) ||
               (function == "gemm" && precision == rocblas_datatype_f32_c)) {
      std::function<decltype(rocblas_cgemm)> cgemm_var = rocblas_cgemm;
      threads.push_back(
          thread(&rocblas_gemm::test_Tgemm<rocblas_complex_num<float>>, this, cgemm_var, &mat));
    }
    // TgemmBatched
    // Disabled due to batched & rotating tensors not being implemented at the same time
    // else if (function == "rocblas_dgemm_batched" && precision == rocblas_datatype_f64_r) {
    //   std::function<decltype(rocblas_dgemm_batched)> dgemm_var =
    //       rocblas_dgemm_batched;
    //   threads.push_back(
    //       thread(&rocblas_gemm::test_Tgemm_batched<double>, this, dgemm_var, &mat));
    // } else if (function == "rocblas_sgemm_batched" && precision == rocblas_datatype_f32_r) {
    //   std::function<decltype(rocblas_sgemm_batched)> sgemm_var =
    //       rocblas_sgemm_batched;
    //   threads.push_back(
    //       thread(&rocblas_gemm::test_Tgemm_batched<float>, this, sgemm_var, &mat));
    // } else if (function == "rocblas_hgemm_batched" && precision == rocblas_datatype_f16_r) {
    //   std::function<decltype(rocblas_hgemm_batched)> hgemm_var =
    //       rocblas_hgemm_batched;
    //   threads.push_back(
    //       thread(&rocblas_gemm::test_Tgemm_batched<rocblas_half>, this, hgemm_var, &mat));
    // } else if (function == "rocblas_zgemm_batched" && precision == rocblas_datatype_f64_c) {
    //   std::function<decltype(rocblas_zgemm_batched)> zgemm_var =
    //       rocblas_zgemm_batched;
    //   threads.push_back(thread(&rocblas_gemm::test_Tgemm_batched<rocblas_complex_num<double>>,
    //                            this, zgemm_var, &mat));
    // } else if (function == "rocblas_cgemm_batched" && precision == rocblas_datatype_f32_c) {
    //   std::function<decltype(rocblas_cgemm_batched)> cgemm_var =
    //       rocblas_cgemm_batched;
    //   threads.push_back(thread(&rocblas_gemm::test_Tgemm_batched<rocblas_complex_num<float>>, this,
    //                            cgemm_var, &mat));
    // }
    // TgemmStridedBatched
    else if (function == "rocblas_dgemm_strided_batched" &&
             precision == rocblas_datatype_f64_r) {
      std::function<decltype(rocblas_dgemm_strided_batched)> dgemm_var =
          rocblas_dgemm_strided_batched;
      threads.push_back(thread(&rocblas_gemm::test_Tgemm_strided_batched<double>,
                               this, dgemm_var, &mat));
    } else if (function == "rocblas_sgemm_strided_batched" &&
               precision == rocblas_datatype_f32_r) {
      std::function<decltype(rocblas_sgemm_strided_batched)> sgemm_var =
          rocblas_sgemm_strided_batched;
      threads.push_back(thread(&rocblas_gemm::test_Tgemm_strided_batched<float>,
                               this, sgemm_var, &mat));
    } else if (function == "rocblas_hgemm_strided_batched" &&
               precision == rocblas_datatype_f16_r) {
      std::function<decltype(rocblas_hgemm_strided_batched)> hgemm_var =
          rocblas_hgemm_strided_batched;
      threads.push_back(thread(&rocblas_gemm::test_Tgemm_strided_batched<rocblas_half>,
                               this, hgemm_var, &mat));
    } else if (function == "rocblas_zgemm_strided_batched" &&
               precision == rocblas_datatype_f64_c) {
      std::function<decltype(rocblas_zgemm_strided_batched)> zgemm_var =
          rocblas_zgemm_strided_batched;
      threads.push_back(
          thread(&rocblas_gemm::test_Tgemm_strided_batched<rocblas_complex_num<double>>, this,
                 zgemm_var, &mat));
    } else if (function == "rocblas_cgemm_strided_batched" &&
               precision == rocblas_datatype_f32_c) {
      std::function<decltype(rocblas_cgemm_strided_batched)> cgemm_var =
          rocblas_cgemm_strided_batched;
      threads.push_back(thread(&rocblas_gemm::test_Tgemm_strided_batched<rocblas_complex_num<float>>,
                               this, cgemm_var, &mat));
    }
    
    // gemmEx
    // else if (strided && function == "rocblas_gemm_strided_batched_ex") {
    else if (strided && function == "rocblas_gemm_strided_batched_ex") {
      // Call the Gemm strided batched deployment script
    } else if (batched && function == "rocblas_gemm_batched_ex") {
      // Call the Gemm batched code
    } else if (function == "rocblas_gemm_ex" || function == "gemm_ex") {
      threads.push_back(thread(&rocblas_gemm::test_gemm_ex, this, &mat));
    }
  }
  // Wait on running jobs
  for (auto &thread : threads) {
    thread.join();
  }

  // Sum all gflops
  gflop_per_second = std::accumulate(
      begin(mat_ptrs), end(mat_ptrs), 0.0,
      [](double i, const rocblas_gemm_inst &o) { return o.gflops + i; });

  gbyte_per_second = std::accumulate(
      begin(mat_ptrs), end(mat_ptrs), 0.0,
      [](double i, const rocblas_gemm_inst &o) { return o.gbytes + i; });

  iter_time_us = std::accumulate(begin(mat_ptrs), end(mat_ptrs), 0.0,
                                 [](double i, const rocblas_gemm_inst &o) {
                                   return o.time_us + i;
                                 }) /
                 mat_ptrs.size();

  return gflop_per_second;
}

std::string rocblas_gemm::get_result_string() {
  std::ostringstream ossValues;
  ossValues << std::setprecision(7);
  ossValues << transA.to_string_short() << ',' << transB.to_string_short() << ',' << m
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

std::tuple<double, double, double> rocblas_gemm::calculate_figure_of_merit(
    double totalTime_ms) {
  double avgTime_ms = totalTime_ms / iters;
  double avgTime_s = avgTime_ms / 1000.0f;
  double avgTime_us = avgTime_ms * 1000.0f;

  int a_sz = type_call_dev<sizeofCUDT>(a_type);
  int b_sz = type_call_dev<sizeofCUDT>(b_type);
  int c_sz = type_call_dev<sizeofCUDT>(c_type);

  int flopPerSize = 2;
  if (precision.is_real()) {
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
void rocblas_gemm::test_Tgemm(std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const*, int, T const*, int, T const*, T*, int)> func, rocblas_gemm_inst *mat) {
  rocblas_status stat;
  rocblas_handle handle;
  hipStream_t stream;
  check_hip(hipSetDevice(mat->devIDX));
  check_rocblas(rocblas_create_handle(&handle));
  check_hip(hipStreamCreate(&stream));
  check_rocblas(rocblas_set_stream(handle, stream));
  // check_rocblas(rocblas_set_workspace(handle, mat->devWork, mat->wSZ));

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    // clang-format off
    int flush_index = rep % flush_batch_count;
    stat = func(handle, transA.convert_to_rocm(), transB.convert_to_rocm(), m, n, k, (T *) alpha, 
               (T *) mat->ptr_dev_a[flush_index], lda, 
               (T *) mat->ptr_dev_b[flush_index], ldb, (T *) beta, 
               (T *) mat->ptr_dev_c[flush_index], ldc);
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
    int flush_index = rep % flush_batch_count;
    stat = func(handle, transA.convert_to_rocm(), transB.convert_to_rocm(), m, n, k, (T *) alpha, 
               (T *) mat->ptr_dev_a[flush_index], lda, 
               (T *) mat->ptr_dev_b[flush_index], ldb, (T *) beta, 
               (T *) mat->ptr_dev_c[flush_index], ldc);
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

// Disabled due to batched & rotating tensors not being implemented at the same time
// template <typename T>
// void rocblas_gemm::test_Tgemm_batched(std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const* const*, int, T const* const*, int, T const*, T* const*, int, int)> func, rocblas_gemm_inst *mat) {
//   rocblas_status stat;
//   rocblas_handle handle;
//   hipStream_t stream;
//   check_hip(hipSetDevice(mat->devIDX));
//   check_rocblas(rocblas_create_handle(&handle));
//   check_hip(hipStreamCreate(&stream));
//   check_rocblas(rocblas_set_stream(handle, stream));
//   // check_rocblas(rocblas_set_workspace(handle, mat->devWork, mat->wSZ));
// 
//   T *alphaP = static_cast<T *>(alpha);
//   T *betaP = static_cast<T *>(beta);
//   T **devAP = reinterpret_cast<T **>(mat->ptr_dev_a);
//   T **devBP = reinterpret_cast<T **>(mat->ptr_dev_b);
//   T **devCP = reinterpret_cast<T **>(mat->ptr_dev_c);
// 
//   // Cold iters
//   for (int rep = 0; rep < cold_iters; rep++) {
//     stat = func(handle, transA.convert_to_rocm(), transB.convert_to_rocm(), m, n, k, alphaP, devAP, lda, devBP, ldb,
//                 betaP, devCP, ldc, batch_count);
// 
//     // Check for errors during the gemm run
//     check_rocblas(stat);
//     check_hip(hipGetLastError());
//   }
//   hipStreamSynchronize(stream);
// 
//   hipEvent_t start, stop;
//   hipEventCreate(&start);
//   hipEventCreate(&stop);
// 
//   /*
//     Run and time the performance test
//   */
//   hipEventRecord(start, stream);
//   for (int rep = 0; rep < iters; rep++) {
//     stat = func(handle, transA.convert_to_rocm(), transB.convert_to_rocm(), m, n, k, alphaP, devAP, lda, devBP, ldb,
//                 betaP, devCP, ldc, batch_count);
//   }
//   hipEventRecord(stop, stream);
//   hipEventSynchronize(stop);
// 
//   // Check for errors during the performance test
//   check_rocblas(stat);
//   check_hip(hipGetLastError());
// 
//   // Calculate and report GFlops
//   float elapsedTime_ms;
//   hipEventElapsedTime(&elapsedTime_ms, start, stop);
//   std::tie(mat->gflops, mat->gbytes, mat->time_us) =
//       calculate_figure_of_merit(static_cast<double>(elapsedTime_ms));
// }

template <typename T>
void rocblas_gemm::test_Tgemm_strided_batched(
          std::function<rocblas_status_(_rocblas_handle*, rocblas_operation_, rocblas_operation_, int, int, int, T const*, T const*, int, long, T const*, int, long, T const*, T*, int, long, int)>
          func, rocblas_gemm_inst *mat) {
  rocblas_status stat;
  rocblas_handle handle;
  hipStream_t stream;
  check_hip(hipSetDevice(mat->devIDX));
  check_rocblas(rocblas_create_handle(&handle));
  check_hip(hipStreamCreate(&stream));
  check_rocblas(rocblas_set_stream(handle, stream));
  // check_rocblas(rocblas_set_workspace(handle, mat->devWork, mat->wSZ));

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    // clang-format off
    int flush_index = rep % flush_batch_count;
    stat = func(handle, transA.convert_to_rocm(), transB.convert_to_rocm(), m, n, k, (T *) alpha, 
                (T *) mat->ptr_dev_a[flush_index], lda, stride_a,
                (T *) mat->ptr_dev_b[flush_index], ldb, stride_b, (T *) beta, 
                (T *) mat->ptr_dev_c[flush_index], ldc, stride_c, batch_count);
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
    int flush_index = rep % flush_batch_count;
    stat = func(handle, transA.convert_to_rocm(), transB.convert_to_rocm(), m, n, k, (T *) alpha, 
                (T *) mat->ptr_dev_a[flush_index], lda, stride_a,
                (T *) mat->ptr_dev_b[flush_index], ldb, stride_b, (T *) beta, 
                (T *) mat->ptr_dev_c[flush_index], ldc, stride_c, batch_count);
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

void rocblas_gemm::test_gemm_ex(rocblas_gemm_inst *mat) {
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
    int flush_index = rep % flush_batch_count;
    // clang-format off
    stat = rocblas_gemm_ex(handle, transA.convert_to_rocm(), transB.convert_to_rocm(), m, n, k, alpha, 
                           mat->ptr_dev_a[flush_index], a_type, lda, 
                           mat->ptr_dev_b[flush_index], b_type, ldb, beta, 
                           mat->ptr_dev_c[flush_index], c_type, ldc, 
                           mat->ptr_dev_d[flush_index], d_type, ldd, compute,
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
    int flush_index = rep % flush_batch_count;
    // clang-format off
    stat = rocblas_gemm_ex(handle, transA.convert_to_rocm(), transB.convert_to_rocm(), m, n, k, alpha, 
                           mat->ptr_dev_a[flush_index], a_type, lda, 
                           mat->ptr_dev_b[flush_index], b_type, ldb, beta, 
                           mat->ptr_dev_c[flush_index], c_type, ldc, 
                           mat->ptr_dev_d[flush_index], d_type, ldd, compute,
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


