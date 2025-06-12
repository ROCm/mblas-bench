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
std::vector<matmul_prec_type> hipblasLtGemm::matmul_supported = {
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

void hipblasLtGemm::parse_dev_iters(std::string deviceStr) {
  // Parse device
  std::stringstream ss(deviceStr);
  while (ss.good()) {
    string deviceSStr;
    getline(ss, deviceSStr, ',');
    int devInt = stoi(deviceSStr);
    hipblasLtGemmInst val = hipblasLtGemmInst(devInt);
    mat_ptrs.push_back(val);
  }
}

void hipblasLtGemm::parse_problem_type(string computeTStr, string scalarTStr,
                               string aStr, string bStr, string cStr,
                               string dStr) {
  compute.set_compute(computeTStr, precision);
  scalar.set_scalar(scalarTStr, precision, compute);
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

void hipblasLtGemm::validate_parameters() {
  // Validate that data types exist in table of supported configurations
  matmul_prec_type selType = {
      compute, scalar, a_type, b_type, c_type, d_type, mblasHipDataType(MBLAS_ANY)};
  auto result =
      std::find(begin(matmul_supported), end(matmul_supported), selType);
  if (result != end(matmul_supported)) {
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
  parse_problem_type(computeT, scalarT, aT, bT, cT, dT);

  parse_dev_iters(result["device"].as<string>());
  std::string tA = result["transposeA"].as<std::string>();
  std::string tB = result["transposeB"].as<std::string>();
  transA = mblasHipOperation(result["transposeA"].as<std::string>());
  transB = mblasHipOperation(result["transposeB"].as<std::string>());
  validate_parameters();

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
  uint64_t a_offset, b_offset, c_offset, d_offset;
  set_flush_batch_count(a_offset, b_offset, c_offset, d_offset, 
      type_call_dev<sizeofCUDT>(a_type), type_call_dev<sizeofCUDT>(b_type), 
      type_call_dev<sizeofCUDT>(c_type), type_call_dev<sizeofCUDT>(d_type), 
      get_packing_count(a_type), 
      get_packing_count(b_type), 
      get_packing_count(c_type), 
      get_packing_count(d_type), 
      inplace);
}

string hipblasLtGemm::prepare_array() {
  alpha = convert_scalar(scalar, alpha);
  beta = convert_scalar(scalar, beta);
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
  run_threaded(&hipblasLtGemm::alloc_dev);
  run_threaded(&hipblasLtGemm::copy_host_to_dev);
  run_threaded(&hipblasLtGemm::prepare_matrix);
  // Enable tuning with a parameter later
  if (false) {
  } else {
    run_threaded(&hipblasLtGemm::no_tuning);
  }
  std::ostringstream ossHeader;
  ossHeader << "transA_option,transB_option,M,N,K,lda,ldb,ldc,";
  if (batched) {
    ossHeader << "batch_count,";
  }
  ossHeader << "hipBLASLt-Gflops,hipBLASLt-GB/s,hipBLASLt-us," << endl;
  return ossHeader.str();
}

void hipblasLtGemm::run_threaded(void (hipblasLtGemm::*func)(hipblasLtGemmInst *)) {
  vector<thread> threads;
  for (auto &instance : mat_ptrs) {
    threads.push_back(thread(func, this, &instance));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void hipblasLtGemm::alloc_host() {
  // auto resultA = std::async(allocate_host_array, a_type, m, k, batch_count);
  // auto resultB = std::async(allocate_host_array, b_type, k, n, batch_count);
  // auto resultC = std::async(allocate_host_array, c_type, n, m, batch_count);
  // host_a = resultA.get();
  // host_b = resultB.get();
  // host_c = resultC.get();
  ptr_host_a =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(a_type));
  ptr_host_b =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(b_type));
  ptr_host_c =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(c_type));
  if (!inplace) {
    ptr_host_d =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(d_type));
  } else {
    ptr_host_d = ptr_host_c;
  }


  for (int i = 0; i < flush_batch_count; i++) {
    ptr_host_a[i] = allocate_host_array(a_type, rows_mem_a, cols_mem_a, batch_count);
    ptr_host_b[i] = allocate_host_array(b_type, rows_mem_b, cols_mem_b, batch_count);
    ptr_host_c[i] = allocate_host_array(c_type, rows_mem_c, cols_mem_c, batch_count);
    if (!inplace) {
      ptr_host_d[i] = allocate_host_array(d_type, rows_mem_d, cols_mem_d, batch_count);
    }
  }
  //host_a = allocate_host_array(a_type, m, k, batch_count);
  //host_b = allocate_host_array(b_type, k, n, batch_count);
  //host_c = allocate_host_array(c_type, m, n, batch_count);
}

void hipblasLtGemm::alloc_dev(hipblasLtGemmInst *mat) {
  hipSetDevice(mat->devIDX);

  mat->ptr_dev_a =
      (void **)malloc(batch_count * flush_batch_count * type_call_dev<sizeofCUDTP>(a_type));
  mat->ptr_dev_b =
      (void **)malloc(batch_count * flush_batch_count * type_call_dev<sizeofCUDTP>(b_type));
  mat->ptr_dev_c =
      (void **)malloc(batch_count * flush_batch_count * type_call_dev<sizeofCUDTP>(c_type));
  if (!inplace) {
    mat->ptr_dev_d =
        (void **)malloc(batch_count * flush_batch_count * type_call_dev<sizeofCUDTP>(d_type));
  } else {
    mat->ptr_dev_d = mat->ptr_dev_c;
  }

  for (int i = 0; i < flush_batch_count; i++) {
    mat->ptr_dev_a[i] = allocate_dev_array(a_type, rows_mem_a, cols_mem_a, batch_count);
    mat->ptr_dev_b[i] = allocate_dev_array(b_type, rows_mem_b, cols_mem_b, batch_count);
    mat->ptr_dev_c[i] = allocate_dev_array(c_type, rows_mem_c, cols_mem_c, batch_count);
    if (!inplace) {
      mat->ptr_dev_d[i] = allocate_dev_array(d_type, rows_mem_d, cols_mem_d, batch_count);
    }
  }
  mat->wSZ = workspace_size;
  hipMalloc(&mat->devWork, mat->wSZ);
}

void hipblasLtGemm::fill_host() {
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
  //                         batch_count, stride_c, 1.f, false));
  // for (auto &thread : threads) {
  //  thread.join();
  //}

  for (int i = 0; i < flush_batch_count; i++){
    type_call_host<initHost>(a_type, initialization, ptr_host_a[i], rows_a, cols_a, lda,
                           batch_count, stride_a, control_a, constant_a, filename_a);
    type_call_host<initHost>(b_type, initialization, ptr_host_b[i], rows_b, cols_b, ldb,
                           batch_count, stride_b, control_b, constant_b, filename_b);
    type_call_host<initHost>(c_type, initialization, ptr_host_c[i], rows_c, cols_c, ldc,
                           batch_count, stride_c, control_c, constant_c, filename_c);
  }
}

void hipblasLtGemm::copy_host_to_dev(hipblasLtGemmInst *mat) {
  hipSetDevice(mat->devIDX);
  for (int i = 0; i < flush_batch_count; i++) {
    copy_and_convert(a_type, ptr_host_a[i], mat->ptr_dev_a[i], m, k, batch_count);
    copy_and_convert(b_type, ptr_host_b[i], mat->ptr_dev_b[i], k, n, batch_count);
    copy_and_convert(c_type, ptr_host_c[i], mat->ptr_dev_c[i], n, m, batch_count);
  }
}

void hipblasLtGemm::prepare_matrix(hipblasLtGemmInst *mat) {
  check_hipblas(hipblasLtMatmulDescCreate(&mat->desc_op, compute, scalar));
  // These values are read in with no type, so they need to be convirted first
  // Thanks for the wonderful standard Nvidia :D!
  hipblasOperation_t transA_local = transA.convertToHip();
  hipblasOperation_t transB_local = transB.convertToHip();
  check_hipblas(hipblasLtMatmulDescSetAttribute(
      mat->desc_op, HIPBLASLT_MATMUL_DESC_TRANSA, &transA_local, sizeof(transA)));
  check_hipblas(hipblasLtMatmulDescSetAttribute(
      mat->desc_op, HIPBLASLT_MATMUL_DESC_TRANSB, &transB_local, sizeof(transB)));

  check_hipblas(
      hipblasLtMatrixLayoutCreate(&mat->desc_a, a_type, rows_a, cols_a, lda));
  check_hipblas(
      hipblasLtMatrixLayoutCreate(&mat->desc_b, b_type, rows_b, cols_b, ldb));
  check_hipblas(
      hipblasLtMatrixLayoutCreate(&mat->desc_c, c_type, rows_c, cols_c, ldc));
  if (!inplace) {
    check_hipblas(
        hipblasLtMatrixLayoutCreate(&mat->desc_d, d_type, rows_d, cold_d, ldd));
  } else {
    mat->desc_d = mat->desc_c;
  }
  if (batch_count > 1) {
    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_a, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_c, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_d, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_a, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_b, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_c, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));
    check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_d, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_d, sizeof(stride_d)));
  }

  check_hipblas(hipblasLtMatmulPreferenceCreate(&mat->pref));
  check_hipblas(hipblasLtMatmulPreferenceSetAttribute(
      mat->pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &mat->wSZ,
      sizeof(mat->wSZ)));
}

void hipblasLtGemm::no_tuning(hipblasLtGemmInst *mat) {
  hipblasStatus_t stat;
  hipblasLtHandle_t handle;
  check_hip(hipSetDevice(mat->devIDX));
  check_hipblas(hipblasLtCreate(&handle));
  int retResults = 0;
  hipblasLtMatmulHeuristicResult_t heuristicResult = {0};

  check_hipblas(hipblasLtMatmulAlgoGetHeuristic(
      handle, mat->desc_op, mat->desc_a, mat->desc_b, mat->desc_c, mat->desc_d,
      mat->pref, 1, &heuristicResult, &retResults));

  if (retResults == 0) {
    check_hipblas(HIPBLAS_STATUS_NOT_SUPPORTED);
  }
  mat->algo = heuristicResult;
}
void hipblasLtGemm::auto_tuning(hipblasLtGemmInst *mat) {
  // Not currently implemented, using simple method
  no_tuning(mat);
}

void hipblasLtGemm::free_mem() {
  free(alpha);
  free(beta);
  //free(host_a);
  //free(host_b);
  //free(host_c);
  for (auto mat : mat_ptrs) {
    hipFree(mat.ptr_dev_a);
    hipFree(mat.ptr_dev_b);
    hipFree(mat.ptr_dev_c);
    hipFree(mat.ptr_dev_d);
  }
}

double hipblasLtGemm::test() {
  vector<thread> threads;
  double gflops = 0.0;
  for (auto &mat : mat_ptrs) {
    threads.push_back(thread(&hipblasLtGemm::test_matmul, this, &mat));
  }
  // Wait on running jobs
  for (auto &thread : threads) {
    thread.join();
  }

  // Sum all gflops
  gflop_per_second = std::accumulate(
      begin(mat_ptrs), end(mat_ptrs), 0.0,
      [](double i, const hipblasLtGemmInst &o) { return o.gflops + i; });

  gbyte_per_second = std::accumulate(
      begin(mat_ptrs), end(mat_ptrs), 0.0,
      [](double i, const hipblasLtGemmInst &o) { return o.gbytes + i; });

  iter_time_us = std::accumulate(begin(mat_ptrs), end(mat_ptrs), 0.0,
                                 [](double i, const hipblasLtGemmInst &o) {
                                   return o.time_us + i;
                                 }) /
                 mat_ptrs.size();

  return gflop_per_second;
}

std::string hipblasLtGemm::get_result_string() {
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

std::tuple<double, double, double> hipblasLtGemm::calculate_figure_of_merit(
    double totalTime_ms) {
  double avgTime_ms = totalTime_ms / iters;
  double avgTime_s = avgTime_ms / 1000.0f;
  double avgTime_us = avgTime_ms * 1000.0f;

  int a_sz = type_call_dev<sizeofCUDT>(a_type);
  int b_sz = type_call_dev<sizeofCUDT>(b_type);
  int c_sz = type_call_dev<sizeofCUDT>(c_type);

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

void hipblasLtGemm::test_matmul(hipblasLtGemmInst *mat) {
  hipblasStatus_t stat;
  hipblasLtHandle_t handle;
  hipStream_t stream;
  check_hip(hipSetDevice(mat->devIDX));
  check_hipblas(hipblasLtCreate(&handle));
  check_hip(hipStreamCreate(&stream));
  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    int flush_index = rep % flush_batch_count;
    stat = hipblasLtMatmul(handle, mat->desc_op, alpha, mat->ptr_dev_a[flush_index], mat->desc_a,
                          mat->ptr_dev_b[flush_index], mat->desc_b, beta, mat->ptr_dev_c[flush_index], mat->desc_c,
                          mat->ptr_dev_d[flush_index], mat->desc_d, &mat->algo.algo, mat->devWork,
                          mat->wSZ, stream);
    // Check for errors during the gemm run
    check_hipblas(stat);
    check_hip(hipGetLastError());
  }
  hipStreamSynchronize(stream);

  hipEvent_t start, stop;
  check_hip(hipEventCreate(&start));
  check_hip(hipEventCreate(&stop));

  /*
    Run and time the performance test
  */
  hipEventRecord(start, stream);
  for (int rep = 0; rep < iters; rep++) {
    int flush_index = rep % flush_batch_count;
    stat = hipblasLtMatmul(handle, mat->desc_op, alpha, mat->ptr_dev_a[flush_index], mat->desc_a,
                          mat->ptr_dev_b[flush_index], mat->desc_b, beta, mat->ptr_dev_c[flush_index], mat->desc_c,
                          mat->ptr_dev_d[flush_index], mat->desc_d, &mat->algo.algo, mat->devWork,
                          mat->wSZ, stream);
  }
  hipEventRecord(stop, stream);
  hipEventSynchronize(stop);

  // Check for errors during the performance test
  check_hipblas(stat);
  check_hip(hipGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  hipEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculate_figure_of_merit(static_cast<double>(elapsedTime_ms));
}