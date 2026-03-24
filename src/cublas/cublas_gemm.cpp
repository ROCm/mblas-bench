#include "cublas_gemm.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <bitset>
#include <future>
#include <iomanip>
#include <numeric>
#include <regex>
#include <string>
#include <thread>

#include "cublas_convert.h"
#include "cublas_create_allocate.h"
#include "cublas_datatype_utils.h"
#include "cuda_error.h"
#include "cxxopts.hpp"
#include "generic_init.h"
#include "mblas_cuda_data_type.h"

using std::cerr;
using std::cout;
using std::endl;
using std::move;
using std::string;
using std::thread;
using std::vector;

// clang-format off
std::vector<gemmPrecType> cublas_gemm::gemm_ex_supported = {
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

std::vector<TgemmPrecType> cublas_gemm::Tgemm_ex_supported = {
    {mblas_cuda_data_type::MBLAS_R_16BF, mblas_cuda_data_type::MBLAS_R_16BF}, {mblas_cuda_data_type::MBLAS_R_16F, mblas_cuda_data_type::MBLAS_R_16F},
    {mblas_cuda_data_type::MBLAS_R_8I, mblas_cuda_data_type::MBLAS_R_32F},    {mblas_cuda_data_type::MBLAS_R_16BF, mblas_cuda_data_type::MBLAS_R_32F},
    {mblas_cuda_data_type::MBLAS_R_16F, mblas_cuda_data_type::MBLAS_R_32F},   {mblas_cuda_data_type::MBLAS_R_32F, mblas_cuda_data_type::MBLAS_R_32F},
    {mblas_cuda_data_type::MBLAS_C_8I, mblas_cuda_data_type::MBLAS_C_32F},    {mblas_cuda_data_type::MBLAS_C_32F, mblas_cuda_data_type::MBLAS_C_32F},

};

void cublas_gemm::init_prec_map() {}

void cublas_gemm::parse_dev_iters(std::string deviceStr) {
  // Parse device
  std::stringstream ss(deviceStr);
  while (ss.good()) {
    string deviceSStr;
    getline(ss, deviceSStr, ',');
    int devInt = stoi(deviceSStr);
    cublasgemmInst val = cublasgemmInst(devInt);
    mat_ptrs.push_back(val);
  }
}

void cublas_gemm::parse_problem_type(string computeTStr, string scalarTStr, string aStr,
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
  a_type = mblas_cuda_data_type(aStr);
  b_type = mblas_cuda_data_type(bStr);
  c_type = mblas_cuda_data_type(cStr);

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
        std::find(begin(gemm_ex_supported), end(gemm_ex_supported), selType);
    if (result == end(gemm_ex_supported)) {
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
        std::find(begin(Tgemm_ex_supported), end(Tgemm_ex_supported), selType);
    if (result == end(Tgemm_ex_supported)) {
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

cublas_gemm::cublas_gemm(cxxopts::ParseResult result) : generic_gemm(result) {
  // cublasCreate(&handle);
  // check_cublas(cublasCreate(&handle));
  init_prec_map();
  // Grab precision from command line
  //precision = mblas_cuda_data_type(result["precision"].as<string>());
  precision = mblas_cuda_data_type(result["precision"].as<string>());
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
  parse_problem_type(computeT, scalarT, aT, bT, cT);

  parse_dev_iters(result["device"].as<string>());
  std::string tA = result["transposeA"].as<std::string>();
  std::string tB = result["transposeB"].as<std::string>();
  transA = mblas_cuda_operation(result["transposeA"].as<std::string>());
  transB = mblas_cuda_operation(result["transposeB"].as<std::string>());

  // Pull in alpha and beta, alloc memory and save to pointers
  // Use local allocation to avoid cross-library malloc/free issues
  string salpha = result["alpha"].as<string>();
  string salphai = result["alphai"].as<string>();
  alpha = malloc(get_malloc_size_scalar(precision));
  type_call_host<set_scalar>(precision, alpha, salpha, salphai);
  
  string sbeta = result["beta"].as<string>();
  string sbetai = result["betai"].as<string>();
  beta = malloc(get_malloc_size_scalar(precision));
  type_call_host<set_scalar>(precision, beta, sbeta, sbetai);
  
  set_flush_batch_count( 
      type_call_dev<sizeofCUDT>(a_type), type_call_dev<sizeofCUDT>(b_type), 
      type_call_dev<sizeofCUDT>(c_type), type_call_dev<sizeofCUDT>(c_type), 
      a_type.get_packing_count(), 
      b_type.get_packing_count(), 
      c_type.get_packing_count(), 
      c_type.get_packing_count(), 
      true);
}

string cublas_gemm::prepare_array() {
  alpha = convert_scalar(scalar, alpha);
  beta = convert_scalar(scalar, beta);
  this->alloc_host();
  this->fill_host();

  int num_devices;
  cudaGetDeviceCount(&num_devices);
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

  run_threaded(&cublas_gemm::alloc_dev);
  run_threaded(&cublas_gemm::copy_host_to_dev);
  std::ostringstream ossHeader;
  ossHeader << "transA_option,transB_option,M,N,K,lda,ldb,ldc,";
  if (batched) {
    ossHeader << "batch_count,";
  }
  ossHeader << "cuBLAS-Gflops,cuBLAS-GB/s,cuBLAS-us," << endl;
  return ossHeader.str();
}

void cublas_gemm::run_threaded(void (cublas_gemm::*func)(cublasgemmInst *)) {
  vector<thread> threads;
  for (auto &instance : mat_ptrs) {
    threads.push_back(thread(func, this, &instance));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void cublas_gemm::alloc_host() {
  ptr_host_a =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(a_type));
  ptr_host_b =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(b_type));
  ptr_host_c =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(c_type));

  for (int i = 0; i < flush_batch_count; i++) {
    ptr_host_a[i] = malloc(get_malloc_size_host(a_type, rows_mem_a, cols_mem_a, batch_count));
    ptr_host_b[i] = malloc(get_malloc_size_host(b_type, rows_mem_b, cols_mem_b, batch_count));
    ptr_host_c[i] = malloc(get_malloc_size_host(c_type, rows_mem_c, cols_mem_c, batch_count));
  }
}

void cublas_gemm::alloc_dev(cublasgemmInst *mat) {
  cudaSetDevice(mat->devIDX);

  mat->ptr_dev_a =
      (void **)malloc(batch_count * flush_batch_count * type_call_dev<sizeofCUDTP>(a_type));
  mat->ptr_dev_b =
      (void **)malloc(batch_count * flush_batch_count * type_call_dev<sizeofCUDTP>(b_type));
  mat->ptr_dev_c =
      (void **)malloc(batch_count * flush_batch_count * type_call_dev<sizeofCUDTP>(c_type));

  for (int i = 0; i < flush_batch_count; i++) {
    cudaMalloc(&mat->ptr_dev_a[i], get_malloc_size_dev(a_type, rows_mem_a, cols_mem_a, batch_count));
    cudaMalloc(&mat->ptr_dev_b[i], get_malloc_size_dev(b_type, rows_mem_b, cols_mem_b, batch_count));
    cudaMalloc(&mat->ptr_dev_c[i], get_malloc_size_dev(c_type, rows_mem_c, cols_mem_c, batch_count));
  }

  mat->wSZ = workspace_size;
  cudaMalloc(&mat->devWork, mat->wSZ);
}

void cublas_gemm::fill_host() {
  type_call_host<initHost>(a_type, a_props.init, ptr_host_a, rows_a, cols_a, lda,
                         batch_count, stride_a, flush_batch_count, control_a, constant_a, filename_a);
  type_call_host<initHost>(b_type, b_props.init, ptr_host_b, rows_b, cols_b, ldb,
                         batch_count, stride_b, flush_batch_count, control_b, constant_b, filename_b);
  type_call_host<initHost>(c_type, c_props.init, ptr_host_c, rows_c, cols_c, ldc,
                         batch_count, stride_c, flush_batch_count, control_c, constant_c, filename_c);
  // D is just output, don't need to init
}

void cublas_gemm::copy_host_to_dev(cublasgemmInst *mat) {
  cudaSetDevice(mat->devIDX);

  for (int i = 0; i < flush_batch_count; i++) {
    copy_and_convert(a_type, ptr_host_a[i], mat->ptr_dev_a[i], rows_mem_a, cols_mem_a, batch_count);
    copy_and_convert(b_type, ptr_host_b[i], mat->ptr_dev_b[i], rows_mem_b, cols_mem_b, batch_count);
    copy_and_convert(c_type, ptr_host_c[i], mat->ptr_dev_c[i], rows_mem_c, cols_mem_c, batch_count);
  }

  //if (batched && !strided) {
  //  // Perform some pointer arithmetic to calculate the arrays we pass to the
  //  // gpu
  //  mat->ptr_host_a =
  //      (void **)malloc(batch_count * type_call_host<sizeofCUDTP>(a_type));
  //  mat->ptr_host_b =
  //      (void **)malloc(batch_count * type_call_host<sizeofCUDTP>(b_type));
  //  mat->ptr_host_c =
  //      (void **)malloc(batch_count * type_call_host<sizeofCUDTP>(c_type));
  //  check_cuda(
  //      cudaMalloc(&mat->ptr_dev_a, batch_count * type_call_host<sizeofCUDTP>(a_type)));
  //  check_cuda(
  //      cudaMalloc(&mat->ptr_dev_b, batch_count * type_call_host<sizeofCUDTP>(b_type)));
  //  check_cuda(
  //      cudaMalloc(&mat->ptr_dev_c, batch_count * type_call_host<sizeofCUDTP>(c_type)));
  //  //type_call_dev<batchedPtrMagic>(a_type, mat->ptr_host_a, mat->ptr_dev_a, mat->devA,
  //  //                            batch_count, rows_mem_a, cols_mem_a);
  //  //type_call_dev<batchedPtrMagic>(b_type, mat->ptr_host_b, mat->ptr_dev_b, mat->devB,
  //  //                            batch_count, rows_mem_b, cols_mem_b);
  //  //type_call_dev<batchedPtrMagic>(c_type, mat->ptr_host_c, mat->ptr_dev_c, mat->devC,
  //  //                            batch_count, rows_mem_c, cols_mem_c);
  //}
}

void cublas_gemm::free_mem() {
  free(alpha);
  free(beta);
  for (int i = 0; i < flush_batch_count; i++) {
    free(ptr_host_a[i]);
    free(ptr_host_b[i]);
    free(ptr_host_c[i]);
  }
  free(ptr_host_a);
  free(ptr_host_b);
  free(ptr_host_c);
  for (auto mat : mat_ptrs) {
    for (int i = 0; i < flush_batch_count; i++) {
      cudaFree(mat.ptr_dev_a[i]);
      cudaFree(mat.ptr_dev_b[i]);
      cudaFree(mat.ptr_dev_c[i]);
    }
    cudaFree(mat.ptr_dev_a);
    cudaFree(mat.ptr_dev_b);
    cudaFree(mat.ptr_dev_c);
    cudaFree(mat.devWork);
    //if (batched && !strided) {
    //  free(mat.ptr_host_a);
    //  free(mat.ptr_host_b);
    //  free(mat.ptr_host_c);
    //  cudaFree(mat.ptr_dev_a);
    //  cudaFree(mat.ptr_dev_b);
    //  cudaFree(mat.ptr_dev_c);
    //}
  }
}

double cublas_gemm::test() {
  vector<thread> threads;
  double gflops = 0.0;
  for (auto &mat : mat_ptrs) {
    // TgemmBatched
    if (match_gemm_type(precision, function, mblas_data_type::MBLAS_R_64F, {"cublasDgemm", "dgemm", "gemm"})) {
      std::function<decltype(cublasDgemm)> dgemm_var = cublasDgemm;
      threads.push_back(
          thread(&cublas_gemm::test_Tgemm<double>, this, dgemm_var, &mat));
    } else if (match_gemm_type(precision, function, mblas_data_type::MBLAS_R_32F, {"cublasSgemm", "sgemm", "gemm"})) {
      std::function<decltype(cublasSgemm)> sgemm_var = cublasSgemm;
      threads.push_back(
          thread(&cublas_gemm::test_Tgemm<float>, this, sgemm_var, &mat));
    } else if (match_gemm_type(precision, function, mblas_data_type::MBLAS_R_16F, {"cublasHgemm", "hgemm", "gemm"})) {
      std::function<decltype(cublasHgemm)> hgemm_var = cublasHgemm;
      threads.push_back(
          thread(&cublas_gemm::test_Tgemm<__half>, this, hgemm_var, &mat));
    } else if (match_gemm_type(precision, function, mblas_data_type::MBLAS_C_64F, {"cublasZgemm", "zgemm", "gemm"})) {
      std::function<decltype(cublasZgemm)> zgemm_var = cublasZgemm;
      threads.push_back(thread(&cublas_gemm::test_Tgemm<cuDoubleComplex>, this,
                               zgemm_var, &mat));
    } else if (match_gemm_type(precision, function, mblas_data_type::MBLAS_C_32F, {"cublasCgemm", "cgemm", "gemm"})) {
      std::function<decltype(cublasCgemm)> cgemm_var = cublasCgemm;
      threads.push_back(
          thread(&cublas_gemm::test_Tgemm<cuComplex>, this, cgemm_var, &mat));
    } else if (match_gemm_type(precision, function, mblas_data_type::MBLAS_C_64F, {"cublasZgemm3m", "zgemm3m", "gemm3m"})) {
      std::function<decltype(cublasZgemm3m)> zgemm3m_var = cublasZgemm3m;
      threads.push_back(thread(&cublas_gemm::test_Tgemm<cuDoubleComplex>, this,
                               zgemm3m_var, &mat));
    } else if (match_gemm_type(precision, function, mblas_data_type::MBLAS_C_32F, {"cublasCgemm3m", "cgemm3m", "gemm3m"})) {
      std::function<decltype(cublasCgemm3m)> cgemm3m_var = cublasCgemm3m;
      threads.push_back(
          thread(&cublas_gemm::test_Tgemm<cuComplex>, this, cgemm3m_var, &mat));
    }
    // TgemmBatched
    // Disabled due to batched & rotating tensors not being implemented at the same time
    // else if (function == "cublasDgemmBatched" && precision == mblas_data_type::MBLAS_R_64F) {
    //   std::function<decltype(cublasDgemmBatched)> dgemm_var =
    //       cublasDgemmBatched;
    //   threads.push_back(
    //       thread(&cublas_gemm::testTgemmBatched<double>, this, dgemm_var, &mat));
    // } else if (function == "cublasSgemmBatched" && precision == mblas_data_type::MBLAS_R_32F) {
    //   std::function<decltype(cublasSgemmBatched)> sgemm_var =
    //       cublasSgemmBatched;
    //   threads.push_back(
    //       thread(&cublas_gemm::testTgemmBatched<float>, this, sgemm_var, &mat));
    // } else if (function == "cublasHgemmBatched" && precision == mblas_data_type::MBLAS_R_16F) {
    //   std::function<decltype(cublasHgemmBatched)> hgemm_var =
    //       cublasHgemmBatched;
    //   threads.push_back(
    //       thread(&cublas_gemm::testTgemmBatched<__half>, this, hgemm_var, &mat));
    // } else if (function == "cublasZgemmBatched" && precision == mblas_data_type::MBLAS_C_64F) {
    //   std::function<decltype(cublasZgemmBatched)> zgemm_var =
    //       cublasZgemmBatched;
    //   threads.push_back(thread(&cublas_gemm::testTgemmBatched<cuDoubleComplex>,
    //                            this, zgemm_var, &mat));
    // } else if (function == "cublasCgemmBatched" && precision == mblas_data_type::MBLAS_C_32F) {
    //   std::function<decltype(cublasCgemmBatched)> cgemm_var =
    //       cublasCgemmBatched;
    //   threads.push_back(thread(&cublas_gemm::testTgemmBatched<cuComplex>, this,
    //                            cgemm_var, &mat));
    // }
    // TgemmStridedBatched
    else if (function == "cublasDgemmStridedBatched" &&
             precision == mblas_data_type::MBLAS_R_64F) {
      std::function<decltype(cublasDgemmStridedBatched)> dgemm_var =
          cublasDgemmStridedBatched;
      threads.push_back(thread(&cublas_gemm::testTgemmStridedBatched<double>,
                               this, dgemm_var, &mat));
    } else if (function == "cublasSgemmStridedBatched" &&
               precision == mblas_data_type::MBLAS_R_32F) {
      std::function<decltype(cublasSgemmStridedBatched)> sgemm_var =
          cublasSgemmStridedBatched;
      threads.push_back(thread(&cublas_gemm::testTgemmStridedBatched<float>,
                               this, sgemm_var, &mat));
    } else if (function == "cublasHgemmStridedBatched" &&
               precision == mblas_data_type::MBLAS_R_16F) {
      std::function<decltype(cublasHgemmStridedBatched)> hgemm_var =
          cublasHgemmStridedBatched;
      threads.push_back(thread(&cublas_gemm::testTgemmStridedBatched<__half>,
                               this, hgemm_var, &mat));
    } else if (function == "cublasZgemmStridedBatched" &&
               precision == mblas_data_type::MBLAS_C_64F) {
      std::function<decltype(cublasZgemmStridedBatched)> zgemm_var =
          cublasZgemmStridedBatched;
      threads.push_back(
          thread(&cublas_gemm::testTgemmStridedBatched<cuDoubleComplex>, this,
                 zgemm_var, &mat));
    } else if (function == "cublasCgemmStridedBatched" &&
               precision == mblas_data_type::MBLAS_C_32F) {
      std::function<decltype(cublasCgemmStridedBatched)> cgemm_var =
          cublasCgemmStridedBatched;
      threads.push_back(thread(&cublas_gemm::testTgemmStridedBatched<cuComplex>,
                               this, cgemm_var, &mat));
    } else if (function == "cublasCgemm3mStridedBatched" &&
               precision == mblas_data_type::MBLAS_C_32F) {
      std::function<decltype(cublasCgemm3mStridedBatched)> cgemm_var =
          cublasCgemm3mStridedBatched;
      threads.push_back(thread(&cublas_gemm::testTgemmStridedBatched<cuComplex>,
                               this, cgemm_var, &mat));
    }
    // TgemmEx
    else if (function == "cublasSgemmEx") {
      std::function<decltype(cublasSgemmEx)> sgemm_var = cublasSgemmEx;
      threads.push_back(
          thread(&cublas_gemm::testTGemmEx<float>, this, sgemm_var, &mat));
    } else if (function == "cublasCgemmEx") {
      std::function<decltype(cublasCgemmEx)> cgemm_var = cublasCgemmEx;
      threads.push_back(
          thread(&cublas_gemm::testTGemmEx<cuComplex>, this, cgemm_var, &mat));
    }
    // gemmEx
    else if (strided && function == "cublasGemmExStridedBatched") {
      // Call the Gemm strided batched deployment script
    } /* else if (batched && function == "cublasGemmExBatched") {
      // Call the Gemm batched code
    } */ else if (function == "cublasGemmEx" || function == "gemm_ex" || function == "gemm_ex3") {
      threads.push_back(thread(&cublas_gemm::testGemmEx, this, &mat));
    }
  }
  // Wait on running jobs
  for (auto &thread : threads) {
    thread.join();
  }

  // Sum all gflops
  gflop_per_second = std::accumulate(
      begin(mat_ptrs), end(mat_ptrs), 0.0,
      [](double i, const cublasgemmInst &o) { return o.gflops + i; });

  gbyte_per_second = std::accumulate(
      begin(mat_ptrs), end(mat_ptrs), 0.0,
      [](double i, const cublasgemmInst &o) { return o.gbytes + i; });

  iter_time_us = std::accumulate(begin(mat_ptrs), end(mat_ptrs), 0.0,
                                 [](double i, const cublasgemmInst &o) {
                                   return o.time_us + i;
                                 }) /
                 mat_ptrs.size();

  return gflop_per_second;
}

std::string cublas_gemm::get_result_string() {
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

std::tuple<double, double, double> cublas_gemm::calculate_figure_of_merit(
    double totalTime_ms) {
  double avgTime_ms = totalTime_ms / iters;
  double avgTime_s = avgTime_ms / 1000.0f;
  double avgTime_us = avgTime_ms * 1000.0f;

  int a_sz = type_call_dev<sizeofCUDT>(a_type);
  int b_sz = type_call_dev<sizeofCUDT>(b_type);
  int c_sz = type_call_dev<sizeofCUDT>(c_type);
  int a_pack = a_type.get_packing_count();
  int b_pack = b_type.get_packing_count();
  int c_pack = c_type.get_packing_count();

  int flopPerSize = 2;
  if (!precision.is_real()) {
    flopPerSize = 8;
  }
  double gbytes = (((static_cast<double>(a_sz) / static_cast<double>(a_pack)) *
                    static_cast<double>(m) * static_cast<double>(k)) +
                   ((static_cast<double>(b_sz) / static_cast<double>(b_pack)) *
                    static_cast<double>(k) * static_cast<double>(n)) +
                   ((static_cast<double>(c_sz) / static_cast<double>(c_pack)) *
                    static_cast<double>(n) * static_cast<double>(m))) /
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
void cublas_gemm::test_Tgemm(
    std::function<cublasStatus_t(
        cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
        const T *, const T *, int, const T *, int, const T *, T *, int)>
        func,
    cublasgemmInst *mat) {
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;
  check_cuda(cudaSetDevice(mat->devIDX));
  check_cublas(cublasCreate(&handle));
  check_cuda(cudaStreamCreate(&stream));
  check_cublas(cublasSetStream(handle, stream));
  // check_cublas(cublasSetWorkspace(handle, mat->devWork, mat->wSZ));

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    int flush_index = rep % flush_batch_count;
    stat = func(handle, transA.convert_to_cuda(), transB.convert_to_cuda(), m, n, k, (T *) alpha, (T *) mat->ptr_dev_a[flush_index], lda, (T *) mat->ptr_dev_b[flush_index], ldb,
                (T *) beta, (T *) mat->ptr_dev_c[flush_index], ldc);

    // Check for errors during the gemm run
    check_cublas(stat);
    check_cuda(cudaGetLastError());
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
    int flush_index = rep % flush_batch_count;
    stat = func(handle, transA.convert_to_cuda(), transB.convert_to_cuda(), m, n, k, (T *) alpha, (T *) mat->ptr_dev_a[flush_index], lda, (T *) mat->ptr_dev_b[flush_index], ldb,
                (T *) beta, (T *) mat->ptr_dev_c[flush_index], ldc);
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  // Check for errors during the performance test
  check_cublas(stat);
  check_cuda(cudaGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  cudaEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculate_figure_of_merit(static_cast<double>(elapsedTime_ms));
}

// Disabled due to batched & rotating tensors not being implemented at the same time
// template <typename T>
// void cublas_gemm::testTgemmBatched(
//     std::function<cublasStatus_t(cublasContext *, cublasOperation_t,
//                                  cublasOperation_t, int, int, int, T const *,
//                                  T const *const *, int, T const *const *, int,
//                                  T const *, T *const *, int, int)>
//         func,
//     cublasgemmInst *mat) {
//   cublasStatus_t stat;
//   cublasHandle_t handle;
//   cudaStream_t stream;
//   check_cuda(cudaSetDevice(mat->devIDX));
//   check_cublas(cublasCreate(&handle));
//   check_cuda(cudaStreamCreate(&stream));
//   check_cublas(cublasSetStream(handle, stream));
//   // check_cublas(cublasSetWorkspace(handle, mat->devWork, mat->wSZ));
// 
//   T *alphaP = static_cast<T *>(alpha);
//   T *betaP = static_cast<T *>(beta);
//   T **devAP = reinterpret_cast<T **>(mat->ptr_dev_a);
//   T **devBP = reinterpret_cast<T **>(mat->ptr_dev_b);
//   T **devCP = reinterpret_cast<T **>(mat->ptr_dev_c);
// 
//   // Cold iters
//   for (int rep = 0; rep < cold_iters; rep++) {
//     stat = func(handle, transA.convert_to_cuda(), transB.convert_to_cuda(), m, n, k, alphaP, devAP, lda, devBP, ldb,
//                 betaP, devCP, ldc, batch_count);
// 
//     // Check for errors during the gemm run
//     check_cublas(stat);
//     check_cuda(cudaGetLastError());
//   }
//   cudaStreamSynchronize(stream);
// 
//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);
// 
//   /*
//     Run and time the performance test
//   */
//   cudaEventRecord(start, stream);
//   for (int rep = 0; rep < iters; rep++) {
//     stat = func(handle, transA.convert_to_cuda(), transB.convert_to_cuda(), m, n, k, alphaP, devAP, lda, devBP, ldb,
//                 betaP, devCP, ldc, batch_count);
//   }
//   cudaEventRecord(stop, stream);
//   cudaEventSynchronize(stop);
// 
//   // Check for errors during the performance test
//   check_cublas(stat);
//   check_cuda(cudaGetLastError());
// 
//   // Calculate and report GFlops
//   float elapsedTime_ms;
//   cudaEventElapsedTime(&elapsedTime_ms, start, stop);
//   std::tie(mat->gflops, mat->gbytes, mat->time_us) =
//       calculate_figure_of_merit(static_cast<double>(elapsedTime_ms));
// }

template <typename T>
void cublas_gemm::testTgemmStridedBatched(
    std::function<cublasStatus_t(
        cublasContext *, cublasOperation_t, cublasOperation_t, int, int, int,
        T const *, T const *, int, long long, T const *, int, long long,
        T const *, T *, int, long long, int)>
        func,
    cublasgemmInst *mat) {
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;
  check_cuda(cudaSetDevice(mat->devIDX));
  check_cublas(cublasCreate(&handle));
  check_cuda(cudaStreamCreate(&stream));
  check_cublas(cublasSetStream(handle, stream));
  // check_cublas(cublasSetWorkspace(handle, mat->devWork, mat->wSZ));

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    int flush_index = rep % flush_batch_count;
    stat = func(handle, transA.convert_to_cuda(), transB.convert_to_cuda(), m, n, k, (T *) alpha, (T *) mat->ptr_dev_a[flush_index], lda, stride_a,
                (T *) mat->ptr_dev_b[flush_index], ldb, stride_b, (T *) beta, (T *) mat->ptr_dev_c[flush_index], ldc, stride_c, batch_count);

    // Check for errors during the gemm run
    check_cublas(stat);
    check_cuda(cudaGetLastError());
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
    int flush_index = rep % flush_batch_count;
    stat = func(handle, transA.convert_to_cuda(), transB.convert_to_cuda(), m, n, k, (T *) alpha, (T *) mat->ptr_dev_a[flush_index], lda, stride_a,
                (T *) mat->ptr_dev_b[flush_index], ldb, stride_b, (T *) beta, (T *) mat->ptr_dev_c[flush_index], ldc, stride_c, batch_count);
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  // Check for errors during the performance test
  check_cublas(stat);
  check_cuda(cudaGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  cudaEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculate_figure_of_merit(static_cast<double>(elapsedTime_ms));
}

template <typename T>
void cublas_gemm::testTGemmEx(
    std::function<cublasStatus_t(
        cublasContext *, cublasOperation_t, cublasOperation_t, int, int, int,
        T const *, void const *, cudaDataType_t, int, void const *,
        cudaDataType_t, int, T const *, void *, cudaDataType_t, int)>
        func,
    cublasgemmInst *mat) {
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;
  check_cuda(cudaSetDevice(mat->devIDX));
  check_cublas(cublasCreate(&handle));
  check_cuda(cudaStreamCreate(&stream));
  check_cublas(cublasSetStream(handle, stream));
  // check_cublas(cublasSetWorkspace(handle, mat->devWork, mat->wSZ));

  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    int flush_index = rep % flush_batch_count;
    stat = func(handle, transA.convert_to_cuda(), transB.convert_to_cuda(), m, n, k, (T *) alpha, mat->ptr_dev_a[flush_index], a_type, lda,
                mat->ptr_dev_b[flush_index], b_type, ldb, (T *) beta, mat->ptr_dev_c[flush_index], c_type, ldc);

    // Check for errors during the gemm run
    check_cublas(stat);
    check_cuda(cudaGetLastError());
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
    int flush_index = rep % flush_batch_count;
    stat = func(handle, transA.convert_to_cuda(), transB.convert_to_cuda(), m, n, k, (T *) alpha, mat->ptr_dev_a[flush_index], a_type, lda,
                mat->ptr_dev_b[flush_index], b_type, ldb, (T *) beta, mat->ptr_dev_c[flush_index], c_type, ldc);
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  // Check for errors during the performance test
  check_cublas(stat);
  check_cuda(cudaGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  cudaEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculate_figure_of_merit(static_cast<double>(elapsedTime_ms));
}

void cublas_gemm::testGemmEx(cublasgemmInst *mat) {
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;
  check_cuda(cudaSetDevice(mat->devIDX));
  check_cublas(cublasCreate(&handle));
  check_cuda(cudaStreamCreate(&stream));
  check_cublas(cublasSetStream(handle, stream));
  check_cublas(cublasSetWorkspace(handle, mat->devWork, mat->wSZ));
  // cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    int flush_index = rep % flush_batch_count;
    stat = cublasGemmEx(handle, transA.convert_to_cuda(), transB.convert_to_cuda(), m, n, k, alpha, mat->ptr_dev_a[flush_index],
                        a_type, lda, mat->ptr_dev_b[flush_index], b_type, ldb, beta, mat->ptr_dev_c[flush_index],
                        c_type, ldc, compute, CUBLAS_GEMM_DEFAULT);

    // Check for errors during the gemm run
    check_cublas(stat);
    check_cuda(cudaGetLastError());
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
    int flush_index = rep % flush_batch_count;
    stat = cublasGemmEx(handle, transA.convert_to_cuda(), transB.convert_to_cuda(), m, n, k, alpha, mat->ptr_dev_a[flush_index],
                        a_type, lda, mat->ptr_dev_b[flush_index], b_type, ldb, beta, mat->ptr_dev_c[flush_index],
                        c_type, ldc, compute, CUBLAS_GEMM_DEFAULT);
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  // Check for errors during the performance test
  check_cublas(stat);
  check_cuda(cudaGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  cudaEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculate_figure_of_merit(static_cast<double>(elapsedTime_ms));
}
