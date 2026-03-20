#include "hipblaslt_gemm.h"

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

#include <future>
#include <iomanip>
#include <numeric>
#include <regex>
#include <string>
#include <thread>

#include "hip_convert.h"
#include "hip_create_allocate.h"
#include "hip_datatype_utils.h"
#include "hip_error.h"
#include "cxxopts.hpp"
#include "generic_setup.h"

using std::cerr;
using std::cout;
using std::endl;
using std::move;
using std::string;
using std::thread;
using std::vector;

// clang-format off
std::vector<matmul_prec_type> hipblaslt_gemm::matmul_supported = {
  // Compute type                 Scale Type    A Type        B Type        C Type        D Type        Bias Type
  {MBLAS_COMPUTE_32F,             MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F},
  {MBLAS_COMPUTE_32F_FAST_TF32,   MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F},
  {MBLAS_COMPUTE_32F,             MBLAS_R_32F,  MBLAS_R_16F,  MBLAS_R_16F,  MBLAS_R_16F,  MBLAS_R_16F,  MBLAS_R_16F},
  {MBLAS_COMPUTE_32F,             MBLAS_R_32F,  MBLAS_R_16F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F,  MBLAS_R_32F},
  {MBLAS_COMPUTE_32F,             MBLAS_R_32F,  MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32I,             MBLAS_R_32I,  MBLAS_R_8I,   MBLAS_R_8I,   MBLAS_R_8I,   MBLAS_R_8I,   MBLAS_ANY},
};

std::vector<matmul_prec_type_f8> hipblaslt_gemm::matmulSupportedF8 = {
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

#if HIP_VERSION >= 70000000
// MX format configurations (ROCm 7.0+)
// Note: These use block scaling with VEC32_UE8M0
std::vector<matmul_prec_type> hipblaslt_gemm::mx_matmul_supported = {
  // MXfp8 x MXfp8 (E4M3)
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_8F_E4M3, MBLAS_R_8F_E4M3, MBLAS_R_16BF, MBLAS_R_8F_E4M3, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_8F_E4M3, MBLAS_R_8F_E4M3, MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_8F_E4M3, MBLAS_R_8F_E4M3, MBLAS_R_16F, MBLAS_R_8F_E4M3, MBLAS_R_16F},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_8F_E4M3, MBLAS_R_8F_E4M3, MBLAS_R_16F, MBLAS_R_16F, MBLAS_R_16F},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_8F_E4M3, MBLAS_R_8F_E4M3, MBLAS_R_32F, MBLAS_R_32F, MBLAS_R_32F},
  
  // MXBF8 x MXBF8 (E5M2)
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_8F_E5M2, MBLAS_R_8F_E5M2, MBLAS_R_16BF, MBLAS_R_8F_E5M2, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_8F_E5M2, MBLAS_R_8F_E5M2, MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_8F_E5M2, MBLAS_R_8F_E5M2, MBLAS_R_16F, MBLAS_R_16F, MBLAS_R_16F},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_8F_E5M2, MBLAS_R_8F_E5M2, MBLAS_R_32F, MBLAS_R_32F, MBLAS_R_32F},
  
  // MXfp8 x MXBF8 (mixed 8-bit)
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_8F_E4M3, MBLAS_R_8F_E5M2, MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_8F_E4M3, MBLAS_R_8F_E5M2, MBLAS_R_16F, MBLAS_R_16F, MBLAS_R_16F},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_8F_E4M3, MBLAS_R_8F_E5M2, MBLAS_R_32F, MBLAS_R_32F, MBLAS_R_32F},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_8F_E5M2, MBLAS_R_8F_E4M3, MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_8F_E5M2, MBLAS_R_8F_E4M3, MBLAS_R_16F, MBLAS_R_16F, MBLAS_R_16F},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_8F_E5M2, MBLAS_R_8F_E4M3, MBLAS_R_32F, MBLAS_R_32F, MBLAS_R_32F},
  
  // MXfp6 x MXfp6 (E2M3)
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_6F_E2M3, MBLAS_R_6F_E2M3, MBLAS_R_16BF, MBLAS_R_6F_E2M3, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_6F_E2M3, MBLAS_R_6F_E2M3, MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_6F_E2M3, MBLAS_R_6F_E2M3, MBLAS_R_16F, MBLAS_R_16F, MBLAS_R_16F},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_6F_E2M3, MBLAS_R_6F_E2M3, MBLAS_R_32F, MBLAS_R_32F, MBLAS_R_32F},
  
  // MXBF6 x MXBF6 (E3M2)
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_6F_E3M2, MBLAS_R_6F_E3M2, MBLAS_R_16BF, MBLAS_R_6F_E3M2, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_6F_E3M2, MBLAS_R_6F_E3M2, MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_6F_E3M2, MBLAS_R_6F_E3M2, MBLAS_R_16F, MBLAS_R_16F, MBLAS_R_16F},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_6F_E3M2, MBLAS_R_6F_E3M2, MBLAS_R_32F, MBLAS_R_32F, MBLAS_R_32F},
  
  // MXfp6 x MXBF6 (mixed 6-bit)
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_6F_E2M3, MBLAS_R_6F_E3M2, MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_6F_E2M3, MBLAS_R_6F_E3M2, MBLAS_R_16F, MBLAS_R_16F, MBLAS_R_16F},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_6F_E2M3, MBLAS_R_6F_E3M2, MBLAS_R_32F, MBLAS_R_32F, MBLAS_R_32F},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_6F_E3M2, MBLAS_R_6F_E2M3, MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_6F_E3M2, MBLAS_R_6F_E2M3, MBLAS_R_16F, MBLAS_R_16F, MBLAS_R_16F},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_6F_E3M2, MBLAS_R_6F_E2M3, MBLAS_R_32F, MBLAS_R_32F, MBLAS_R_32F},
  
  // MXfp4 x MXfp4
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_4F_E2M1, MBLAS_R_4F_E2M1, MBLAS_R_16BF, MBLAS_R_4F_E2M1, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_4F_E2M1, MBLAS_R_4F_E2M1, MBLAS_R_16BF, MBLAS_R_16BF, MBLAS_R_16BF},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_4F_E2M1, MBLAS_R_4F_E2M1, MBLAS_R_16F, MBLAS_R_4F_E2M1, MBLAS_R_16F},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_4F_E2M1, MBLAS_R_4F_E2M1, MBLAS_R_16F, MBLAS_R_16F, MBLAS_R_16F},
  {MBLAS_COMPUTE_32F, MBLAS_R_32F, MBLAS_R_4F_E2M1, MBLAS_R_4F_E2M1, MBLAS_R_32F, MBLAS_R_32F, MBLAS_R_32F},
};
#endif
// clang-format on

void hipblaslt_gemm::parse_dev_iters(std::string deviceStr) {
  // Parse device
  std::stringstream ss(deviceStr);
  while (ss.good()) {
    string deviceSStr;
    getline(ss, deviceSStr, ',');
    int devInt = stoi(deviceSStr);
    hipblaslt_gemm_inst val = hipblaslt_gemm_inst(devInt);
    mat_ptrs.push_back(val);
  }
}

void hipblaslt_gemm::parse_problem_type(string computeTStr, string scalarTStr,
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
    a_type = mblas_hip_data_type(aStr);
    b_type = mblas_hip_data_type(bStr);
    c_type = mblas_hip_data_type(cStr);
    d_type = mblas_hip_data_type(dStr);
  }
}

#if HIP_VERSION >= 70000000
std::tuple<mblas_hip_data_type, hipblasLtMatmulMatrixScale_t, scale_size> 
hipblaslt_gemm::configure_scaling(matrix_desc desc, mblas_hip_data_type type, string matrix_id) {
  mblas_hip_data_type scale_type;
  hipblasLtMatmulMatrixScale_t scale_mode;
  scale_size scale_size_result;
  
  if (desc.scale_mode == scaling_type::Block) {
    scale_type = type.get_scale_type();  // Returns MBLAS_R_8F_UE8M0 for MX
    std::cout << "scale_type: " << scale_type.to_string() << std::endl;
    scale_mode = get_scale_mode(type);  // Returns VEC32_UE8M0 for MX
    scale_size_result = get_scale_tensor_size(desc.rows_mem, desc.cols_mem, scale_mode);
  } else if (type.is_mx_possible()) {
    string errorString = 
        "Non-block scaled MX formats not supported in hipblaslt. "
        "Matrix: " + matrix_id + "\nType: " + type.to_string();
    std::cerr << scaling_string(desc.scale_mode) << std::endl;
    throw std::invalid_argument(errorString);
  } else if (desc.scale_mode == scaling_type::Vector) {
    scale_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    long scaling_vec_len = (matrix_id == "B") ? n : m;
    scale_size_result = std::make_pair<size_t, size_t>(1, scaling_vec_len);
    scale_type = MBLAS_R_32F;
  } else if (desc.scale_mode == scaling_type::Scalar) {
    scale_size_result = std::make_pair<size_t, size_t>(1, 1);
    scale_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    scale_type = MBLAS_R_32F;
  } else {
    scale_size_result.rows = 0;
    scale_size_result.cols = 0;
  }
  
  return std::make_tuple(scale_type, scale_mode, scale_size_result);
}
#endif

void hipblaslt_gemm::validate_parameters() {
  // Validate that data types exist in table of supported configurations
  matmul_prec_type selType = {
      compute, scalar, a_type, b_type, c_type, d_type, mblas_hip_data_type(MBLAS_ANY)};
  auto result =
      std::find(begin(matmul_supported), end(matmul_supported), selType);
  if (result != end(matmul_supported)) {
    return;
  } else if (compute == mblas_hipblas_compute_type::MBLAS_COMPUTE_32F && a_type.is_fp8() && b_type.is_fp8()) {
    // Special FP8 type filtering
    matmul_prec_type_f8 selTypeF8 = {
        scalar, c_type, d_type, mblas_hip_data_type(MBLAS_ANY)};
    auto result = std::find(begin(matmulSupportedF8), end(matmulSupportedF8), selTypeF8);
    if (result != end(matmulSupportedF8)) {
      return;
    }
  }
#if HIP_VERSION >= 70000000
  else if (a_type.is_mx_possible() && b_type.is_mx_possible()) {
    // Check MX format configurations
    auto mx_result = std::find(begin(mx_matmul_supported), end(mx_matmul_supported), selType);
    if (mx_result != end(mx_matmul_supported)) {
      return;
    }
  }
  
  // Validate MX format constraints (initial implementation)
  if (use_scaling) {
    // Check batch size = 1
    if (batch_count != 1) {
      throw std::invalid_argument("MX formats currently only support batch_count=1");
    }
    
    // Check M, N divisible by 16
    if (m % 16 != 0 || n % 16 != 0) {
      throw std::invalid_argument("MX formats require M and N divisible by 16");
    }
    
    // Check K divisible by 32
    if (k % 32 != 0) {
      throw std::invalid_argument("MX formats require K divisible by 32");
    }
    
    // Check C and D types (must be FP32, FP16, or BF16)
    if (!(c_type == mblas_hip_data_type::MBLAS_R_32F || c_type == mblas_hip_data_type::MBLAS_R_16F || c_type == mblas_hip_data_type::MBLAS_R_16BF) ||
        !(d_type == mblas_hip_data_type::MBLAS_R_32F || d_type == mblas_hip_data_type::MBLAS_R_16F || d_type == mblas_hip_data_type::MBLAS_R_16BF)) {
      throw std::invalid_argument("MX formats require C and D types to be FP32, FP16, or BF16");
    }
  }
#endif
  // Unable to find matching config, not supported
  string errorString =
      "Invalid GEMM specification for MatMul.  Combination of parameters "
      "not supported"
      "\nCompute type: " +
      compute.to_string() + "\nScalar type: " + scalar.to_string() +
      "\nA type: " + a_type.to_string() +
      "\nB type: " + b_type.to_string() +
      "\nC type: " + c_type.to_string() +
      "\nD type: " + d_type.to_string();
  throw std::invalid_argument(errorString);
}

hipblaslt_gemm::hipblaslt_gemm(cxxopts::ParseResult result) : generic_gemm(result),
  scale_host_a(nullptr), scale_host_b(nullptr), scale_host_c(nullptr), scale_host_d(nullptr) {
  // Grab precision from command line
  precision = mblas_hip_data_type(result["precision"].as<string>());
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
  transA = mblas_hipblas_operation(result["transposeA"].as<std::string>());
  transB = mblas_hipblas_operation(result["transposeB"].as<std::string>());
  
#if HIP_VERSION >= 70000000
  use_scaling = a_type.is_mx_possible() || b_type.is_mx_possible() || 
                c_type.is_mx_possible() || d_type.is_mx_possible();
  if (use_scaling) {
    std::tie(a_scale_type, a_scale_mode, a_scale_size) = configure_scaling(a_props, a_type, "A");
    std::cout << "a_scale_type: " << a_scale_type.to_string() << std::endl;
    std::tie(b_scale_type, b_scale_mode, b_scale_size) = configure_scaling(b_props, b_type, "B");
    std::cout << "b_scale_type: " << b_scale_type.to_string() << std::endl;
    std::tie(c_scale_type, c_scale_mode, c_scale_size) = configure_scaling(c_props, c_type, "C");
    std::tie(d_scale_type, d_scale_mode, d_scale_size) = configure_scaling(d_props, d_type, "D");
  }
#endif
  
  validate_parameters();

  // Pull in alpha and beta, alloc memory and save to pointers
  string salpha = result["alpha"].as<string>();
  string salphai = result["alphai"].as<string>();
  alpha = malloc(get_malloc_size_scalar(precision));
  type_call_host<set_scalar>(precision, alpha, salpha, salphai);
  
  string sbeta = result["beta"].as<string>();
  string sbetai = result["betai"].as<string>();
  beta = malloc(get_malloc_size_scalar(precision));
  type_call_host<set_scalar>(precision, beta, sbeta, sbetai);
  // std::cout << *((float *)alpha) << std::endl;
  // std::cout << *((float *)beta) << std::endl;
  uint64_t a_offset, b_offset, c_offset, d_offset;
  set_flush_batch_count( 
      type_call_dev<sizeofCUDT>(a_type), type_call_dev<sizeofCUDT>(b_type), 
      type_call_dev<sizeofCUDT>(c_type), type_call_dev<sizeofCUDT>(d_type), 
      a_type.get_packing_count(), 
      b_type.get_packing_count(), 
      c_type.get_packing_count(), 
      d_type.get_packing_count(), 
      inplace);
}

string hipblaslt_gemm::prepare_array() {
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
  run_threaded(&hipblaslt_gemm::alloc_dev);
  run_threaded(&hipblaslt_gemm::copy_host_to_dev);
  run_threaded(&hipblaslt_gemm::prepare_matrix);
  // Enable tuning with a parameter later
  if (false) {
  } else {
    run_threaded(&hipblaslt_gemm::no_tuning);
  }
  std::ostringstream ossHeader;
  ossHeader << "transA_option,transB_option,M,N,K,lda,ldb,ldc,";
  if (batched) {
    ossHeader << "batch_count,";
  }
  ossHeader << "hipBLASLt-Gflops,hipBLASLt-GB/s,hipBLASLt-us," << endl;
  return ossHeader.str();
}

void hipblaslt_gemm::run_threaded(void (hipblaslt_gemm::*func)(hipblaslt_gemm_inst *)) {
  vector<thread> threads;
  for (auto &instance : mat_ptrs) {
    threads.push_back(thread(func, this, &instance));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void hipblaslt_gemm::alloc_host() {
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
    ptr_host_a[i] = malloc(get_malloc_size_host(a_type, rows_mem_a, cols_mem_a, batch_count));
    ptr_host_b[i] = malloc(get_malloc_size_host(b_type, rows_mem_b, cols_mem_b, batch_count));
    ptr_host_c[i] = malloc(get_malloc_size_host(c_type, rows_mem_c, cols_mem_c, batch_count));
    if (!inplace) {
      ptr_host_d[i] = malloc(get_malloc_size_host(d_type, rows_mem_d, cols_mem_d, batch_count));
    }
  }
  
#if HIP_VERSION >= 70000000
  if (a_props.scale_mode != scaling_type::None) {
    std::cout << "a_scale_size.rows: " << a_scale_size.rows << std::endl;
    std::cout << "a_scale_size.cols: " << a_scale_size.cols << std::endl;
    std::cout << "a_scale_size.get_size: " << a_scale_size.get_size() << std::endl;
    scale_host_a = malloc(a_scale_size.get_size()*type_call_host<sizeofCUDT>(a_scale_type));
  }
  if (b_props.scale_mode != scaling_type::None) {
    scale_host_b = malloc(b_scale_size.get_size()*type_call_host<sizeofCUDT>(b_scale_type));
  }
  if (c_props.scale_mode != scaling_type::None) {
    scale_host_c = malloc(c_scale_size.get_size()*type_call_host<sizeofCUDT>(c_scale_type));
  }
  if (d_props.scale_mode != scaling_type::None) {
    scale_host_d = malloc(d_scale_size.get_size()*type_call_host<sizeofCUDT>(d_scale_type));
  }
#endif
}

void hipblaslt_gemm::alloc_dev(hipblaslt_gemm_inst *mat) {
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
    hipMalloc(&mat->ptr_dev_a[i], get_malloc_size_dev(a_type, rows_mem_a, cols_mem_a, batch_count));
    hipMalloc(&mat->ptr_dev_b[i], get_malloc_size_dev(b_type, rows_mem_b, cols_mem_b, batch_count));
    hipMalloc(&mat->ptr_dev_c[i], get_malloc_size_dev(c_type, rows_mem_c, cols_mem_c, batch_count));
    if (!inplace) {
      hipMalloc(&mat->ptr_dev_d[i], get_malloc_size_dev(d_type, rows_mem_d, cols_mem_d, batch_count));
    }
  }
  mat->wSZ = workspace_size;
  hipMalloc(&mat->devWork, mat->wSZ);
  
#if HIP_VERSION >= 70000000
  if (a_props.scale_mode != scaling_type::None) {
    hipMalloc(&mat->scale_dev_a, a_scale_size.get_size()*type_call_dev<sizeofCUDT>(a_scale_type));
  }
  if (b_props.scale_mode != scaling_type::None) {
    hipMalloc(&mat->scale_dev_b, b_scale_size.get_size()*type_call_dev<sizeofCUDT>(b_scale_type));
  }
  if (c_props.scale_mode != scaling_type::None) {
    hipMalloc(&mat->scale_dev_c, c_scale_size.get_size()*type_call_dev<sizeofCUDT>(c_scale_type));
  }
  if (d_props.scale_mode != scaling_type::None) {
    hipMalloc(&mat->scale_dev_d, d_scale_size.get_size()*type_call_dev<sizeofCUDT>(d_scale_type));
  }
#endif
}

void hipblaslt_gemm::fill_host() {
  type_call_host<initHost>(a_type, initialization, ptr_host_a, rows_a, cols_a, lda,
                         batch_count, stride_a, flush_batch_count, control_a, constant_a, filename_a);
  type_call_host<initHost>(b_type, initialization, ptr_host_b, rows_b, cols_b, ldb,
                         batch_count, stride_b, flush_batch_count, control_b, constant_b, filename_b);
  type_call_host<initHost>(c_type, initialization, ptr_host_c, rows_c, cols_c, ldc,
                         batch_count, stride_c, flush_batch_count, control_c, constant_c, filename_c);
#if HIP_VERSION >= 70000000
  // Initialize scale factors to 1.0 for all MX formats
  if (a_props.scale_mode != scaling_type::None) {
    std::cout << "a_scale_size.rows: " << a_scale_size.rows << std::endl;
    std::cout << "a_scale_size.cols: " << a_scale_size.cols << std::endl;
    std::cout << "a_scale_size.get_size: " << a_scale_size.get_size() << std::endl;
    type_call_host<initHost>(a_scale_type, "constant", &scale_host_a, a_scale_size.rows, a_scale_size.cols,
                            a_scale_size.rows, 1, 0, 0.0, 1.0, "");
  }
  if (b_props.scale_mode != scaling_type::None) {
    std::cout << "b_scale_size.rows: " << b_scale_size.rows << std::endl;
    std::cout << "b_scale_size.cols: " << b_scale_size.cols << std::endl;
    type_call_host<initHost>(b_scale_type, "constant", &scale_host_b, b_scale_size.rows, b_scale_size.cols,
                            b_scale_size.rows, 1, 0, 0.0, 1.0, "");
  }
  if (c_props.scale_mode != scaling_type::None) {
    type_call_host<initHost>(c_scale_type, "constant", &scale_host_c, c_scale_size.rows, c_scale_size.cols,
                            c_scale_size.cols, 1, 0, 0.0, 1.0, "");
  }
  if (d_props.scale_mode != scaling_type::None) {
    type_call_host<initHost>(d_scale_type, "constant", &scale_host_d, d_scale_size.rows, d_scale_size.cols,
                            d_scale_size.cols, 1, 0, 0.0, 1.0, "");
  }
#endif
}

void hipblaslt_gemm::copy_host_to_dev(hipblaslt_gemm_inst *mat) {
  hipSetDevice(mat->devIDX);
  for (int i = 0; i < flush_batch_count; i++) {
    copy_and_convert(a_type, ptr_host_a[i], mat->ptr_dev_a[i], rows_mem_a, cols_mem_a, batch_count);
    copy_and_convert(b_type, ptr_host_b[i], mat->ptr_dev_b[i], rows_mem_b, cols_mem_b, batch_count);
    copy_and_convert(c_type, ptr_host_c[i], mat->ptr_dev_c[i], rows_mem_c, cols_mem_c, batch_count);
  }
  
#if HIP_VERSION >= 70000000
  // Copy scale factors to device
  if (a_props.scale_mode != scaling_type::None) {
    copy_and_convert(a_scale_type, scale_host_a, mat->scale_dev_a, a_scale_size.rows, a_scale_size.cols, 1);
  }
  if (b_props.scale_mode != scaling_type::None) {
    copy_and_convert(b_scale_type, scale_host_b, mat->scale_dev_b, b_scale_size.rows, b_scale_size.cols, 1);
  }
  if (c_props.scale_mode != scaling_type::None) {
    copy_and_convert(c_scale_type, scale_host_c, mat->scale_dev_c, c_scale_size.rows, c_scale_size.cols, 1);
  }
  if (d_props.scale_mode != scaling_type::None) {
    copy_and_convert(d_scale_type, scale_host_d, mat->scale_dev_d, d_scale_size.rows, d_scale_size.cols, 1);
  }
#endif
}

void hipblaslt_gemm::prepare_matrix(hipblaslt_gemm_inst *mat) {
  check_hipblas(hipblasLtMatmulDescCreate(&mat->desc_op, compute, scalar));
  // These values are read in with no type, so they need to be convirted first
  // Thanks for the wonderful standard Nvidia :D!
  hipblasOperation_t transA_local = transA.convert_to_hip();
  hipblasOperation_t transB_local = transB.convert_to_hip();
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
        hipblasLtMatrixLayoutCreate(&mat->desc_d, d_type, rows_d, cols_d, ldd));
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
      
#if HIP_VERSION >= 70000000
  // Set scaling attributes for MX formats
  if (a_props.scale_mode != scaling_type::None) {
    check_hipblas(hipblasLtMatmulDescSetAttribute(mat->desc_op,  
        HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &a_scale_mode, sizeof(a_scale_mode)));
    //check_hipblas(hipblasLtMatmulDescSetAttribute(mat->desc_a, 
    //    HIPBLASLT_MATRIX_LAYOUT_MATRIX_SCALE_TYPE, &a_scale_type, sizeof(a_scale_type)));
    check_hipblas(hipblasLtMatmulDescSetAttribute(mat->desc_op, 
        HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &mat->scale_dev_a, sizeof(mat->scale_dev_a)));
  }
  if (b_props.scale_mode != scaling_type::None) {
    check_hipblas(hipblasLtMatmulDescSetAttribute(mat->desc_op, 
        HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &b_scale_mode, sizeof(b_scale_mode)));
    //check_hipblas(hipblasLtMatmulDescSetAttribute(mat->desc_b, 
    //    HIPBLASLT_MATRIX_LAYOUT_MATRIX_SCALE_TYPE, &b_scale_type, sizeof(b_scale_type)));
    check_hipblas(hipblasLtMatmulDescSetAttribute(mat->desc_op, 
        HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &mat->scale_dev_b, sizeof(mat->scale_dev_b)));
  }
  //if (c_props.scale_mode != scaling_type::None) {
  //  check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_c, 
  //      HIPBLASLT_MATRIX_LAYOUT_MATRIX_SCALE_MODE, &c_scale_mode, sizeof(c_scale_mode)));
  //  check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_c, 
  //      HIPBLASLT_MATRIX_LAYOUT_MATRIX_SCALE_TYPE, &c_scale_type, sizeof(c_scale_type)));
  //  check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_c, 
  //      HIPBLASLT_MATRIX_LAYOUT_MATRIX_SCALE, &mat->scale_dev_c, sizeof(mat->scale_dev_c)));
  //}
  //if (d_props.scale_mode != scaling_type::None && !inplace) {
  //  check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_d, 
  //      HIPBLASLT_MATRIX_LAYOUT_MATRIX_SCALE_MODE, &d_scale_mode, sizeof(d_scale_mode)));
  //  check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_d, 
  //      HIPBLASLT_MATRIX_LAYOUT_MATRIX_SCALE_TYPE, &d_scale_type, sizeof(d_scale_type)));
  //  check_hipblas(hipblasLtMatrixLayoutSetAttribute(mat->desc_d, 
  //      HIPBLASLT_MATRIX_LAYOUT_MATRIX_SCALE, &mat->scale_dev_d, sizeof(mat->scale_dev_d)));
  //}
#endif
}

void hipblaslt_gemm::no_tuning(hipblaslt_gemm_inst *mat) {
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
  hipblasLtDestroy(handle);
}
void hipblaslt_gemm::auto_tuning(hipblaslt_gemm_inst *mat) {
  // Not currently implemented, using simple method
  no_tuning(mat);
}

void hipblaslt_gemm::free_mem() {
  free(alpha);
  free(beta);
  for (int i = 0; i < flush_batch_count; i++) {
    free(ptr_host_a[i]);
    free(ptr_host_b[i]);
    free(ptr_host_c[i]);
    if (!inplace) {
      free(ptr_host_d[i]);
    }
  }
  free(ptr_host_a);
  free(ptr_host_b);
  free(ptr_host_c);
  if (!inplace) {
    free(ptr_host_d);
  }
  
#if HIP_VERSION >= 70000000
  // Free scale host buffers
  if (scale_host_a) free(scale_host_a);
  if (scale_host_b) free(scale_host_b);
  if (scale_host_c) free(scale_host_c);
  if (scale_host_d) free(scale_host_d);
#endif
  
  for (auto mat : mat_ptrs) {
    hipSetDevice(mat.devIDX);
    for (int i = 0; i < flush_batch_count; i++) {
      hipFree(mat.ptr_dev_a[i]);
      hipFree(mat.ptr_dev_b[i]);
      hipFree(mat.ptr_dev_c[i]);
      if (!inplace) {
        hipFree(mat.ptr_dev_d[i]);
      }
    }
    free(mat.ptr_dev_a);
    free(mat.ptr_dev_b);
    free(mat.ptr_dev_c);
    if (!inplace) {
      free(mat.ptr_dev_d);
    }
    hipFree(mat.devWork);
    
#if HIP_VERSION >= 70000000
    // Free scale device buffers
    if (mat.scale_dev_a) hipFree(mat.scale_dev_a);
    if (mat.scale_dev_b) hipFree(mat.scale_dev_b);
    if (mat.scale_dev_c) hipFree(mat.scale_dev_c);
    if (mat.scale_dev_d) hipFree(mat.scale_dev_d);
#endif
    
    hipblasLtMatmulDescDestroy(mat.desc_op);
    hipblasLtMatrixLayoutDestroy(mat.desc_a);
    hipblasLtMatrixLayoutDestroy(mat.desc_b);
    hipblasLtMatrixLayoutDestroy(mat.desc_c);
    if (!inplace) {
      hipblasLtMatrixLayoutDestroy(mat.desc_d);
    }
    hipblasLtMatmulPreferenceDestroy(mat.pref);
  }
}

double hipblaslt_gemm::test() {
  vector<thread> threads;
  double gflops = 0.0;
  for (auto &mat : mat_ptrs) {
    threads.push_back(thread(&hipblaslt_gemm::test_matmul, this, &mat));
  }
  // Wait on running jobs
  for (auto &thread : threads) {
    thread.join();
  }

  // Sum all gflops
  gflop_per_second = std::accumulate(
      begin(mat_ptrs), end(mat_ptrs), 0.0,
      [](double i, const hipblaslt_gemm_inst &o) { return o.gflops + i; });

  gbyte_per_second = std::accumulate(
      begin(mat_ptrs), end(mat_ptrs), 0.0,
      [](double i, const hipblaslt_gemm_inst &o) { return o.gbytes + i; });

  iter_time_us = std::accumulate(begin(mat_ptrs), end(mat_ptrs), 0.0,
                                 [](double i, const hipblaslt_gemm_inst &o) {
                                   return o.time_us + i;
                                 }) /
                 mat_ptrs.size();

  return gflop_per_second;
}

std::string hipblaslt_gemm::get_result_string() {
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

std::tuple<double, double, double> hipblaslt_gemm::calculate_figure_of_merit(
    double totalTime_ms) {
  double avgTime_ms = totalTime_ms / iters;
  double avgTime_s = avgTime_ms / 1000.0f;
  double avgTime_us = avgTime_ms * 1000.0f;

  int a_sz = type_call_dev<sizeofCUDT>(a_type);
  int b_sz = type_call_dev<sizeofCUDT>(b_type);
  int c_sz = type_call_dev<sizeofCUDT>(c_type);

  int flopPerSize = 2;
  if (!precision.is_real()) {
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

void hipblaslt_gemm::test_matmul(hipblaslt_gemm_inst *mat) {
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
  
  // Cleanup
  check_hip(hipEventDestroy(start));
  check_hip(hipEventDestroy(stop));
  check_hip(hipStreamDestroy(stream));
  check_hipblas(hipblasLtDestroy(handle));
}

//#if HIP_VERSION >= 70000000
//std::tuple<mblas_hip_data_type, hipblasLtMatmulMatrixScale_t, scale_size>
//hipblaslt_gemm::configure_scaling(matrix_desc desc, mblas_hip_data_type type, std::string matrix_id) {
//  mblas_hip_data_type scale_type = type.get_scale_type();
//  hipblasLtMatmulMatrixScale_t scale_mode = get_scale_mode(type);
//  
//  // Get scale tensor dimensions based on matrix dimensions and scale mode
//  int rows = (matrix_id == "A") ? rows_a : 
//             (matrix_id == "B") ? rows_b :
//             (matrix_id == "C") ? rows_c : rows_d;
//  int cols = (matrix_id == "A") ? cols_a : 
//             (matrix_id == "B") ? cols_b :
//             (matrix_id == "C") ? cols_c : cols_d;
//  
//  scale_size sz = get_scale_tensor_size(rows, cols, scale_mode);
//  
//  return std::make_tuple(scale_type, scale_mode, sz);
//}
//#endif
