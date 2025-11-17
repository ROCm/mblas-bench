#include "cublaslt_gemm.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <future>
#include <iomanip>
#include <numeric>
#include <regex>
#include <string>
#include <thread>

#include "generic_setup.h"
#include "cublas_convert.h"
#include "cublas_create_allocate.h"
#include "cublas_datatype_utils.h"
#include "cuda_error.h"
#include "cxxopts.hpp"
#include "cuda_monitor.h"

using std::cerr;
using std::cout;
using std::endl;
using std::move;
using std::string;
using std::thread;
using std::vector;

// clang-format off
std::vector<matmul_prec_type> cublaslt_gemm::matmul_supported = {
  // Compute type                   Scale Type    A Type            B Type            C Type        D Type            Bias Type
  {mblas_compute_type::MBLAS_COMPUTE_16F,              mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F},
  {mblas_compute_type::MBLAS_COMPUTE_16F_PEDANTIC,     mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F},
  {mblas_compute_type::MBLAS_COMPUTE_32I,              mblas_data_type::MBLAS_R_32I,   mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_32I,   mblas_data_type::MBLAS_R_32I,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32I_PEDANTIC,     mblas_data_type::MBLAS_R_32I,   mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_32I,   mblas_data_type::MBLAS_R_32I,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32I,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_8I,    mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32I_PEDANTIC,     mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_8I,    mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32F ,             mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_16BF,      mblas_data_type::MBLAS_R_16BF,      mblas_data_type::MBLAS_R_16BF,  mblas_data_type::MBLAS_R_16BF,      mblas_data_type::MBLAS_R_16BF},
  {mblas_compute_type::MBLAS_COMPUTE_32F_PEDANTIC,     mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_16BF,      mblas_data_type::MBLAS_R_16BF,      mblas_data_type::MBLAS_R_16BF,  mblas_data_type::MBLAS_R_16BF,      mblas_data_type::MBLAS_R_16BF},
  {mblas_compute_type::MBLAS_COMPUTE_32F ,             mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F},
  {mblas_compute_type::MBLAS_COMPUTE_32F_PEDANTIC,     mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F},
  {mblas_compute_type::MBLAS_COMPUTE_32F ,             mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32F_PEDANTIC,     mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32F ,             mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_16BF,      mblas_data_type::MBLAS_R_16BF,      mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F},
  {mblas_compute_type::MBLAS_COMPUTE_32F_PEDANTIC,     mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_16BF,      mblas_data_type::MBLAS_R_16BF,      mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F},
  {mblas_compute_type::MBLAS_COMPUTE_32F ,             mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F},
  {mblas_compute_type::MBLAS_COMPUTE_32F_PEDANTIC,     mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F},
  {mblas_compute_type::MBLAS_COMPUTE_32F ,             mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F},
  {mblas_compute_type::MBLAS_COMPUTE_32F_PEDANTIC,     mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F},
  {mblas_compute_type::MBLAS_COMPUTE_32F ,             mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_8I,        mblas_data_type::MBLAS_C_8I,        mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32F_PEDANTIC,     mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_8I,        mblas_data_type::MBLAS_C_8I,        mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32F ,             mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32F_PEDANTIC,     mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32F_FAST_16F,     mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F},
  {mblas_compute_type::MBLAS_COMPUTE_32F_FAST_16BF,    mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F},
  {mblas_compute_type::MBLAS_COMPUTE_32F_FAST_TF32,    mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F},
  {mblas_compute_type::MBLAS_COMPUTE_32F_EMULATED_16BFX9,    mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_32F},
  {mblas_compute_type::MBLAS_COMPUTE_32F_FAST_16F,     mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32F_FAST_16BF,    mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32F_FAST_TF32,    mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32F_EMULATED_16BFX9,    mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_64F,              mblas_data_type::MBLAS_R_64F,   mblas_data_type::MBLAS_R_64F,       mblas_data_type::MBLAS_R_64F,       mblas_data_type::MBLAS_R_64F,   mblas_data_type::MBLAS_R_64F,       mblas_data_type::MBLAS_R_64F},
  {mblas_compute_type::MBLAS_COMPUTE_64F_PEDANTIC,     mblas_data_type::MBLAS_R_64F,   mblas_data_type::MBLAS_R_64F,       mblas_data_type::MBLAS_R_64F,       mblas_data_type::MBLAS_R_64F,   mblas_data_type::MBLAS_R_64F,       mblas_data_type::MBLAS_R_64F},
  {mblas_compute_type::MBLAS_COMPUTE_64F,              mblas_data_type::MBLAS_C_64F,   mblas_data_type::MBLAS_C_64F,       mblas_data_type::MBLAS_C_64F,       mblas_data_type::MBLAS_C_64F,   mblas_data_type::MBLAS_C_64F,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_64F_PEDANTIC,     mblas_data_type::MBLAS_C_64F,   mblas_data_type::MBLAS_C_64F,       mblas_data_type::MBLAS_C_64F,       mblas_data_type::MBLAS_C_64F,   mblas_data_type::MBLAS_C_64F,       mblas_data_type::MBLAS_ANY },
  // IMMA kernels
  // Compute type                   Scale Type    A Type            B Type            C Type        D Type            Bias Type
  {mblas_compute_type::MBLAS_COMPUTE_32I,              mblas_data_type::MBLAS_R_32I,   mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_32I,   mblas_data_type::MBLAS_R_32I,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32I_PEDANTIC,     mblas_data_type::MBLAS_R_32I,   mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_32I,   mblas_data_type::MBLAS_R_32I,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32I,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_8I,    mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32I_PEDANTIC,     mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_R_8I,    mblas_data_type::MBLAS_R_8I,        mblas_data_type::MBLAS_ANY},
  // FP8 kernels
  // Compute type                   Scale Type    A Type            B Type            C Type        D Type            Bias Type
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16BF,  mblas_data_type::MBLAS_R_16BF,      mblas_data_type::MBLAS_R_16BF},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16BF,  mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16BF},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16F},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_16BF},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_16BF,  mblas_data_type::MBLAS_R_16BF,      mblas_data_type::MBLAS_R_16BF},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_16BF,  mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16BF},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_16BF,  mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_16BF},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16F},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_16F},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_16BF},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16BF,  mblas_data_type::MBLAS_R_16BF,      mblas_data_type::MBLAS_R_16BF},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16BF,  mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16BF},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16BF,  mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_16BF},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16F},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_16F},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_8F_E5M2,   mblas_data_type::MBLAS_R_8F_E4M3,   mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_16BF},
#if (ENABLE_CUDA_FP4)
  // FP4 Kernels
  // Compute type                   Scale Type    A Type            B Type            C Type        D Type            Bias Type
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_4F_E2M1,   mblas_data_type::MBLAS_R_4F_E2M1,   mblas_data_type::MBLAS_R_16BF,   mblas_data_type::MBLAS_R_4F_E2M1,       mblas_data_type::MBLAS_R_16BF},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_4F_E2M1,   mblas_data_type::MBLAS_R_4F_E2M1,   mblas_data_type::MBLAS_R_16BF,   mblas_data_type::MBLAS_R_16BF,       mblas_data_type::MBLAS_R_16BF},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_4F_E2M1,   mblas_data_type::MBLAS_R_4F_E2M1,   mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_4F_E2M1,       mblas_data_type::MBLAS_R_16F},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_4F_E2M1,   mblas_data_type::MBLAS_R_4F_E2M1,   mblas_data_type::MBLAS_R_16F,   mblas_data_type::MBLAS_R_16F,       mblas_data_type::MBLAS_R_16F},
  {mblas_compute_type::MBLAS_COMPUTE_32F,              mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_4F_E2M1,   mblas_data_type::MBLAS_R_4F_E2M1,   mblas_data_type::MBLAS_R_32F,   mblas_data_type::MBLAS_R_32F,       mblas_data_type::MBLAS_R_16BF},
#endif
  // Mixed precision complex kernels
  // Compute type                   Scale Type    A Type            B Type            C Type        D Type            Bias Type
  {mblas_compute_type::MBLAS_COMPUTE_32F ,             mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_16F,       mblas_data_type::MBLAS_C_16F,       mblas_data_type::MBLAS_C_16F,   mblas_data_type::MBLAS_C_16F,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32F ,             mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_16F,       mblas_data_type::MBLAS_C_16F,       mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32F ,             mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_16BF,      mblas_data_type::MBLAS_C_16BF,      mblas_data_type::MBLAS_C_16BF,  mblas_data_type::MBLAS_C_16BF,      mblas_data_type::MBLAS_ANY},
  {mblas_compute_type::MBLAS_COMPUTE_32F ,             mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_16BF,      mblas_data_type::MBLAS_C_16BF,      mblas_data_type::MBLAS_C_32F,   mblas_data_type::MBLAS_C_32F,       mblas_data_type::MBLAS_ANY},
};
// clang-format on

void cublaslt_gemm::parse_dev_iters(std::string deviceStr) {
  // Parse device
  std::stringstream ss(deviceStr);
  while (ss.good()) {
    string deviceSStr;
    getline(ss, deviceSStr, ',');
    int devInt = stoi(deviceSStr);
    cublaslt_gemm_inst val = cublaslt_gemm_inst(devInt);
    mat_ptrs.push_back(val);
  }
}

std::tuple<mblas_cuda_data_type, cublasLtMatmulMatrixScale_t, scale_size> cublaslt_gemm::configure_scaling(matrix_desc desc, mblas_cuda_data_type type, string matrix_id) {
  mblas_cuda_data_type scale_type;
  cublasLtMatmulMatrixScale_t scale_mode;
  scale_size scale_size;
  if (desc.scale_mode == scaling_type::Block){
    // Determine scale types (calculated from a,b,c,d type)
    scale_type = type.get_scale_type();

    // Scale modes
    scale_mode = type.get_scale_mode();

    // Calculate lengths
    scale_size = get_scale_tensor_size(desc.rows_mem, desc.cols_mem, scale_mode);
  } else if (type.is_fp4()) {
    string errorString =
        "Non-block scaled fp4 is not supported in cublaslt"
        "Matrix: " + matrix_id +
        "\nType: " + type.to_string();
    std::cerr << scaling_string(desc.scale_mode) << std::endl;
    throw std::invalid_argument(errorString);
  } else if (desc.scale_mode == scaling_type::Vector) {
#if (CUDART_VERSION >= 12090)
    // Dependent on if this is the A or B matrix
    // Use the columns for B, rows for everything else (A,C,D)
    scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F; 
    //long scaling_vec_len = (matrix_id == "B") ? desc.cols : desc.rows;
    //long scaling_vec_len = (matrix_id == "A") ? desc.cols : desc.rows;
    //std::cout << ((matrix_id == "B") ? n : m) << std::endl;
    long scaling_vec_len = (matrix_id == "B") ? n : m;
    scale_size = std::make_pair<size_t, size_t>(1, scaling_vec_len);
    scale_type = MBLAS_R_32F;
#else
    string errorString =
        "Vector scaling mode requires CUDA 12.9.0 or later. "
        "Matrix: " + matrix_id +
        "\nType: " + type.to_string();
    std::cerr << scaling_string(desc.scale_mode) << std::endl;
    throw std::invalid_argument(errorString);
#endif
  } else if (desc.scale_mode == scaling_type::Scalar) {
    scale_size = std::make_pair<size_t, size_t>(1, 1);
    scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F; 
    scale_type = MBLAS_R_32F;
  } else {
    scale_size.rows = 0;
    scale_size.cols = 0;
  }
    
  return std::make_tuple(scale_type, scale_mode, scale_size);
}

void cublaslt_gemm::parse_problem_type(string computeTStr, string scalarTStr,
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
    a_type = mblas_cuda_data_type(aStr);
    b_type = mblas_cuda_data_type(bStr);
    c_type = mblas_cuda_data_type(cStr);
    d_type = mblas_cuda_data_type(dStr);
  }

#if (ENABLE_CUDA_FP4)
  use_scaling = a_type.is_fp4() || b_type.is_fp4() || c_type.is_fp4() || d_type.is_fp4();
  std::tie(a_scale_type, a_scale_mode, a_scale_size) = configure_scaling(a_props, a_type, "A");
  std::tie(b_scale_type, b_scale_mode, b_scale_size) = configure_scaling(b_props, b_type, "B");
  std::tie(c_scale_type, c_scale_mode, c_scale_size) = configure_scaling(c_props, c_type, "C");
  std::tie(d_scale_type, d_scale_mode, d_scale_size) = configure_scaling(d_props, d_type, "D");
#endif
}

void cublaslt_gemm::validate_parameters() {
  // Validate that data types exist in table of supported configurations
  matmul_prec_type selType = {
      compute, scalar, a_type, b_type, c_type, d_type, mblas_data_type::MBLAS_ANY};
  auto result =
      std::find(begin(matmul_supported), end(matmul_supported), selType);
  if (result == end(matmul_supported)) {
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
  // Validate that FP8 kernels will use TN format only
  // GEMM fails if not
  if ((a_type.is_fp8() || b_type.is_fp8() || c_type.is_fp8() || d_type.is_fp8()) &&
      (transA != mblas_operation::MBLAS_OP_T || transB != mblas_operation::MBLAS_OP_N)) {
    string errorString =
        "Transpose operation selection not supported"
        "\nOnly TN format is supported"
        "\nTransA: " +
        transA.to_string() + "\nTransB: " + transB.to_string();
    throw std::invalid_argument(errorString);
  }
}

cublaslt_gemm::cublaslt_gemm(cxxopts::ParseResult result) : generic_gemm(result) {
  // Grab precision from command line
  precision = mblas_cuda_data_type(result["precision"].as<string>());
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
  transA = mblas_cuda_operation(result["transposeA"].as<std::string>());
  transB = mblas_cuda_operation(result["transposeB"].as<std::string>());
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
  set_flush_batch_count( 
      type_call_dev<sizeofCUDT>(a_type), type_call_dev<sizeofCUDT>(b_type), 
      type_call_dev<sizeofCUDT>(c_type), type_call_dev<sizeofCUDT>(d_type), 
      a_type.get_packing_count(), 
      b_type.get_packing_count(), 
      c_type.get_packing_count(), 
      d_type.get_packing_count(), 
      inplace);
}

string cublaslt_gemm::prepare_array() {
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
  run_threaded(&cublaslt_gemm::alloc_dev);
  run_threaded(&cublaslt_gemm::copy_host_to_dev);
  run_threaded(&cublaslt_gemm::prepare_matrix);
  // Enable tuning with a parameter later
  if (false) {
  } else {
    run_threaded(&cublaslt_gemm::no_tuning);
  }
  std::ostringstream ossHeader;
  ossHeader << "transA_option,transB_option,M,N,K,lda,ldb,ldc,ldd,";
  // if (batched) {
    ossHeader << "batch_count,";
  // }
  ossHeader << "alpha,beta,";
  ossHeader << "a_type,b_type,c_type,d_type,compute_type,scalar_type,";
  ossHeader << "a_scale_type,b_scale_type,c_scale_type,d_scale_type,bias_type,";
  ossHeader << "rotating_buffer,";
  ossHeader << "cuBLAS-Gflops,cuBLAS-GB/s,cuBLAS-us,";
  if (cuda_monitor::monitor::enabled()) {
    ossHeader << "avg_sysclk_mhz,med_sysclk_mhz,avg_memclk_mhz,med_memclk_mhz,";
  }
  ossHeader << endl;
  return ossHeader.str();
}

void cublaslt_gemm::run_threaded(void (cublaslt_gemm::*func)(cublaslt_gemm_inst *)) {
  vector<thread> threads;
  for (auto &instance : mat_ptrs) {
    threads.push_back(thread(func, this, &instance));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void cublaslt_gemm::alloc_host() {
  ptr_host_a =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(a_type));
  ptr_host_b =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(b_type));
  ptr_host_c =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(c_type));
  ptr_host_d =
      (void **)malloc(flush_batch_count * type_call_host<sizeofCUDTP>(d_type));

  for (int i = 0; i < flush_batch_count; i++) {
    ptr_host_a[i] = malloc(get_malloc_size_host(a_type, rows_mem_a, cols_mem_a, batch_count));
    ptr_host_b[i] = malloc(get_malloc_size_host(b_type, rows_mem_b, cols_mem_b, batch_count));
    ptr_host_c[i] = malloc(get_malloc_size_host(c_type, rows_mem_c, cols_mem_c, batch_count));
    ptr_host_d[i] = malloc(get_malloc_size_host(d_type, rows_mem_d, cols_mem_d, batch_count));
  }

  //if (use_scaling) {
  //  scale_host_a = malloc(a_scale_size.get_size()*type_call_host<sizeofCUDT>(a_scale_type));
  //  scale_host_b = malloc(b_scale_size.get_size()*type_call_host<sizeofCUDT>(b_scale_type));
  //  scale_host_c = malloc(c_scale_size.get_size()*type_call_host<sizeofCUDT>(c_scale_type));
  //  scale_host_d = malloc(d_scale_size.get_size()*type_call_host<sizeofCUDT>(d_scale_type));
  //}

  if (a_props.scale_mode != scaling_type::None) {
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
}

void cublaslt_gemm::alloc_dev(cublaslt_gemm_inst *mat) {
  cudaSetDevice(mat->devIDX);

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
    cudaMalloc(&mat->ptr_dev_a[i], get_malloc_size_dev(a_type, rows_mem_a, cols_mem_a, batch_count));
    cudaMalloc(&mat->ptr_dev_b[i], get_malloc_size_dev(b_type, rows_mem_b, cols_mem_b, batch_count));
    cudaMalloc(&mat->ptr_dev_c[i], get_malloc_size_dev(c_type, rows_mem_c, cols_mem_c, batch_count));
    cudaMalloc(&mat->ptr_dev_d[i], get_malloc_size_dev(d_type, rows_mem_d, cols_mem_d, batch_count));
  }

  mat->wSZ = workspace_size;
  cudaMalloc(&mat->devWork, mat->wSZ);
  // if (use_scaling) {
  //   cudaMalloc(&mat->scale_dev_a, a_scale_size.get_size()*type_call_dev<sizeofCUDT>(a_scale_type));
  //   cudaMalloc(&mat->scale_dev_b, b_scale_size.get_size()*type_call_dev<sizeofCUDT>(b_scale_type));
  //   cudaMalloc(&mat->scale_dev_c, c_scale_size.get_size()*type_call_dev<sizeofCUDT>(c_scale_type));
  //   cudaMalloc(&mat->scale_dev_d, d_scale_size.get_size()*type_call_dev<sizeofCUDT>(d_scale_type));
  // }
  if (a_props.scale_mode != scaling_type::None) {
    cudaMalloc(&mat->scale_dev_a, a_scale_size.get_size()*type_call_dev<sizeofCUDT>(a_scale_type));
  }
  if (b_props.scale_mode != scaling_type::None) {
    cudaMalloc(&mat->scale_dev_b, b_scale_size.get_size()*type_call_dev<sizeofCUDT>(b_scale_type));
  }
  if (c_props.scale_mode != scaling_type::None) {
    cudaMalloc(&mat->scale_dev_c, c_scale_size.get_size()*type_call_dev<sizeofCUDT>(c_scale_type));
  }
  if (d_props.scale_mode != scaling_type::None) {
    cudaMalloc(&mat->scale_dev_d, d_scale_size.get_size()*type_call_dev<sizeofCUDT>(d_scale_type));
  }
}

void cublaslt_gemm::fill_host() {
  for (int i = 0; i < flush_batch_count; i++){
    type_call_host<initHost>(a_type, a_props.init, ptr_host_a[i], rows_a, cols_a, lda,
                           batch_count, stride_a, control_a, constant_a, filename_a);
    type_call_host<initHost>(b_type, b_props.init, ptr_host_b[i], rows_b, cols_b, ldb,
                           batch_count, stride_b, control_b, constant_b, filename_b);
    type_call_host<initHost>(c_type, c_props.init, ptr_host_c[i], rows_c, cols_c, ldc,
                           batch_count, stride_c, control_c, constant_c, filename_c);
    // D is just output, don't need to init
  }
  if (a_props.scale_mode != scaling_type::None) {
    type_call_host<initHost>(a_scale_type, scale_init, scale_host_a, a_scale_size.rows, a_scale_size.cols, a_scale_size.rows, 1, 0LL, false, scale_factor_a, string(""));
  }
  if (b_props.scale_mode != scaling_type::None) {
    type_call_host<initHost>(b_scale_type, scale_init, scale_host_b, b_scale_size.rows, b_scale_size.cols, b_scale_size.rows, 1, 0LL, false, scale_factor_b, string(""));
  }
  if (c_props.scale_mode != scaling_type::None) {
    type_call_host<initHost>(c_scale_type, scale_init, scale_host_c, c_scale_size.rows, c_scale_size.cols, c_scale_size.rows, 1, 0LL, false, scale_factor_c, string(""));
  }
  if (d_props.scale_mode != scaling_type::None) {
    type_call_host<initHost>(d_scale_type, scale_init, scale_host_d, d_scale_size.rows, d_scale_size.cols, d_scale_size.rows, 1, 0LL, false, scale_factor_d, string(""));
  }
}

void cublaslt_gemm::copy_host_to_dev(cublaslt_gemm_inst *mat) {
  cudaSetDevice(mat->devIDX);
  for (int i = 0; i < flush_batch_count; i++) {
    copy_and_convert(a_type, ptr_host_a[i], mat->ptr_dev_a[i], rows_mem_a, cols_mem_a, batch_count);
    copy_and_convert(b_type, ptr_host_b[i], mat->ptr_dev_b[i], rows_mem_b, cols_mem_b, batch_count);
    copy_and_convert(c_type, ptr_host_c[i], mat->ptr_dev_c[i], rows_mem_c, cols_mem_c, batch_count);
  }

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
}

void cublaslt_gemm::prepare_matrix(cublaslt_gemm_inst *mat) {
  cublasOperation_t transACU = transA.convert_to_cuda();
  cublasOperation_t transBCU = transB.convert_to_cuda();
  check_cublas(cublasLtMatmulDescCreate(&mat->desc_op, compute, scalar));
  check_cublas(cublasLtMatmulDescSetAttribute(
      mat->desc_op, CUBLASLT_MATMUL_DESC_TRANSA, &transACU, sizeof(transACU)));
  check_cublas(cublasLtMatmulDescSetAttribute(
      mat->desc_op, CUBLASLT_MATMUL_DESC_TRANSB, &transBCU, sizeof(transBCU)));
  // set block scaling mode
  // set scaling factors
  if (a_props.scale_mode != scaling_type::None) {
    check_cublas(cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &a_scale_mode, sizeof(a_scale_mode)));
    check_cublas(cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &mat->scale_dev_a, sizeof(mat->scale_dev_a)));
  }
  if (b_props.scale_mode != scaling_type::None) {
    check_cublas(cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &b_scale_mode, sizeof(b_scale_mode)));
    check_cublas(cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &mat->scale_dev_b, sizeof(mat->scale_dev_b)));
  }
  if (c_props.scale_mode != scaling_type::None) {
    check_cublas(cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_C_SCALE_MODE, &c_scale_mode, sizeof(c_scale_mode)));
    check_cublas(cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &mat->scale_dev_c, sizeof(mat->scale_dev_c)));
  }
#if (ENABLE_CUDA_FP4)
  if (d_props.scale_mode != scaling_type::None) {
    check_cublas(cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &d_scale_mode, sizeof(d_scale_mode)));
    check_cublas(cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &mat->scale_dev_d, sizeof(mat->scale_dev_d)));
  }
#endif
  check_cublas(
      cublasLtMatrixLayoutCreate(&mat->desc_a, a_type, rows_a, cols_a, lda));
  check_cublas(
      cublasLtMatrixLayoutCreate(&mat->desc_b, b_type, rows_b, cols_b, ldb));
  check_cublas(
      cublasLtMatrixLayoutCreate(&mat->desc_c, c_type, rows_c, cols_c, ldc));
  if (!inplace) {
    check_cublas(
        cublasLtMatrixLayoutCreate(&mat->desc_d, d_type, rows_d, cols_d, ldd));
  } else {
    mat->desc_d = mat->desc_c;
  }
  if (batch_count > 1) {
    check_cublas(cublasLtMatrixLayoutSetAttribute(mat->desc_a, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    check_cublas(cublasLtMatrixLayoutSetAttribute(mat->desc_b, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    check_cublas(cublasLtMatrixLayoutSetAttribute(mat->desc_c, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    check_cublas(cublasLtMatrixLayoutSetAttribute(mat->desc_d, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

    check_cublas(cublasLtMatrixLayoutSetAttribute(mat->desc_a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
    check_cublas(cublasLtMatrixLayoutSetAttribute(mat->desc_b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
    check_cublas(cublasLtMatrixLayoutSetAttribute(mat->desc_c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));
    check_cublas(cublasLtMatrixLayoutSetAttribute(mat->desc_d, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_d, sizeof(stride_d)));
  }

  check_cublas(cublasLtMatmulPreferenceCreate(&mat->pref));
  check_cublas(cublasLtMatmulPreferenceSetAttribute(
      mat->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &mat->wSZ,
      sizeof(mat->wSZ)));
  if (a_type.is_fp8() || b_type.is_fp8() || c_type.is_fp8() || d_type.is_fp8()) {
    // Default is 0, enable for faster fp8 results
    int8_t fastAccuMode = 1;
    cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                   &fastAccuMode, sizeof(fastAccuMode));
  }
}

void cublaslt_gemm::no_tuning(cublaslt_gemm_inst *mat) {
  cublasStatus_t stat;
  cublasLtHandle_t handle;
  check_cuda(cudaSetDevice(mat->devIDX));
  check_cublas(cublasLtCreate(&handle));
  int retResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {0};

  check_cublas(cublasLtMatmulAlgoGetHeuristic(
      handle, mat->desc_op, mat->desc_a, mat->desc_b, mat->desc_c, mat->desc_d,
      mat->pref, 1, &heuristicResult, &retResults));

  if (retResults == 0) {
    check_cublas(CUBLAS_STATUS_NOT_SUPPORTED);
  }
  mat->algo = heuristicResult;
}
void cublaslt_gemm::auto_tuning(cublaslt_gemm_inst *mat) {
  // Not currently implemented, using simple method
  no_tuning(mat);
}

void cublaslt_gemm::free_mem() {
  free(alpha);
  free(beta);
  for (int i = 0; i < flush_batch_count; i++) {
        free(ptr_host_a[i]);
        free(ptr_host_b[i]);
        free(ptr_host_c[i]);
        free(ptr_host_d[i]);
  }
  free(ptr_host_a);
  free(ptr_host_b);
  free(ptr_host_c);
  free(ptr_host_d);
  if (use_scaling) {
    free(scale_host_a);
    free(scale_host_b);
    free(scale_host_c);
    free(scale_host_d);
  }
  for (auto mat : mat_ptrs) {
    for(int i = 0; i < flush_batch_count; i++) {
          cudaFree(mat.ptr_dev_a[i]);
          cudaFree(mat.ptr_dev_b[i]);
          cudaFree(mat.ptr_dev_c[i]);
          if (!inplace) {
            cudaFree(mat.ptr_dev_d[i]);
          }
    }
    cudaFree(mat.ptr_dev_a);
    cudaFree(mat.ptr_dev_b);
    cudaFree(mat.ptr_dev_c);
    if (!inplace) {
      cudaFree(mat.ptr_dev_d);
    }
    cudaFree(mat.devWork);
    if (use_scaling) {
      cudaFree(mat.scale_dev_a);
      cudaFree(mat.scale_dev_b);
      cudaFree(mat.scale_dev_c);
      cudaFree(mat.scale_dev_d);
    }
    cublasLtMatmulDescDestroy(mat.desc_op);
    cublasLtMatrixLayoutDestroy(mat.desc_a);
    cublasLtMatrixLayoutDestroy(mat.desc_b);
    cublasLtMatrixLayoutDestroy(mat.desc_c);
    if (!inplace) {
      cublasLtMatrixLayoutDestroy(mat.desc_d);
    }
    cublasLtMatmulPreferenceDestroy(mat.pref);
  }
}

double cublaslt_gemm::test() {
  vector<thread> threads;
  double gflops = 0.0;
  for (auto &mat : mat_ptrs) {
    threads.push_back(thread(&cublaslt_gemm::test_matmul, this, &mat));
  }
  // Wait on running jobs
  for (auto &thread : threads) {
    thread.join();
  }

  // Sum all gflops
  gflop_per_second = std::accumulate(
      begin(mat_ptrs), end(mat_ptrs), 0.0,
      [](double i, const cublaslt_gemm_inst &o) { return o.gflops + i; });

  gbyte_per_second = std::accumulate(
      begin(mat_ptrs), end(mat_ptrs), 0.0,
      [](double i, const cublaslt_gemm_inst &o) { return o.gbytes + i; });

  iter_time_us = std::accumulate(begin(mat_ptrs), end(mat_ptrs), 0.0,
                                 [](double i, const cublaslt_gemm_inst &o) {
                                   return o.time_us + i;
                                 }) /
                 mat_ptrs.size();

  return gflop_per_second;
}

std::string cublaslt_gemm::get_result_string() {
  std::ostringstream ossValues;
  ossValues << std::setprecision(7);
  ossValues << transA.to_string_short() << ',' << transB.to_string_short() << ',' << m
            << ',' << n << ',' << k << ',' << lda << ',' << ldb << ',' << ldc
            << ',' << ldd << ',';
  // if (batched) {
    ossValues << batch_count << ',';
  // }
  ossValues << *((float *)alpha) << ',';
  ossValues << *((float *)beta)   << ',';
  ossValues << a_type.to_string() << ',';
  ossValues << b_type.to_string() << ',';
  ossValues << c_type.to_string() << ',';
  ossValues << d_type.to_string() << ',';
  ossValues << compute.to_string() << ',';
  ossValues << scalar.to_string() << ',';
  ossValues << a_scale_type.to_string() << ',';
  ossValues << b_scale_type.to_string() << ',';
  ossValues << c_scale_type.to_string() << ',';
  ossValues << d_scale_type.to_string() << ',';
  ossValues << bias_type.to_string() << ',';
  ossValues << flush_memory_size << ','; // rotating buffer size
  ossValues << gflop_per_second << ',';
  ossValues << gbyte_per_second << ',';
  ossValues << iter_time_us << ',';
  if (cuda_monitor::monitor::enabled()) {
    ossValues << avg_sysclk_mhz << ',';
    ossValues << med_sysclk_mhz << ',';
    ossValues << avg_memclk_mhz << ',';
    ossValues << med_memclk_mhz << ',';
  }
  ossValues << endl;
  return ossValues.str();
}

std::tuple<double, double, double> cublaslt_gemm::calculate_figure_of_merit(
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

void cublaslt_gemm::test_matmul(cublaslt_gemm_inst *mat) {
  cublasStatus_t stat;
  cublasLtHandle_t handle;
  cudaStream_t stream;
  check_cuda(cudaSetDevice(mat->devIDX));
  check_cublas(cublasLtCreate(&handle));
  check_cuda(cudaStreamCreate(&stream));
  // Cold iters
  for (int rep = 0; rep < cold_iters; rep++) {
    int flush_index = rep % flush_batch_count;
    stat = cublasLtMatmul(handle, mat->desc_op, alpha, mat->ptr_dev_a[flush_index], mat->desc_a,
                          mat->ptr_dev_b[flush_index], mat->desc_b, beta, mat->ptr_dev_c[flush_index], mat->desc_c,
                          mat->ptr_dev_d[flush_index], mat->desc_d, &mat->algo.algo, mat->devWork,
                          mat->wSZ, stream);
    // Check for errors during the gemm run
    check_cublas(stat);
    check_cuda(cudaGetLastError());
  }
  check_cuda(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start));
  check_cuda(cudaEventCreate(&stop));

  /*
    Run and time the performance test
  */
  auto freq_monitor = cuda_monitor::monitor();
  freq_monitor.set_device_id(mat->devIDX);
  
  freq_monitor.start();
  cudaEventRecord(start, stream);
  for (int rep = 0; rep < iters; rep++) {
    int flush_index = rep % flush_batch_count;
    stat = cublasLtMatmul(handle, mat->desc_op, alpha, mat->ptr_dev_a[flush_index], mat->desc_a,
                          mat->ptr_dev_b[flush_index], mat->desc_b, beta, mat->ptr_dev_c[flush_index], mat->desc_c,
                          mat->ptr_dev_d[flush_index], mat->desc_d, &mat->algo.algo, mat->devWork,
                          mat->wSZ, stream);
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  freq_monitor.stop();

  // Check for errors during the performance test
  check_cublas(stat);
  check_cuda(cudaGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  cudaEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculate_figure_of_merit(static_cast<double>(elapsedTime_ms));

  if (cuda_monitor::monitor::enabled()) {
    avg_sysclk_mhz = freq_monitor.get_avg_sysclk_mhz();
    med_sysclk_mhz = freq_monitor.get_med_sysclk_mhz();
    avg_memclk_mhz = freq_monitor.get_avg_memclk_mhz();
    med_memclk_mhz = freq_monitor.get_med_memclk_mhz();
  }

}
