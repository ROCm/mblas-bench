#include "cublasLtGemm.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <future>
#include <iomanip>
#include <numeric>
#include <regex>
#include <string>
#include <thread>

#include "genericSetup.h"
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
std::vector<matmul_prec_type> cublasLtGemm::matmul_supported = {
  // Compute type                   Scale Type    A Type            B Type            C Type        D Type            Bias Type
  {mblasComputeType::MBLAS_COMPUTE_16F,              mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F},
  {mblasComputeType::MBLAS_COMPUTE_16F_PEDANTIC,     mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F},
  {mblasComputeType::MBLAS_COMPUTE_32I,              mblasDataType::MBLAS_R_32I,   mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_32I,   mblasDataType::MBLAS_R_32I,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32I_PEDANTIC,     mblasDataType::MBLAS_R_32I,   mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_32I,   mblasDataType::MBLAS_R_32I,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32I,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_8I,    mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32I_PEDANTIC,     mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_8I,    mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_16BF,      mblasDataType::MBLAS_R_16BF,      mblasDataType::MBLAS_R_16BF,  mblasDataType::MBLAS_R_16BF,      mblasDataType::MBLAS_R_16BF},
  {mblasComputeType::MBLAS_COMPUTE_32F_PEDANTIC,     mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_16BF,      mblasDataType::MBLAS_R_16BF,      mblasDataType::MBLAS_R_16BF,  mblasDataType::MBLAS_R_16BF,      mblasDataType::MBLAS_R_16BF},
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F},
  {mblasComputeType::MBLAS_COMPUTE_32F_PEDANTIC,     mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F},
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32F_PEDANTIC,     mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_16BF,      mblasDataType::MBLAS_R_16BF,      mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F},
  {mblasComputeType::MBLAS_COMPUTE_32F_PEDANTIC,     mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_16BF,      mblasDataType::MBLAS_R_16BF,      mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F},
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F},
  {mblasComputeType::MBLAS_COMPUTE_32F_PEDANTIC,     mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F},
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F},
  {mblasComputeType::MBLAS_COMPUTE_32F_PEDANTIC,     mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F},
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_8I,        mblasDataType::MBLAS_C_8I,        mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32F_PEDANTIC,     mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_8I,        mblasDataType::MBLAS_C_8I,        mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32F_PEDANTIC,     mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32F_FAST_16F,     mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F},
  {mblasComputeType::MBLAS_COMPUTE_32F_FAST_16BF,    mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F},
  {mblasComputeType::MBLAS_COMPUTE_32F_FAST_TF32,    mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_32F},
  {mblasComputeType::MBLAS_COMPUTE_32F_FAST_16F,     mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32F_FAST_16BF,    mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32F_FAST_TF32,    mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_64F,              mblasDataType::MBLAS_R_64F,   mblasDataType::MBLAS_R_64F,       mblasDataType::MBLAS_R_64F,       mblasDataType::MBLAS_R_64F,   mblasDataType::MBLAS_R_64F,       mblasDataType::MBLAS_R_64F},
  {mblasComputeType::MBLAS_COMPUTE_64F_PEDANTIC,     mblasDataType::MBLAS_R_64F,   mblasDataType::MBLAS_R_64F,       mblasDataType::MBLAS_R_64F,       mblasDataType::MBLAS_R_64F,   mblasDataType::MBLAS_R_64F,       mblasDataType::MBLAS_R_64F},
  {mblasComputeType::MBLAS_COMPUTE_64F,              mblasDataType::MBLAS_C_64F,   mblasDataType::MBLAS_C_64F,       mblasDataType::MBLAS_C_64F,       mblasDataType::MBLAS_C_64F,   mblasDataType::MBLAS_C_64F,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_64F_PEDANTIC,     mblasDataType::MBLAS_C_64F,   mblasDataType::MBLAS_C_64F,       mblasDataType::MBLAS_C_64F,       mblasDataType::MBLAS_C_64F,   mblasDataType::MBLAS_C_64F,       mblasDataType::MBLAS_ANY },
  // IMMA kernels
  // Compute type                   Scale Type    A Type            B Type            C Type        D Type            Bias Type
  {mblasComputeType::MBLAS_COMPUTE_32I,              mblasDataType::MBLAS_R_32I,   mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_32I,   mblasDataType::MBLAS_R_32I,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32I_PEDANTIC,     mblasDataType::MBLAS_R_32I,   mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_32I,   mblasDataType::MBLAS_R_32I,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32I,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_8I,    mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32I_PEDANTIC,     mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_R_8I,    mblasDataType::MBLAS_R_8I,        mblasDataType::MBLAS_ANY},
  // FP8 kernels
  // Compute type                   Scale Type    A Type            B Type            C Type        D Type            Bias Type
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16BF,  mblasDataType::MBLAS_R_16BF,      mblasDataType::MBLAS_R_16BF},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16BF,  mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16BF},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16F},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_16BF},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_16BF,  mblasDataType::MBLAS_R_16BF,      mblasDataType::MBLAS_R_16BF},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_16BF,  mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16BF},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_16BF,  mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_16BF},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16F},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_16F},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_16BF},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16BF,  mblasDataType::MBLAS_R_16BF,      mblasDataType::MBLAS_R_16BF},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16BF,  mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16BF},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16BF,  mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_16BF},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16F},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_16F},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_8F_E5M2,   mblasDataType::MBLAS_R_8F_E4M3,   mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_16BF},
#if (CUDART_VERSION >= 12080)
  // FP4 Kernels
  // Compute type                   Scale Type    A Type            B Type            C Type        D Type            Bias Type
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_4F_E2M1,   mblasDataType::MBLAS_R_4F_E2M1,   mblasDataType::MBLAS_R_16BF,   mblasDataType::MBLAS_R_4F_E2M1,       mblasDataType::MBLAS_R_16BF},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_4F_E2M1,   mblasDataType::MBLAS_R_4F_E2M1,   mblasDataType::MBLAS_R_16BF,   mblasDataType::MBLAS_R_16BF,       mblasDataType::MBLAS_R_16BF},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_4F_E2M1,   mblasDataType::MBLAS_R_4F_E2M1,   mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_4F_E2M1,       mblasDataType::MBLAS_R_16F},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_4F_E2M1,   mblasDataType::MBLAS_R_4F_E2M1,   mblasDataType::MBLAS_R_16F,   mblasDataType::MBLAS_R_16F,       mblasDataType::MBLAS_R_16F},
  {mblasComputeType::MBLAS_COMPUTE_32F,              mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_4F_E2M1,   mblasDataType::MBLAS_R_4F_E2M1,   mblasDataType::MBLAS_R_32F,   mblasDataType::MBLAS_R_32F,       mblasDataType::MBLAS_R_16BF},
#endif
  // Mixed precision complex kernels
  // Compute type                   Scale Type    A Type            B Type            C Type        D Type            Bias Type
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_16F,       mblasDataType::MBLAS_C_16F,       mblasDataType::MBLAS_C_16F,   mblasDataType::MBLAS_C_16F,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_16F,       mblasDataType::MBLAS_C_16F,       mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_16BF,      mblasDataType::MBLAS_C_16BF,      mblasDataType::MBLAS_C_16BF,  mblasDataType::MBLAS_C_16BF,      mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_16BF,      mblasDataType::MBLAS_C_16BF,      mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_ANY},
};
// clang-format on

void cublasLtGemm::parse_dev_iters(std::string deviceStr) {
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

void cublasLtGemm::parse_problem_type(string computeTStr, string scalarTStr,
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
    a_type = mblasCuDataType(aStr);
    b_type = mblasCuDataType(bStr);
    c_type = mblasCuDataType(cStr);
    d_type = mblasCuDataType(dStr);
  }

#if (CUDART_VERSION >= 12080)
  use_scaling = a_type.isFp4() || b_type.isFp4() || c_type.isFp4() || d_type.isFp4();
  if (use_scaling)
  {
    // Determine scale types (calculated from a,b,c,d type)
    a_scale_type = a_type.get_scale_type();
    b_scale_type = b_type.get_scale_type();
    c_scale_type = c_type.get_scale_type();
    d_scale_type = d_type.get_scale_type();
    // Scale modes
    a_scale_mode = a_type.get_scale_mode();
    b_scale_mode = b_type.get_scale_mode();
    c_scale_mode = c_type.get_scale_mode();
    d_scale_mode = d_type.get_scale_mode();
    // Calculate lengths
    a_scale_size = get_scale_tensor_size(m, k, a_scale_mode);
    b_scale_size = get_scale_tensor_size(k, n, b_scale_mode);
    c_scale_size = get_scale_tensor_size(m, n, c_scale_mode);
    d_scale_size = get_scale_tensor_size(m, n, d_scale_mode);

    if (a_type.isFp4()) {
      constant_a = 3.2;
    }
    if (b_type.isFp4()) {
      constant_b = 3.2;
    }
    if (c_type.isFp4()) {
      constant_c = 3.2;
    }
    if (d_type.isFp4()) {
      constant_d = 3.2;
    }
  }
#endif
}

void cublasLtGemm::validate_parameters() {
  // Validate that data types exist in table of supported configurations
  matmul_prec_type selType = {
      compute, scalar, a_type, b_type, c_type, d_type, mblasDataType::MBLAS_ANY};
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
  if ((a_type.isFp8() || b_type.isFp8() || c_type.isFp8() || d_type.isFp8()) &&
      (transA != mblasOperation::MBLAS_OP_T || transB != mblasOperation::MBLAS_OP_N)) {
    string errorString =
        "Transpose operation selection not supported"
        "\nOnly TN format is supported"
        "\nTransA: " +
        transA.to_string() + "\nTransB: " + transB.to_string();
    throw std::invalid_argument(errorString);
  }
}

cublasLtGemm::cublasLtGemm(cxxopts::ParseResult result) : genericGemm(result) {
  // Grab precision from command line
  precision = mblasCuDataType(result["precision"].as<string>());
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
  parse_problem_type(computeT, scalarT, aT, bT, cT, dT);

  parse_dev_iters(result["device"].as<string>());
  std::string tA = result["transposeA"].as<std::string>();
  std::string tB = result["transposeB"].as<std::string>();
  transA = mblasCuOperation(result["transposeA"].as<std::string>());
  transB = mblasCuOperation(result["transposeB"].as<std::string>());
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

string cublasLtGemm::prepare_array() {
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
  run_threaded(&cublasLtGemm::alloc_dev);
  run_threaded(&cublasLtGemm::copy_host_to_dev);
  run_threaded(&cublasLtGemm::prepare_matrix);
  // Enable tuning with a parameter later
  if (false) {
  } else {
    run_threaded(&cublasLtGemm::no_tuning);
  }
  std::ostringstream ossHeader;
  ossHeader << "transA_option,transB_option,M,N,K,lda,ldb,ldc,";
  if (batched) {
    ossHeader << "batch_count,";
  }
  ossHeader << "cuBLAS-Gflops,cuBLAS-GB/s,cuBLAS-us," << endl;
  return ossHeader.str();
}

void cublasLtGemm::run_threaded(void (cublasLtGemm::*func)(cublaslt_gemm_inst *)) {
  vector<thread> threads;
  for (auto &instance : mat_ptrs) {
    threads.push_back(thread(func, this, &instance));
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void cublasLtGemm::alloc_host() {
  // auto resultA = std::async(allocate_host_array, a_type, m, k, batch_count);
  // auto resultB = std::async(allocate_host_array, b_type, k, n, batch_count);
  // auto resultC = std::async(allocate_host_array, c_type, n, m, batch_count);
  // host_a = resultA.get();
  // host_b = resultB.get();
  // host_c = resultC.get();
  //host_a = allocate_host_array(a_type, rows_mem_a, cols_mem_a, batch_count);
  //host_b = allocate_host_array(b_type, rows_mem_b, cols_mem_b, batch_count);
  //host_c = allocate_host_array(c_type, rows_mem_c, cols_mem_c, batch_count);
  // Calculate size
  // total_block_size_host = calculate_offsets( 
  //   rows_mem_a, cols_mem_a, rows_mem_b, cols_mem_b, rows_mem_c, cols_mem_c, rows_mem_d, cols_mem_d,
  //   a_offset_host, b_offset_host, c_offset_host, d_offset_host,
  //   type_call_host<sizeofCUDT>(a_type),
  //   type_call_host<sizeofCUDT>(b_type),
  //   type_call_host<sizeofCUDT>(c_type),
  //   type_call_host<sizeofCUDT>(d_type),
  //   get_packing_count(a_type),
  //   get_packing_count(b_type),
  //   get_packing_count(c_type),
  //   get_packing_count(d_type),
  //   batch_count, inplace
  //  );
  // // Allocate big block of memory
  // dataHost = (void *)malloc(total_block_size_host * flush_batch_count);

  // Generate pointers 
  //if (batched && !strided) {
  // Perform some pointer arithmetic to calculate the arrays we pass to the
  // gpu
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

  //batched_pointer_magic_generic(ptr_host_a, dataHost + a_offset_host, batch_count, rows_mem_a, cols_mem_a, flush_batch_count, total_block_size_host, a_type);
  //batched_pointer_magic_generic(ptr_host_b, dataHost + b_offset_host, batch_count, rows_mem_b, cols_mem_b, flush_batch_count, total_block_size_host, b_type);
  //batched_pointer_magic_generic(ptr_host_c, dataHost + c_offset_host, batch_count, rows_mem_c, cols_mem_c, flush_batch_count, total_block_size_host, c_type);
  //batched_pointer_magic_generic(ptr_host_d, dataHost + d_offset_host, batch_count, rows_mem_d, cols_mem_d, flush_batch_count, total_block_size_host, d_type);
  if (use_scaling) {
    scale_host_a = malloc(a_scale_size.get_size()*type_call_host<sizeofCUDT>(a_scale_type));
    scale_host_b = malloc(b_scale_size.get_size()*type_call_host<sizeofCUDT>(b_scale_type));
    scale_host_c = malloc(c_scale_size.get_size()*type_call_host<sizeofCUDT>(c_scale_type));
    scale_host_d = malloc(d_scale_size.get_size()*type_call_host<sizeofCUDT>(d_scale_type));
  }

}

void cublasLtGemm::alloc_dev(cublaslt_gemm_inst *mat) {
  cudaSetDevice(mat->devIDX);

  //// Will update offset vars
  //total_block_size_dev = calculate_offsets( 
  //  rows_mem_a, cols_mem_a, rows_mem_b, cols_mem_b, rows_mem_c, cols_mem_c, rows_mem_d, cols_mem_d,
  //  a_offset_dev, b_offset_dev, c_offset_dev, d_offset_dev,
  //  type_call_dev<sizeofCUDT>(a_type),
  //  type_call_dev<sizeofCUDT>(b_type),
  //  type_call_dev<sizeofCUDT>(c_type),
  //  type_call_dev<sizeofCUDT>(d_type),
  //  get_packing_count(a_type),
  //  get_packing_count(b_type),
  //  get_packing_count(c_type),
  //  get_packing_count(d_type),
  //  batch_count, inplace
  // );

  //// Allocate big block of memory
  //check_cuda(cudaMalloc(&mat->dataDev, total_block_size_dev * flush_batch_count));
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
    mat->ptr_dev_d[i] = allocate_dev_array(d_type, rows_mem_d, cols_mem_d, batch_count);
  }

  //batched_pointer_magic_generic(mat->ptr_dev_a, mat->dataDev + a_offset_dev, batch_count, rows_mem_a, cols_mem_a, flush_batch_count, total_block_size_dev, a_type);
  //batched_pointer_magic_generic(mat->ptr_dev_b, mat->dataDev + b_offset_dev, batch_count, rows_mem_b, cols_mem_b, flush_batch_count, total_block_size_dev, b_type);
  //batched_pointer_magic_generic(mat->ptr_dev_c, mat->dataDev + c_offset_dev, batch_count, rows_mem_c, cols_mem_c, flush_batch_count, total_block_size_dev, c_type);
  //batched_pointer_magic_generic(mat->ptr_dev_d, mat->dataDev + d_offset_dev, batch_count, rows_mem_d, cols_mem_d, flush_batch_count, total_block_size_dev, d_type);
  // mat->devA = allocate_dev_array(a_type, rows_mem_a, cols_mem_a, batch_count);
  // mat->devB = allocate_dev_array(b_type, rows_mem_b, cols_mem_b, batch_count);
  // mat->devC = allocate_dev_array(c_type, rows_mem_c, cols_mem_c, batch_count);
  // if (!inplace) {
  //   mat->devD = allocate_dev_array(d_type, rows_mem_d, cols_mem_d, batch_count);
  // } else {
  //   mat->devD = mat->devC;
  // }
  mat->wSZ = workspace_size;
  cudaMalloc(&mat->devWork, mat->wSZ);
  if (use_scaling) {
    cudaMalloc(&mat->scale_dev_a, a_scale_size.get_size()*type_call_dev<sizeofCUDT>(a_scale_type));
    cudaMalloc(&mat->scale_dev_b, b_scale_size.get_size()*type_call_dev<sizeofCUDT>(b_scale_type));
    cudaMalloc(&mat->scale_dev_c, c_scale_size.get_size()*type_call_dev<sizeofCUDT>(c_scale_type));
    cudaMalloc(&mat->scale_dev_d, d_scale_size.get_size()*type_call_dev<sizeofCUDT>(d_scale_type));
  }
}

void cublasLtGemm::fill_host() {
  // Some random functions treat the matrix as a vectors, some require a matrix
  // vector<thread> threads;
  // threads.push_back(thread(initHostH, a_type, initialization, host_a, m, k,
  // lda,
  //                         batch_count, stride_a, 1.f, false));
  // threads.push_back(thread(initHostH, b_type, initialization, host_b, k, n,
  // ldb,
  //                         batch_count, stride_b, 2.f, true));
  // threads.push_back(thread(initHostH, c_type, initialization, host_c, m, n,
  // ldc,
  //                         batch_count, stride_c, 0.f, false));
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
    // D is just output, don't need to init
  }
  if (use_scaling) {
    type_call_host<initHost>(a_scale_type, string("constant"), scale_host_a, a_scale_size.rows, a_scale_size.cols, 1, 1, 0LL, false, scale_factor_a, string(""));
    type_call_host<initHost>(b_scale_type, string("constant"), scale_host_b, b_scale_size.rows, b_scale_size.cols, 1, 1, 0LL, false, scale_factor_b, string(""));
    type_call_host<initHost>(c_scale_type, string("constant"), scale_host_c, c_scale_size.rows, c_scale_size.cols, 1, 1, 0LL, false, scale_factor_c, string(""));
    type_call_host<initHost>(d_scale_type, string("constant"), scale_host_d, d_scale_size.rows, d_scale_size.cols, 1, 1, 0LL, false, scale_factor_d, string(""));
  }
}

void cublasLtGemm::copy_host_to_dev(cublaslt_gemm_inst *mat) {
  cudaSetDevice(mat->devIDX);
  for (int i = 0; i < flush_batch_count; i++) {
    copy_and_convert(a_type, ptr_host_a[i], mat->ptr_dev_a[i], rows_mem_a, cols_mem_a, batch_count);
    copy_and_convert(b_type, ptr_host_b[i], mat->ptr_dev_b[i], rows_mem_b, cols_mem_b, batch_count);
    copy_and_convert(c_type, ptr_host_c[i], mat->ptr_dev_c[i], rows_mem_c, cols_mem_c, batch_count);
  }

  ////if (batched && !strided) {
  //// Perform some pointer arithmetic to calculate the arrays we pass to the
  //// gpu
  //type_call_dev<batchedPtrCopy>(a_type, mat->ptr_dev_a, data + a_offset_host,
  //                            batch_count, rows_mem_a, cols_mem_a, flush_batch_count, total_block_size);
  //type_call_dev<batchedPtrCopy>(b_type, mat->ptr_dev_b, data + b_offset_host,
  //                            batch_count, rows_mem_b, cols_mem_b, flush_batch_count, total_block_size);
  //type_call_dev<batchedPtrCopy>(c_type, mat->ptr_dev_c, data + c_offset_host,
  //                            batch_count, rows_mem_c, cols_mem_c, flush_batch_count, total_block_size);
  //type_call_dev<batchedPtrCopy>(d_type, mat->ptr_dev_d, data + d_offset_host,
  //                            batch_count, rows_mem_d, cols_mem_d, flush_batch_count, total_block_size);
  //}
  if (use_scaling) {
    copy_and_convert(a_scale_type, scale_host_a, mat->scale_dev_a, a_scale_size.rows, a_scale_size.cols, 1);
    copy_and_convert(b_scale_type, scale_host_b, mat->scale_dev_b, b_scale_size.rows, b_scale_size.cols, 1);
    copy_and_convert(c_scale_type, scale_host_c, mat->scale_dev_c, c_scale_size.rows, c_scale_size.cols, 1);
    copy_and_convert(d_scale_type, scale_host_d, mat->scale_dev_d, d_scale_size.rows, d_scale_size.cols, 1);
  }
}

void cublasLtGemm::prepare_matrix(cublaslt_gemm_inst *mat) {
  cublasOperation_t transACU = transA.convert_to_cuda();
  cublasOperation_t transBCU = transB.convert_to_cuda();
  check_cublas(cublasLtMatmulDescCreate(&mat->desc_op, compute, scalar));
  check_cublas(cublasLtMatmulDescSetAttribute(
      mat->desc_op, CUBLASLT_MATMUL_DESC_TRANSA, &transACU, sizeof(transACU)));
  check_cublas(cublasLtMatmulDescSetAttribute(
      mat->desc_op, CUBLASLT_MATMUL_DESC_TRANSB, &transBCU, sizeof(transBCU)));
#if (CUDART_VERSION >= 12080)
  if (use_scaling) {
    // set block scaling mode
    check_cublas(cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &a_scale_mode, sizeof(a_scale_mode)));
    check_cublas(cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &b_scale_mode, sizeof(b_scale_mode)));
    //cublasLtMatmulMatrixScale_t d_scale_mode = CUBLASLT_MATMUL
    //check_cublas(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &DOutScaleMode, sizeof(DOutScaleMode)));

    // set scaling factors
    check_cublas(cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &mat->scale_dev_a, sizeof(mat->scale_dev_a)));
    check_cublas(cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &mat->scale_dev_b, sizeof(mat->scale_dev_b)));

    if (d_type.isFp4()) {
      check_cublas(cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &d_scale_mode, sizeof(d_scale_mode)));
      check_cublas(cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &mat->scale_dev_d, sizeof(mat->scale_dev_d)));
    }
    //check_cublas(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &d_out_scale, sizeof(d_out_scale)));
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
        cublasLtMatrixLayoutCreate(&mat->desc_d, d_type, rows_d, cold_d, ldd));
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
  if (a_type.isFp8() || b_type.isFp8() || c_type.isFp8() || d_type.isFp8()) {
    // Default is 0, enable for faster fp8 results
    int8_t fastAccuMode = 1;
    cublasLtMatmulDescSetAttribute(mat->desc_op, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                   &fastAccuMode, sizeof(fastAccuMode));
  }
}

void cublasLtGemm::no_tuning(cublaslt_gemm_inst *mat) {
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
void cublasLtGemm::auto_tuning(cublaslt_gemm_inst *mat) {
  // Not currently implemented, using simple method
  no_tuning(mat);
}

void cublasLtGemm::free_mem() {
  free(alpha);
  free(beta);
  free(ptr_host_a);
  free(ptr_host_b);
  free(ptr_host_c);
  free(ptr_host_d);
  free(dataHost);
  for (auto mat : mat_ptrs) {
    cudaFree(mat.dataDev);
    cudaFree(mat.ptr_dev_a);
    cudaFree(mat.ptr_dev_b);
    cudaFree(mat.ptr_dev_c);
    cudaFree(mat.ptr_dev_d);
  }
}

double cublasLtGemm::test() {
  vector<thread> threads;
  double gflops = 0.0;
  for (auto &mat : mat_ptrs) {
    threads.push_back(thread(&cublasLtGemm::test_matmul, this, &mat));
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

std::string cublasLtGemm::get_result_string() {
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

std::tuple<double, double, double> cublasLtGemm::calculate_figure_of_merit(
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

void cublasLtGemm::test_matmul(cublaslt_gemm_inst *mat) {
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

  // Check for errors during the performance test
  check_cublas(stat);
  check_cuda(cudaGetLastError());

  // Calculate and report GFlops
  float elapsedTime_ms;
  cudaEventElapsedTime(&elapsedTime_ms, start, stop);
  std::tie(mat->gflops, mat->gbytes, mat->time_us) =
      calculate_figure_of_merit(static_cast<double>(elapsedTime_ms));
}
