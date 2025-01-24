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
std::vector<matmulPrecType> cublasLtGemm::matmulSupported = {
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
  // Mixed precision complex kernels
  // Compute type                   Scale Type    A Type            B Type            C Type        D Type            Bias Type
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_16F,       mblasDataType::MBLAS_C_16F,       mblasDataType::MBLAS_C_16F,   mblasDataType::MBLAS_C_16F,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_16F,       mblasDataType::MBLAS_C_16F,       mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_16BF,      mblasDataType::MBLAS_C_16BF,      mblasDataType::MBLAS_C_16BF,  mblasDataType::MBLAS_C_16BF,      mblasDataType::MBLAS_ANY},
  {mblasComputeType::MBLAS_COMPUTE_32F ,             mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_16BF,      mblasDataType::MBLAS_C_16BF,      mblasDataType::MBLAS_C_32F,   mblasDataType::MBLAS_C_32F,       mblasDataType::MBLAS_ANY},
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
  compute.setCompute(computeTStr, precision);
  scalar.setScalar(scalarTStr, precision, compute);
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
}

void cublasLtGemm::validateParameters() {
  // Validate that data types exist in table of supported configurations
  matmulPrecType selType = {
      compute, scalar, a_type, b_type, c_type, d_type, mblasDataType::MBLAS_ANY};
  auto result =
      std::find(begin(matmulSupported), end(matmulSupported), selType);
  if (result == end(matmulSupported)) {
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
  }
  // Validate that FP8 kernels will use TN format only
  // GEMM fails if not
  if ((a_type.isFp8() || b_type.isFp8() || c_type.isFp8() || d_type.isFp8()) &&
      (transA != mblasOperation::MBLAS_OP_T || transB != mblasOperation::MBLAS_OP_N)) {
    string errorString =
        "Transpose operation selection not supported"
        "\nOnly TN format is supported"
        "\nTransA: " +
        transA.toString() + "\nTransB: " + transB.toString();
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
  parseMType(computeT, scalarT, aT, bT, cT, dT);

  parseDevIters(result["device"].as<string>());
  std::string tA = result["transposeA"].as<std::string>();
  std::string tB = result["transposeB"].as<std::string>();
  transA = mblasCuOperation(result["transposeA"].as<std::string>());
  transB = mblasCuOperation(result["transposeB"].as<std::string>());
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
  //hostA = allocateHostArr(a_type, rowsMemA, colsMemA, batchct);
  //hostB = allocateHostArr(b_type, rowsMemB, colsMemB, batchct);
  //hostC = allocateHostArr(c_type, rowsMemC, colsMemC, batchct);
  // Calculate size
  total_block_size_host = calculate_offsets( 
    rowsMemA, colsMemA, rowsMemB, colsMemB, rowsMemC, colsMemC, rowsMemD, colsMemD,
    a_offset_host, b_offset_host, c_offset_host, d_offset_host,
    typeCallHost<sizeofCUDT>(a_type),
    typeCallHost<sizeofCUDT>(b_type),
    typeCallHost<sizeofCUDT>(c_type),
    typeCallHost<sizeofCUDT>(d_type),
    batchct, inplace
   );
  // Allocate big block of memory
  dataHost = (void *)malloc(total_block_size_host * flush_batch_count);

  // Generate pointers 
  //if (batched && !strided) {
  // Perform some pointer arithmetic to calculate the arrays we pass to the
  // gpu
  ptrHostA =
      (void **)malloc(batchct * flush_batch_count * typeCallHost<sizeofCUDTP>(a_type));
  ptrHostB =
      (void **)malloc(batchct * flush_batch_count * typeCallHost<sizeofCUDTP>(b_type));
  ptrHostC =
      (void **)malloc(batchct * flush_batch_count * typeCallHost<sizeofCUDTP>(c_type));
  ptrHostD =
      (void **)malloc(batchct * flush_batch_count * typeCallHost<sizeofCUDTP>(d_type));
  batchedPtrMagicGeneric(ptrHostA, dataHost + a_offset_host, batchct, rowsMemA, colsMemA, flush_batch_count, total_block_size_host, typeCallHost<sizeofCUDT>(a_type));
  batchedPtrMagicGeneric(ptrHostB, dataHost + b_offset_host, batchct, rowsMemB, colsMemB, flush_batch_count, total_block_size_host, typeCallHost<sizeofCUDT>(b_type));
  batchedPtrMagicGeneric(ptrHostC, dataHost + c_offset_host, batchct, rowsMemC, colsMemC, flush_batch_count, total_block_size_host, typeCallHost<sizeofCUDT>(c_type));
  batchedPtrMagicGeneric(ptrHostD, dataHost + d_offset_host, batchct, rowsMemD, colsMemD, flush_batch_count, total_block_size_host, typeCallHost<sizeofCUDT>(d_type));
}

void cublasLtGemm::allocDev(cublasltgemmInst *mat) {
  cudaSetDevice(mat->devIDX);

  // Will update offset vars
  total_block_size_dev = calculate_offsets( 
    rowsMemA, colsMemA, rowsMemB, colsMemB, rowsMemC, colsMemC, rowsMemD, colsMemD,
    a_offset_dev, b_offset_dev, c_offset_dev, d_offset_dev,
    typeCallDev<sizeofCUDT>(a_type),
    typeCallDev<sizeofCUDT>(b_type),
    typeCallDev<sizeofCUDT>(c_type),
    typeCallDev<sizeofCUDT>(d_type),
    batchct, inplace
   );

  // Allocate big block of memory
  checkCuda(cudaMalloc(&mat->dataDev, total_block_size_dev * flush_batch_count));
  mat->ptrDevA =
      (void **)malloc(batchct * flush_batch_count * typeCallDev<sizeofCUDTP>(a_type));
  mat->ptrDevB =
      (void **)malloc(batchct * flush_batch_count * typeCallDev<sizeofCUDTP>(b_type));
  mat->ptrDevC =
      (void **)malloc(batchct * flush_batch_count * typeCallDev<sizeofCUDTP>(c_type));
  mat->ptrDevD =
      (void **)malloc(batchct * flush_batch_count * typeCallDev<sizeofCUDTP>(d_type));

  batchedPtrMagicGeneric(mat->ptrDevA, mat->dataDev + a_offset_dev, batchct, rowsMemA, colsMemA, flush_batch_count, total_block_size_dev, typeCallDev<sizeofCUDT>(a_type));
  batchedPtrMagicGeneric(mat->ptrDevB, mat->dataDev + b_offset_dev, batchct, rowsMemB, colsMemB, flush_batch_count, total_block_size_dev, typeCallDev<sizeofCUDT>(b_type));
  batchedPtrMagicGeneric(mat->ptrDevC, mat->dataDev + c_offset_dev, batchct, rowsMemC, colsMemC, flush_batch_count, total_block_size_dev, typeCallDev<sizeofCUDT>(c_type));
  batchedPtrMagicGeneric(mat->ptrDevD, mat->dataDev + d_offset_dev, batchct, rowsMemD, colsMemD, flush_batch_count, total_block_size_dev, typeCallDev<sizeofCUDT>(d_type));
  // mat->devA = allocateDevArr(a_type, rowsMemA, colsMemA, batchct);
  // mat->devB = allocateDevArr(b_type, rowsMemB, colsMemB, batchct);
  // mat->devC = allocateDevArr(c_type, rowsMemC, colsMemC, batchct);
  // if (!inplace) {
  //   mat->devD = allocateDevArr(d_type, rowsMemD, colsMemD, batchct);
  // } else {
  //   mat->devD = mat->devC;
  // }
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
  for (int i = 0; i < flush_batch_count; i++){
    typeCallHost<initHost>(a_type, initialization, ptrHostA[i], rowsA, colsA, lda,
                           batchct, stride_a, controlA, constantA, filenameA);
    typeCallHost<initHost>(b_type, initialization, ptrHostB[i], rowsB, colsB, ldb,
                           batchct, stride_b, controlB, constantB, filenameB);
    typeCallHost<initHost>(c_type, initialization, ptrHostC[i], rowsC, colsC, ldc,
                           batchct, stride_c, controlC, constantC, filenameC);
    // D is just output, don't need to init
  }
}

void cublasLtGemm::copyHostToDev(cublasltgemmInst *mat) {
  cudaSetDevice(mat->devIDX);
  for (int j = 0; j < flush_batch_count; j++) {
    //for (int i = 0; i < batchct; i++) {
      //copyAndConvert(precision, ptrHostA[j*batchct + i], mat->ptrDevA[j*batchct + i], rowsMemA, colsMemA, batchct);
      //copyAndConvert(precision, ptrHostB[j*batchct + i], mat->ptrDevB[j*batchct + i], rowsMemB, colsMemB, batchct);
      //copyAndConvert(precision, ptrHostC[j*batchct + i], mat->ptrDevC[j*batchct + i], rowsMemC, colsMemC, batchct);
    copyAndConvert(precision, ptrHostA[j*batchct], mat->ptrDevA[j*batchct], rowsMemA, colsMemA, batchct);
    copyAndConvert(precision, ptrHostB[j*batchct], mat->ptrDevB[j*batchct], rowsMemB, colsMemB, batchct);
    copyAndConvert(precision, ptrHostC[j*batchct], mat->ptrDevC[j*batchct], rowsMemC, colsMemC, batchct);
  }

  ////if (batched && !strided) {
  //// Perform some pointer arithmetic to calculate the arrays we pass to the
  //// gpu
  //typeCallDev<batchedPtrCopy>(a_type, mat->ptrDevA, data + a_offset_host,
  //                            batchct, rowsMemA, colsMemA, flush_batch_count, total_block_size);
  //typeCallDev<batchedPtrCopy>(b_type, mat->ptrDevB, data + b_offset_host,
  //                            batchct, rowsMemB, colsMemB, flush_batch_count, total_block_size);
  //typeCallDev<batchedPtrCopy>(c_type, mat->ptrDevC, data + c_offset_host,
  //                            batchct, rowsMemC, colsMemC, flush_batch_count, total_block_size);
  //typeCallDev<batchedPtrCopy>(d_type, mat->ptrDevD, data + d_offset_host,
  //                            batchct, rowsMemD, colsMemD, flush_batch_count, total_block_size);
  //}
}

void cublasLtGemm::prepareMatrix(cublasltgemmInst *mat) {
  cublasOperation_t transACU = transA.convertToCuda();
  cublasOperation_t transBCU = transB.convertToCuda();
  checkCublas(cublasLtMatmulDescCreate(&mat->descOP, compute, scalar));
  checkCublas(cublasLtMatmulDescSetAttribute(
      mat->descOP, CUBLASLT_MATMUL_DESC_TRANSA, &transACU, sizeof(transACU)));
  checkCublas(cublasLtMatmulDescSetAttribute(
      mat->descOP, CUBLASLT_MATMUL_DESC_TRANSB, &transBCU, sizeof(transBCU)));

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
  if (a_type.isFp8() || b_type.isFp8() || c_type.isFp8() || d_type.isFp8()) {
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
  //free(alpha);
  //free(beta);
  //free(hostA);
  //free(hostB);
  //free(hostC);
  //for (auto mat : matPtrs) {
  //  cudaFree(mat.ptrDevA);
  //  cudaFree(mat.ptrDevB);
  //  cudaFree(mat.ptrDevC);
  //}
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
  ossValues << transA.toStringShort() << ',' << transB.toStringShort() << ',' << m
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
    int flush_index = rep % flush_batch_count;
    stat = cublasLtMatmul(handle, mat->descOP, alpha, mat->ptrDevA[flush_index * batchct], mat->descA,
                          mat->ptrDevB[flush_index * batchct], mat->descB, beta, mat->ptrDevC[flush_index * batchct], mat->descC,
                          mat->ptrDevD[flush_index * batchct], mat->descD, &mat->algo.algo, mat->devWork,
                          mat->wSZ, stream);
    // Check for errors during the gemm run
    checkCublas(stat);
    checkCuda(cudaGetLastError());
  }
  checkCuda(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));

  /*
    Run and time the performance test
  */
  cudaEventRecord(start, stream);
  for (int rep = 0; rep < iters; rep++) {
    int flush_index = rep % flush_batch_count;
    stat = cublasLtMatmul(handle, mat->descOP, alpha, mat->ptrDevA[flush_index * batchct], mat->descA,
                          mat->ptrDevB[flush_index * batchct], mat->descB, beta, mat->ptrDevC[flush_index * batchct], mat->descC,
                          mat->ptrDevD[flush_index * batchct], mat->descD, &mat->algo.algo, mat->devWork,
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
