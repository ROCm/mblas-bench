#include "cublasDtypeUtils.h"

#include <cuda_runtime.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>
using namespace std;

bool isReal(cudaDataType_t type) {
  // You could also do this based on the string version with _R_ or _C_, but
  // those are hardcoded anyway
  switch (type) {
    case CUDA_R_16F:
    case CUDA_R_16BF:
    case CUDA_R_32F:
    case CUDA_R_64F:
    case CUDA_R_4I:
    case CUDA_R_4U:
    case CUDA_R_8I:
    case CUDA_R_8U:
    case CUDA_R_16I:
    case CUDA_R_16U:
    case CUDA_R_32I:
    case CUDA_R_32U:
    case CUDA_R_64I:
    case CUDA_R_64U:
    case CUDA_R_8F_E4M3:
    case CUDA_R_8F_E5M2:
      return true;
      break;

    // Complex numbers
    case CUDA_C_16F:
    case CUDA_C_16BF:
    case CUDA_C_32F:
    case CUDA_C_64F:
    case CUDA_C_4I:
    case CUDA_C_4U:
    case CUDA_C_8I:
    case CUDA_C_8U:
    case CUDA_C_16I:
    case CUDA_C_16U:
    case CUDA_C_32I:
    case CUDA_C_32U:
    case CUDA_C_64I:
    case CUDA_C_64U:
      return false;
      break;
    // Assume real I guess
    default:
      return true;
  }
}

bool isFp8(cudaDataType precision) {
  if (precision == CUDA_R_8F_E4M3 || precision == CUDA_R_8F_E5M2) {
    return true;
  }
  return false;
}

std::string precToString(cudaDataType precision) {
  for (auto ele : precDType) {
    if (ele.second == precision && ele.first.find("CUDA") != string::npos) {
      return ele.first;
    }
  }
  return "";
}

std::string computeToString(cublasComputeType_t compute) {
  for (auto ele : computeDType) {
    if (ele.second == compute && ele.first.find("CUBLAS") != string::npos) {
      return ele.first;
    }
  }
  return "";
}

cudaDataType_t precisionStringToDType(std::string stringPrecision) {
  try {
    return precDType.at(stringPrecision);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << stringPrecision << std::endl;
    throw e;
    return CUDA_R_32F;
  }
}

cublasComputeType_t selectCompute(std::string computestr,
                                  cudaDataType_t precision) {
  if (computestr == "" || computestr == "PEDANTIC") {
    // If the user doesnt specify, just guess based on precision
    cublasComputeType_t compute;
    try {
      compute = precToCompute.at(precision);
    } catch (std::out_of_range &e) {
      compute = CUBLAS_COMPUTE_32F;
    }
    if (computestr == "PEDANTIC") {
      // Borderline insane statement to enable the user to select pedantic
      // version without specifying compute type directly
      compute = static_cast<cublasComputeType_t>(static_cast<int>(compute) + 1);
    }
    return compute;
  }
  try {
    return computeDType.at(computestr);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << computestr << std::endl;
    throw e;
    return CUBLAS_COMPUTE_32F;
  }
}

cudaDataType_t selectScalar(std::string scalarstr, cudaDataType_t precision,
                            cublasComputeType_t compute) {
  if (scalarstr == "") {
    // Scalar type not specified, setting based on compute type
    for (auto ele : precToCompute) {
      if (ele.second == compute && isReal(precision) == isReal(ele.first)) {
        return ele.first;
      }
    }
    // something terrible has happened
    return precision;
  }
  return precisionStringToDType(scalarstr);
}

cublasOperation_t opStringToOp(std::string opstr) {
  if (opstr.empty()) {
    return CUBLAS_OP_N;
  }
  try {
    return opType.at(opstr);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << opstr << std::endl;
    throw e;
  }
}
std::string opToString(cublasOperation_t op) {
  for (auto ele : opType) {
    if (ele.second == op) {
      return ele.first;
    }
  }
  return "N";
}


bool matchGemmType(cudaDataType_t precision, std::string function, cudaDataType_t desiredPrec, std::vector<string> acceptable) {
  if (precision != desiredPrec) {
    return false;
  }
  for (auto afunc : acceptable) {
    if (function == afunc) {
      return true;
    }
  }
  return false;
}