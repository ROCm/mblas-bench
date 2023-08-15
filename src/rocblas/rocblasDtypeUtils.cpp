#include "hipblasDtypeUtils.h"

#include <hip/hip_runtime.h>

#include <iostream>
#include <map>
#include <string>
using namespace std;

bool isReal(hipDataType type) {
  // You could also do this based on the string version with _R_ or _C_, but
  // those are hardcoded anyway
  switch (type) {
    case HIP_R_16F:
    case HIP_R_16BF:
    case HIP_R_32F:
    case HIP_R_64F:
    case HIP_R_8I:
    case HIP_R_8U:
    case HIP_R_32I:
    case HIP_R_32U:
      return true;
      break;

    // Complex numbers
    case HIP_C_16F:
    case HIP_C_16BF:
    case HIP_C_32F:
    case HIP_C_64F:
    case HIP_C_8I:
    case HIP_C_8U:
    case HIP_C_32I:
    case HIP_C_32U:
      return false;
      break;
    // Assume real I guess
    default:
      return true;
  }
}

bool isFp8(hipDataType precision) {
  // if (precision == CUDA_R_8F_E4M3 || precision == CUDA_R_8F_E5M2) {
  //   return true;
  // }
  return false;
}

std::string precToString(hipDataType precision) {
  for (auto ele : precDType) {
    if (ele.second == precision && ele.first.find("HIP") != string::npos) {
      return ele.first;
    }
  }
  return "";
}

std::string computeToString(hipblasComputeType_t compute) {
  for (auto ele : computeDType) {
    if (ele.second == compute && ele.first.find("HIPBLAS") != string::npos) {
      return ele.first;
    }
  }
  return "";
}

hipDataType precisionStringToDType(std::string stringPrecision) {
  try {
    return precDType.at(stringPrecision);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << stringPrecision << std::endl;
    throw e;
    return HIP_R_32F;
  }
}

hipblasComputeType_t selectCompute(std::string computestr,
                                   hipDataType precision) {
  if (computestr == "" || computestr == "PEDANTIC") {
    // If the user doesnt specify, just guess based on precision
    hipblasComputeType_t compute;
    try {
      compute = precToCompute.at(precision);
    } catch (std::out_of_range &e) {
      compute = HIPBLAS_COMPUTE_32F;
    }
    // if (computestr == "PEDANTIC") {
    //   // Borderline insane statement to enable the user to select pedantic
    //   // version without specifying compute type directly
    //   compute = static_cast<cublasComputeType_t>(static_cast<int>(compute) + 1);
    // }
    return compute;
  }
  try {
    return computeDType.at(computestr);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << computestr << std::endl;
    throw e;
    return HIPBLAS_COMPUTE_32F;
  }
}

hipDataType selectScalar(std::string scalarstr, hipDataType precision,
                         hipblasComputeType_t compute) {
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

hipblasOperation_t opStringToOp(std::string opstr) {
  if (opstr.empty()) {
    return HIPBLAS_OP_N;
  }
  try {
    return opType.at(opstr);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << opstr << std::endl;
    throw e;
  }
}
std::string opToString(HIPblasOperation_t op) {
  for (auto ele : opType) {
    if (ele.second == op) {
      return ele.first;
    }
  }
  return "N";
}