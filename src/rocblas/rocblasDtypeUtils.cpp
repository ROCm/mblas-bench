#include "rocblasDtypeUtils.h"

#include <hip/hip_runtime.h>

#include <iostream>
#include <map>
#include <string>
using namespace std;

bool isReal(rocblas_datatype type) {
  // You could also do this based on the string version with _R_ or _C_, but
  // those are hardcoded anyway
  switch (type) {
    case rocblas_datatype_f16_r:
    case rocblas_datatype_bf16_r:
    case rocblas_datatype_f32_r:
    case rocblas_datatype_f64_r:
    case rocblas_datatype_i8_r:
    case rocblas_datatype_u8_r:
    case rocblas_datatype_i32_r:
    case rocblas_datatype_u32_r:
      return true;
      break;

    // Complex numbers
    case rocblas_datatype_f16_c:
    case rocblas_datatype_bf16_c:
    case rocblas_datatype_f32_c:
    case rocblas_datatype_f64_c:
    case rocblas_datatype_i8_c:
    case rocblas_datatype_u8_c:
    case rocblas_datatype_i32_c:
    case rocblas_datatype_u32_c:
      return false;
      break;
    // Assume real I guess
    default:
      return true;
  }
}

bool isFp8(rocblas_datatype precision) {
  // if (precision == CUDA_R_8F_E4M3 || precision == CUDA_R_8F_E5M2) {
  //   return true;
  // }
  return false;
}

std::string precToString(rocblas_datatype precision) {
  for (auto ele : precDType) {
    if (ele.second == precision && ele.first.find("rocblas_datatype") != string::npos) {
      return ele.first;
    }
  }
  return "";
}

std::string computeToString(rocblas_datatype compute) {
  for (auto ele : computeDType) {
    if (ele.second == compute && ele.first.find("rocblas_datatype") != string::npos) {
      return ele.first;
    }
  }
  return "";
}

rocblas_datatype precisionStringToDType(std::string stringPrecision) {
  try {
    return precDType.at(stringPrecision);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << stringPrecision << std::endl;
    throw e;
    return rocblas_datatype_f32_r;
  }
}

rocblas_datatype selectCompute(std::string computestr,
                               rocblas_datatype precision) {
  if (computestr == "") {
    // If the user doesnt specify, just guess based on precision
    rocblas_datatype compute;
    try {
      compute = precToCompute.at(precision);
    } catch (std::out_of_range &e) {
      compute = rocblas_datatype_f32_r;
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
    return rocblas_datatype_f32_r;
  }
}

rocblas_datatype selectScalar(std::string scalarstr, rocblas_datatype precision,
                              rocblas_datatype compute) {
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

rocblas_operation opStringToOp(std::string opstr) {
  if (opstr.empty()) {
    return rocblas_operation_none;
  }
  try {
    return opType.at(opstr);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse precision: " << opstr << std::endl;
    throw e;
  }
}
std::string opToString(rocblas_operation op) {
  for (auto ele : opType) {
    if (ele.second == op) {
      return ele.first;
    }
  }
  return "N";
}