#include "mblasComputeType.h"
#include "mblasDataType.h"
#include <string>
#include <iostream>

const std::map<std::string, mblasComputeTypeEnum> mblasComputeType::computeDType = {
    // Generic, similar to rocblas format
    {"f32_r", MBLAS_COMPUTE_32F},
    {"f64_r", MBLAS_COMPUTE_64F},
    {"i32_r", MBLAS_COMPUTE_32I},
    {"MBLAS_COMPUTE_16F", MBLAS_COMPUTE_16F},
    {"MBLAS_COMPUTE_16F_PEDANTIC", MBLAS_COMPUTE_16F_PEDANTIC},
    {"MBLAS_COMPUTE_32F", MBLAS_COMPUTE_32F},
    {"MBLAS_COMPUTE_32F_PEDANTIC", MBLAS_COMPUTE_32F_PEDANTIC},
    {"MBLAS_COMPUTE_32F_FAST_16F", MBLAS_COMPUTE_32F_FAST_16F},
    {"MBLAS_COMPUTE_32F_FAST_16BF", MBLAS_COMPUTE_32F_FAST_16BF},
    {"MBLAS_COMPUTE_32F_FAST_TF32", MBLAS_COMPUTE_32F_FAST_TF32},
    {"MBLAS_COMPUTE_64F", MBLAS_COMPUTE_64F},
    {"MBLAS_COMPUTE_64F_PEDANTIC", MBLAS_COMPUTE_64F_PEDANTIC},
    {"MBLAS_COMPUTE_32I", MBLAS_COMPUTE_32I},
    {"MBLAS_COMPUTE_32I_PEDANTIC", MBLAS_COMPUTE_32I_PEDANTIC},
    {"CUBLAS_COMPUTE_16F", MBLAS_COMPUTE_16F},
    {"CUBLAS_COMPUTE_16F_PEDANTIC", MBLAS_COMPUTE_16F_PEDANTIC},
    {"CUBLAS_COMPUTE_32F", MBLAS_COMPUTE_32F},
    {"CUBLAS_COMPUTE_32F_PEDANTIC", MBLAS_COMPUTE_32F_PEDANTIC},
    {"CUBLAS_COMPUTE_32F_FAST_16F", MBLAS_COMPUTE_32F_FAST_16F},
    {"CUBLAS_COMPUTE_32F_FAST_16BF", MBLAS_COMPUTE_32F_FAST_16BF},
    {"CUBLAS_COMPUTE_32F_FAST_TF32", MBLAS_COMPUTE_32F_FAST_TF32},
    {"CUBLAS_COMPUTE_64F", MBLAS_COMPUTE_64F},
    {"CUBLAS_COMPUTE_64F_PEDANTIC", MBLAS_COMPUTE_64F_PEDANTIC},
    {"CUBLAS_COMPUTE_32I", MBLAS_COMPUTE_32I},
    {"CUBLAS_COMPUTE_32I_PEDANTIC", MBLAS_COMPUTE_32I_PEDANTIC},
    {"HIPBLAS_COMPUTE_16F", MBLAS_COMPUTE_16F},
    {"HIPBLAS_COMPUTE_16F_PEDANTIC", MBLAS_COMPUTE_16F_PEDANTIC},
    {"HIPBLAS_COMPUTE_32F", MBLAS_COMPUTE_32F},
    {"HIPBLAS_COMPUTE_32F_PEDANTIC", MBLAS_COMPUTE_32F_PEDANTIC},
    {"HIPBLAS_COMPUTE_32F_FAST_16F", MBLAS_COMPUTE_32F_FAST_16F},
    {"HIPBLAS_COMPUTE_32F_FAST_16BF", MBLAS_COMPUTE_32F_FAST_16BF},
    {"HIPBLAS_COMPUTE_32F_FAST_TF32", MBLAS_COMPUTE_32F_FAST_TF32},
    {"HIPBLAS_COMPUTE_64F", MBLAS_COMPUTE_64F},
    {"HIPBLAS_COMPUTE_64F_PEDANTIC", MBLAS_COMPUTE_64F_PEDANTIC},
    {"HIPBLAS_COMPUTE_32I", MBLAS_COMPUTE_32I},
    {"HIPBLAS_COMPUTE_32I_PEDANTIC", MBLAS_COMPUTE_32I_PEDANTIC},
    {"rocblas_compute_type_f32", MBLAS_COMPUTE_32F},
    {"rocblas_compute_type_f8_f8_f32", MBLAS_COMPUTE_32F_8F_8F},
    {"rocblas_compute_type_f8_bf8_f32", MBLAS_COMPUTE_32F_8F_8BF},
    {"rocblas_compute_type_bf8_f8_f32", MBLAS_COMPUTE_32F_8BF_8F},
    {"rocblas_compute_type_bf8_bf8_f32", MBLAS_COMPUTE_32F_8BF_8BF},
    {"rocblas_compute_type_invalid", MBLAS_COMPUTE_NULL}

};


// Manually defined
bool mblasComputeType::operator==(const mblasComputeType& other) const {
  return value == other.value;
}

bool mblasComputeType::operator<(const mblasComputeType& other) const {
  return value < other.value;
}

// Defined based on above
bool mblasComputeType::operator!=(const mblasComputeType& other) const {
  return !(*this == other);
}

bool mblasComputeType::operator>(const mblasComputeType& other) const {
  return (!(*this == other)) && (!(*this < other));
}

bool mblasComputeType::operator<=(const mblasComputeType& other) const {
  return (*this == other) || (*this < other);
}

bool mblasComputeType::operator>=(const mblasComputeType& other) const {
  return !(*this < other);
}

mblasComputeType::mblasComputeType(std::string instr) {
  if (computeDType.find(instr) != computeDType.end())
    value = computeDType.at(instr);
  else {
    value = mblasComputeType::MBLAS_COMPUTE_NULL;
  }
}

std::string mblasComputeType::toString(std::string prefix) const {
  for (auto ele : computeDType) {
    if (ele.second == value && ele.first.find(prefix) != std::string::npos) {
      return ele.first;
    }
  }
  // Try again
  for (auto ele : computeDType) {
    if (ele.second == value) {
      return ele.first;
    }
  }
  return "(Compute Type name not found)";
}

void mblasComputeType::setCompute(std::string computestr, mblasDataType & precision) {
  if (computestr != "") {
    // Attempt to parse the user's input with a map
    try {
      set(mblasComputeType(computestr));
    } catch (std::out_of_range &e) {
      std::cerr << "Failed to parse precision: " << computestr << std::endl;
      throw e;
      set(mblasComputeType::MBLAS_COMPUTE_32F);
    }
  } else {
    // If the user doesnt specify, just guess based on precision
    try {
      set(precToCompute.at(precision));
    } catch (std::out_of_range &e) {
      set(mblasComputeType::MBLAS_COMPUTE_32F);
    }
  }
}
//mblasComputeType::mblasComputeType() {
//  value = mblasComputeTypeEnum::MBLAS_NULL;
//}