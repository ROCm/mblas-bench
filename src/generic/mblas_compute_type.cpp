#include "mblas_compute_type.h"
#include "mblas_data_type.h"
#include <string>
#include <iostream>

const std::map<std::string, mblas_compute_type_enum> mblas_compute_type::computeDType = {
    // Generic, similar to rocblas format
    {"f32_r", MBLAS_COMPUTE_32F},
    {"c_f32_r", MBLAS_COMPUTE_32F},
    {"f64_r", MBLAS_COMPUTE_64F},
    {"c_f64_r", MBLAS_COMPUTE_64F},
    {"i32_r", MBLAS_COMPUTE_32I},
    {"c_i32_r", MBLAS_COMPUTE_32I},
    {"f16_r", MBLAS_COMPUTE_16F},
    {"c_f16_r", MBLAS_COMPUTE_16F},
    {"xf32_r", MBLAS_COMPUTE_32F_FAST_TF32},
    {"c_xf32_r", MBLAS_COMPUTE_32F_FAST_TF32},
    {"tf32_r", MBLAS_COMPUTE_32F_FAST_TF32},
    {"c_tf32_r", MBLAS_COMPUTE_32F_FAST_TF32},
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
    {"CUBLAS_COMPUTE_32F_EMULATED_16BFX9", MBLAS_COMPUTE_32F_EMULATED_16BFX9},
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
bool mblas_compute_type::operator==(const mblas_compute_type& other) const {
  return value == other.value;
}

bool mblas_compute_type::operator<(const mblas_compute_type& other) const {
  return value < other.value;
}

// Defined based on above
bool mblas_compute_type::operator!=(const mblas_compute_type& other) const {
  return !(*this == other);
}

bool mblas_compute_type::operator>(const mblas_compute_type& other) const {
  return (!(*this == other)) && (!(*this < other));
}

bool mblas_compute_type::operator<=(const mblas_compute_type& other) const {
  return (*this == other) || (*this < other);
}

bool mblas_compute_type::operator>=(const mblas_compute_type& other) const {
  return !(*this < other);
}

mblas_compute_type::mblas_compute_type(std::string instr) {
  if (computeDType.find(instr) != computeDType.end())
    value = computeDType.at(instr);
  else {
    value = mblas_compute_type::MBLAS_COMPUTE_NULL;
  }
}

std::string mblas_compute_type::to_string(std::string prefix) const {
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

void mblas_compute_type::set_compute(std::string computestr, mblas_data_type & precision) {
  if (computestr != "") {
    // Attempt to parse the user's input with a map
    try {
      set(mblas_compute_type(computestr));
    } catch (std::out_of_range &e) {
      std::cerr << "Failed to parse precision: " << computestr << std::endl;
      throw e;
      set(mblas_compute_type::MBLAS_COMPUTE_32F);
    }
  } else {
    // If the user doesnt specify, just guess based on precision
    try {
      set(precToCompute.at(precision));
    } catch (std::out_of_range &e) {
      set(mblas_compute_type::MBLAS_COMPUTE_32F);
    }
  }
}
//mblas_compute_type::mblas_compute_type() {
//  value = mblas_compute_type_enum::MBLAS_NULL;
//}
