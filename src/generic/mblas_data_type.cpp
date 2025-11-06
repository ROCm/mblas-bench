#include "mblas_data_type.h"
#include "mblas_compute_type.h"
#include <string>
#include <iostream>

const std::map<std::string, mblas_data_type_enum> mblas_data_type::precDType = {
    // Generic, similar to rocblas format
    {"h", mblas_data_type_enum::MBLAS_R_16F},       {"s", mblas_data_type_enum::MBLAS_R_32F},     {"d", mblas_data_type_enum::MBLAS_R_64F},
    {"c", mblas_data_type_enum::MBLAS_C_32F},       {"z", mblas_data_type_enum::MBLAS_C_64F},     {"f16_r", mblas_data_type_enum::MBLAS_R_16F},
    {"f16_c", mblas_data_type_enum::MBLAS_C_16F},   {"f32_r", mblas_data_type_enum::MBLAS_R_32F}, {"f32_c", mblas_data_type_enum::MBLAS_C_32F},
    {"f64_r", mblas_data_type_enum::MBLAS_R_64F},   {"f64_c", mblas_data_type_enum::MBLAS_C_64F}, {"bf16_r", mblas_data_type_enum::MBLAS_R_16BF},
    {"bf16_c", mblas_data_type_enum::MBLAS_C_16BF}, {"i8_r", mblas_data_type_enum::MBLAS_R_8I},   {"i8_c", mblas_data_type_enum::MBLAS_C_8I},
    {"i32_r", mblas_data_type_enum::MBLAS_R_32I},   {"i32_c", mblas_data_type_enum::MBLAS_C_32I},
    {"f8_r", mblas_data_type_enum::MBLAS_R_8F_E4M3},   {"bf8_r", mblas_data_type_enum::MBLAS_R_8F_E5M2},
    {"f6_r", mblas_data_type_enum::MBLAS_R_6F_E3M2},
    {"f4_r", mblas_data_type_enum::MBLAS_R_4F_E2M1},
    // MBLAS
    {"MBLAS_ANY",  mblas_data_type_enum::MBLAS_ANY},
    {"MBLAS_NULL",  mblas_data_type_enum::MBLAS_NULL},
    {"MBLAS_R_16F",  mblas_data_type_enum::MBLAS_R_16F},
    {"MBLAS_C_16F",  mblas_data_type_enum::MBLAS_C_16F},
    {"MBLAS_R_16BF", mblas_data_type_enum::MBLAS_R_16BF},
    {"MBLAS_C_16BF", mblas_data_type_enum::MBLAS_C_16BF},
    {"MBLAS_R_32F",  mblas_data_type_enum::MBLAS_R_32F},
    {"MBLAS_C_32F",  mblas_data_type_enum::MBLAS_C_32F},
    {"MBLAS_R_64F",  mblas_data_type_enum::MBLAS_R_64F},
    {"MBLAS_C_64F",  mblas_data_type_enum::MBLAS_C_64F},
    {"MBLAS_R_4I",   mblas_data_type_enum::MBLAS_R_4I},
    {"MBLAS_C_4I",   mblas_data_type_enum::MBLAS_C_4I},
    {"MBLAS_R_4U",   mblas_data_type_enum::MBLAS_R_4U},
    {"MBLAS_C_4U",   mblas_data_type_enum::MBLAS_C_4U},
    {"MBLAS_R_8I",   mblas_data_type_enum::MBLAS_R_8I},
    {"MBLAS_C_8I",   mblas_data_type_enum::MBLAS_C_8I},
    {"MBLAS_R_8U",   mblas_data_type_enum::MBLAS_R_8U},
    {"MBLAS_C_8U",   mblas_data_type_enum::MBLAS_C_8U},
    {"MBLAS_R_16I",  mblas_data_type_enum::MBLAS_R_16I},
    {"MBLAS_C_16I",  mblas_data_type_enum::MBLAS_C_16I},
    {"MBLAS_R_16U",  mblas_data_type_enum::MBLAS_R_16U},
    {"MBLAS_C_16U",  mblas_data_type_enum::MBLAS_C_16U},
    {"MBLAS_R_32I",  mblas_data_type_enum::MBLAS_R_32I},
    {"MBLAS_C_32I",  mblas_data_type_enum::MBLAS_C_32I},
    {"MBLAS_R_32U",  mblas_data_type_enum::MBLAS_R_32U},
    {"MBLAS_C_32U",  mblas_data_type_enum::MBLAS_C_32U},
    {"MBLAS_R_64I",  mblas_data_type_enum::MBLAS_R_64I},
    {"MBLAS_C_64I",  mblas_data_type_enum::MBLAS_C_64I},
    {"MBLAS_R_64U",  mblas_data_type_enum::MBLAS_R_64U},
    {"MBLAS_C_64U",  mblas_data_type_enum::MBLAS_C_64U},
    {"MBLAS_R_8F_E4M3", mblas_data_type_enum::MBLAS_R_8F_E4M3},
    {"MBLAS_R_8F_E5M2", mblas_data_type_enum::MBLAS_R_8F_E5M2},
    {"MBLAS_R_8F_UE4M3", mblas_data_type_enum::MBLAS_R_8F_UE4M3},
    {"MBLAS_R_8F_UE8M0", mblas_data_type_enum::MBLAS_R_8F_UE8M0},
    {"MBLAS_R_6F_E2M3", mblas_data_type_enum::MBLAS_R_6F_E2M3},
    {"MBLAS_R_6F_E3M2", mblas_data_type_enum::MBLAS_R_6F_E3M2},
    {"MBLAS_R_4F_E2M1", mblas_data_type_enum::MBLAS_R_4F_E2M1},
    // CUDA
    {"CUDA_R_16F",  mblas_data_type_enum::MBLAS_R_16F},
    {"CUDA_C_16F",  mblas_data_type_enum::MBLAS_C_16F},
    {"CUDA_R_16BF", mblas_data_type_enum::MBLAS_R_16BF},
    {"CUDA_C_16BF", mblas_data_type_enum::MBLAS_C_16BF},
    {"CUDA_R_32F",  mblas_data_type_enum::MBLAS_R_32F},
    {"CUDA_C_32F",  mblas_data_type_enum::MBLAS_C_32F},
    {"CUDA_R_64F",  mblas_data_type_enum::MBLAS_R_64F},
    {"CUDA_C_64F",  mblas_data_type_enum::MBLAS_C_64F},
    {"CUDA_R_4I",   mblas_data_type_enum::MBLAS_R_4I},
    {"CUDA_C_4I",   mblas_data_type_enum::MBLAS_C_4I},
    {"CUDA_R_4U",   mblas_data_type_enum::MBLAS_R_4U},
    {"CUDA_C_4U",   mblas_data_type_enum::MBLAS_C_4U},
    {"CUDA_R_8I",   mblas_data_type_enum::MBLAS_R_8I},
    {"CUDA_C_8I",   mblas_data_type_enum::MBLAS_C_8I},
    {"CUDA_R_8U",   mblas_data_type_enum::MBLAS_R_8U},
    {"CUDA_C_8U",   mblas_data_type_enum::MBLAS_C_8U},
    {"CUDA_R_16I",  mblas_data_type_enum::MBLAS_R_16I},
    {"CUDA_C_16I",  mblas_data_type_enum::MBLAS_C_16I},
    {"CUDA_R_16U",  mblas_data_type_enum::MBLAS_R_16U},
    {"CUDA_C_16U",  mblas_data_type_enum::MBLAS_C_16U},
    {"CUDA_R_32I",  mblas_data_type_enum::MBLAS_R_32I},
    {"CUDA_C_32I",  mblas_data_type_enum::MBLAS_C_32I},
    {"CUDA_R_32U",  mblas_data_type_enum::MBLAS_R_32U},
    {"CUDA_C_32U",  mblas_data_type_enum::MBLAS_C_32U},
    {"CUDA_R_64I",  mblas_data_type_enum::MBLAS_R_64I},
    {"CUDA_C_64I",  mblas_data_type_enum::MBLAS_C_64I},
    {"CUDA_R_64U",  mblas_data_type_enum::MBLAS_R_64U},
    {"CUDA_C_64U",  mblas_data_type_enum::MBLAS_C_64U},
    {"CUDA_R_8F_E4M3", mblas_data_type_enum::MBLAS_R_8F_E4M3},
    {"CUDA_R_8F_E5M2", mblas_data_type_enum::MBLAS_R_8F_E5M2},
    {"CUDA_R_8F_UE4M3", mblas_data_type_enum::MBLAS_R_8F_UE4M3},
    {"CUDA_R_8F_UE8M0", mblas_data_type_enum::MBLAS_R_8F_UE8M0},
    {"CUDA_R_6F_E2M3", mblas_data_type_enum::MBLAS_R_6F_E2M3},
    {"CUDA_R_6F_E3M2", mblas_data_type_enum::MBLAS_R_6F_E3M2},
    {"CUDA_R_4F_E2M1", mblas_data_type_enum::MBLAS_R_4F_E2M1},
    // HIP
    {"HIP_R_16F",  mblas_data_type_enum::MBLAS_R_16F},
    {"HIP_C_16F",  mblas_data_type_enum::MBLAS_C_16F},
    {"HIP_R_16BF", mblas_data_type_enum::MBLAS_R_16BF},
    {"HIP_C_16BF", mblas_data_type_enum::MBLAS_C_16BF},
    {"HIP_R_32F",  mblas_data_type_enum::MBLAS_R_32F},
    {"HIP_C_32F",  mblas_data_type_enum::MBLAS_C_32F},
    {"HIP_R_64F",  mblas_data_type_enum::MBLAS_R_64F},
    {"HIP_C_64F",  mblas_data_type_enum::MBLAS_C_64F},
    {"HIP_R_4I",   mblas_data_type_enum::MBLAS_R_4I},
    {"HIP_C_4I",   mblas_data_type_enum::MBLAS_C_4I},
    {"HIP_R_4U",   mblas_data_type_enum::MBLAS_R_4U},
    {"HIP_C_4U",   mblas_data_type_enum::MBLAS_C_4U},
    {"HIP_R_8I",   mblas_data_type_enum::MBLAS_R_8I},
    {"HIP_C_8I",   mblas_data_type_enum::MBLAS_C_8I},
    {"HIP_R_8U",   mblas_data_type_enum::MBLAS_R_8U},
    {"HIP_C_8U",   mblas_data_type_enum::MBLAS_C_8U},
    {"HIP_R_16I",  mblas_data_type_enum::MBLAS_R_16I},
    {"HIP_C_16I",  mblas_data_type_enum::MBLAS_C_16I},
    {"HIP_R_16U",  mblas_data_type_enum::MBLAS_R_16U},
    {"HIP_C_16U",  mblas_data_type_enum::MBLAS_C_16U},
    {"HIP_R_32I",  mblas_data_type_enum::MBLAS_R_32I},
    {"HIP_C_32I",  mblas_data_type_enum::MBLAS_C_32I},
    {"HIP_R_32U",  mblas_data_type_enum::MBLAS_R_32U},
    {"HIP_C_32U",  mblas_data_type_enum::MBLAS_C_32U},
    {"HIP_R_64I",  mblas_data_type_enum::MBLAS_R_64I},
    {"HIP_C_64I",  mblas_data_type_enum::MBLAS_C_64I},
    {"HIP_R_64U",  mblas_data_type_enum::MBLAS_R_64U},
    {"HIP_C_64U",  mblas_data_type_enum::MBLAS_C_64U},
    {"HIP_R_8F_E4M3", mblas_data_type_enum::MBLAS_R_8F_E4M3},
    {"HIP_R_8F_E5M2", mblas_data_type_enum::MBLAS_R_8F_E5M2},
    {"HIP_R_8F_E4M3_FUNZ", mblas_data_type_enum::MBLAS_R_8F_E4M3},
    {"HIP_R_8F_E5M2_FUNZ", mblas_data_type_enum::MBLAS_R_8F_E5M2},
};

// Manually defined
bool mblas_data_type::operator==(const mblas_data_type& other) const {
  return value == other.value;
}

bool mblas_data_type::operator<(const mblas_data_type& other) const {
  return value < other.value;
}

// Defined based on above
bool mblas_data_type::operator!=(const mblas_data_type& other) const {
  return !(*this == other);
}

bool mblas_data_type::operator>(const mblas_data_type& other) const {
  return (!(*this == other)) && (!(*this < other));
}

bool mblas_data_type::operator<=(const mblas_data_type& other) const {
  return (*this == other) || (*this < other);
}

bool mblas_data_type::operator>=(const mblas_data_type& other) const {
  return !(*this < other);
}

mblas_data_type::mblas_data_type(std::string instr) {
  if (precDType.find(instr) != precDType.end())
    value = precDType.at(instr);
  else {
    value = mblas_data_type::MBLAS_NULL;
  }
}

std::string mblas_data_type::to_string(std::string prefix) const {
  for (auto ele : precDType) {
    if (ele.second == value && ele.first.find(prefix) != std::string::npos) {
      return ele.first;
    }
  }
  for (auto ele : precDType) {
    if (ele.second == value) {
      return ele.first;
    }
  }
  return "(DataType name not found)";
}

bool mblas_data_type::is_real() const {
  // Uses toString for maintenence reasons, not perf critical
  std::string str = to_string();
  if (str.find("_C_") != std::string::npos) {
    return false;
  }
  // Assume real
  return true;
}

bool mblas_data_type::is_fp8() const {
  if (value == MBLAS_R_8F_E4M3 || value == MBLAS_R_8F_E5M2) {
    return true;
  }
  return false;
}

bool mblas_data_type::is_fp4() const {
  if (value == MBLAS_R_4F_E2M1) {
    return true;
  }
  return false;
}

int mblas_data_type::get_packing_count() const {
  if (value == MBLAS_R_4F_E2M1) {
    // Two 4-bit FP4 floats per byte
    return 2;
  } else {
    return 1;
  }
}

void mblas_data_type::set_scalar(std::string scalarstr, mblas_data_type precision,
                            mblas_compute_type& compute) {
  if (scalarstr != "") {
    set(mblas_data_type(scalarstr));
    return;
  } else {
    // Scalar type not specified, setting based on compute type
    for (auto ele : precToCompute) {
      mblas_data_type selDtype = mblas_data_type(ele.first);
      mblas_compute_type selCtype = mblas_compute_type(ele.second);
      if (selCtype == compute && precision.is_real() == selDtype.is_real()) {
        set(selDtype);
        return;
      }
    }
    // something terrible has happened
    set(precision);
  }
}

//mblas_data_type::mblas_data_type() {
//  value = mblas_data_type_enum::MBLAS_NULL;
//}