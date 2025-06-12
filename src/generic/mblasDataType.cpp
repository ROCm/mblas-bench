#include "mblasDataType.h"
#include "mblasComputeType.h"
#include <string>
#include <iostream>

const std::map<std::string, mblasDataTypeEnum> mblasDataType::precDType = {
    // Generic, similar to rocblas format
    {"h", mblasDataTypeEnum::MBLAS_R_16F},       {"s", mblasDataTypeEnum::MBLAS_R_32F},     {"d", mblasDataTypeEnum::MBLAS_R_64F},
    {"c", mblasDataTypeEnum::MBLAS_C_32F},       {"z", mblasDataTypeEnum::MBLAS_C_64F},     {"f16_r", mblasDataTypeEnum::MBLAS_R_16F},
    {"f16_c", mblasDataTypeEnum::MBLAS_C_16F},   {"f32_r", mblasDataTypeEnum::MBLAS_R_32F}, {"f32_c", mblasDataTypeEnum::MBLAS_C_32F},
    {"f64_r", mblasDataTypeEnum::MBLAS_R_64F},   {"f64_c", mblasDataTypeEnum::MBLAS_C_64F}, {"bf16_r", mblasDataTypeEnum::MBLAS_R_16BF},
    {"bf16_c", mblasDataTypeEnum::MBLAS_C_16BF}, {"i8_r", mblasDataTypeEnum::MBLAS_R_8I},   {"i8_c", mblasDataTypeEnum::MBLAS_C_8I},
    {"i32_r", mblasDataTypeEnum::MBLAS_R_32I},   {"i32_c", mblasDataTypeEnum::MBLAS_C_32I},
    {"f8_r", mblasDataTypeEnum::MBLAS_R_8F_E4M3},   {"bf8_r", mblasDataTypeEnum::MBLAS_R_8F_E5M2},
    {"f6_r", mblasDataTypeEnum::MBLAS_R_6F_E3M2},
    {"f4_r", mblasDataTypeEnum::MBLAS_R_4F_E2M1},
    // MBLAS
    {"MBLAS_ANY",  mblasDataTypeEnum::MBLAS_ANY},
    {"MBLAS_NULL",  mblasDataTypeEnum::MBLAS_NULL},
    {"MBLAS_R_16F",  mblasDataTypeEnum::MBLAS_R_16F},
    {"MBLAS_C_16F",  mblasDataTypeEnum::MBLAS_C_16F},
    {"MBLAS_R_16BF", mblasDataTypeEnum::MBLAS_R_16BF},
    {"MBLAS_C_16BF", mblasDataTypeEnum::MBLAS_C_16BF},
    {"MBLAS_R_32F",  mblasDataTypeEnum::MBLAS_R_32F},
    {"MBLAS_C_32F",  mblasDataTypeEnum::MBLAS_C_32F},
    {"MBLAS_R_64F",  mblasDataTypeEnum::MBLAS_R_64F},
    {"MBLAS_C_64F",  mblasDataTypeEnum::MBLAS_C_64F},
    {"MBLAS_R_4I",   mblasDataTypeEnum::MBLAS_R_4I},
    {"MBLAS_C_4I",   mblasDataTypeEnum::MBLAS_C_4I},
    {"MBLAS_R_4U",   mblasDataTypeEnum::MBLAS_R_4U},
    {"MBLAS_C_4U",   mblasDataTypeEnum::MBLAS_C_4U},
    {"MBLAS_R_8I",   mblasDataTypeEnum::MBLAS_R_8I},
    {"MBLAS_C_8I",   mblasDataTypeEnum::MBLAS_C_8I},
    {"MBLAS_R_8U",   mblasDataTypeEnum::MBLAS_R_8U},
    {"MBLAS_C_8U",   mblasDataTypeEnum::MBLAS_C_8U},
    {"MBLAS_R_16I",  mblasDataTypeEnum::MBLAS_R_16I},
    {"MBLAS_C_16I",  mblasDataTypeEnum::MBLAS_C_16I},
    {"MBLAS_R_16U",  mblasDataTypeEnum::MBLAS_R_16U},
    {"MBLAS_C_16U",  mblasDataTypeEnum::MBLAS_C_16U},
    {"MBLAS_R_32I",  mblasDataTypeEnum::MBLAS_R_32I},
    {"MBLAS_C_32I",  mblasDataTypeEnum::MBLAS_C_32I},
    {"MBLAS_R_32U",  mblasDataTypeEnum::MBLAS_R_32U},
    {"MBLAS_C_32U",  mblasDataTypeEnum::MBLAS_C_32U},
    {"MBLAS_R_64I",  mblasDataTypeEnum::MBLAS_R_64I},
    {"MBLAS_C_64I",  mblasDataTypeEnum::MBLAS_C_64I},
    {"MBLAS_R_64U",  mblasDataTypeEnum::MBLAS_R_64U},
    {"MBLAS_C_64U",  mblasDataTypeEnum::MBLAS_C_64U},
    {"MBLAS_R_8F_E4M3", mblasDataTypeEnum::MBLAS_R_8F_E4M3},
    {"MBLAS_R_8F_E5M2", mblasDataTypeEnum::MBLAS_R_8F_E5M2},
    {"MBLAS_R_8F_UE4M3", mblasDataTypeEnum::MBLAS_R_8F_UE4M3},
    {"MBLAS_R_8F_UE8M0", mblasDataTypeEnum::MBLAS_R_8F_UE8M0},
    {"MBLAS_R_6F_E2M3", mblasDataTypeEnum::MBLAS_R_6F_E2M3},
    {"MBLAS_R_6F_E3M2", mblasDataTypeEnum::MBLAS_R_6F_E3M2},
    {"MBLAS_R_4F_E2M1", mblasDataTypeEnum::MBLAS_R_4F_E2M1},
    // CUDA
    {"CUDA_R_16F",  mblasDataTypeEnum::MBLAS_R_16F},
    {"CUDA_C_16F",  mblasDataTypeEnum::MBLAS_C_16F},
    {"CUDA_R_16BF", mblasDataTypeEnum::MBLAS_R_16BF},
    {"CUDA_C_16BF", mblasDataTypeEnum::MBLAS_C_16BF},
    {"CUDA_R_32F",  mblasDataTypeEnum::MBLAS_R_32F},
    {"CUDA_C_32F",  mblasDataTypeEnum::MBLAS_C_32F},
    {"CUDA_R_64F",  mblasDataTypeEnum::MBLAS_R_64F},
    {"CUDA_C_64F",  mblasDataTypeEnum::MBLAS_C_64F},
    {"CUDA_R_4I",   mblasDataTypeEnum::MBLAS_R_4I},
    {"CUDA_C_4I",   mblasDataTypeEnum::MBLAS_C_4I},
    {"CUDA_R_4U",   mblasDataTypeEnum::MBLAS_R_4U},
    {"CUDA_C_4U",   mblasDataTypeEnum::MBLAS_C_4U},
    {"CUDA_R_8I",   mblasDataTypeEnum::MBLAS_R_8I},
    {"CUDA_C_8I",   mblasDataTypeEnum::MBLAS_C_8I},
    {"CUDA_R_8U",   mblasDataTypeEnum::MBLAS_R_8U},
    {"CUDA_C_8U",   mblasDataTypeEnum::MBLAS_C_8U},
    {"CUDA_R_16I",  mblasDataTypeEnum::MBLAS_R_16I},
    {"CUDA_C_16I",  mblasDataTypeEnum::MBLAS_C_16I},
    {"CUDA_R_16U",  mblasDataTypeEnum::MBLAS_R_16U},
    {"CUDA_C_16U",  mblasDataTypeEnum::MBLAS_C_16U},
    {"CUDA_R_32I",  mblasDataTypeEnum::MBLAS_R_32I},
    {"CUDA_C_32I",  mblasDataTypeEnum::MBLAS_C_32I},
    {"CUDA_R_32U",  mblasDataTypeEnum::MBLAS_R_32U},
    {"CUDA_C_32U",  mblasDataTypeEnum::MBLAS_C_32U},
    {"CUDA_R_64I",  mblasDataTypeEnum::MBLAS_R_64I},
    {"CUDA_C_64I",  mblasDataTypeEnum::MBLAS_C_64I},
    {"CUDA_R_64U",  mblasDataTypeEnum::MBLAS_R_64U},
    {"CUDA_C_64U",  mblasDataTypeEnum::MBLAS_C_64U},
    {"CUDA_R_8F_E4M3", mblasDataTypeEnum::MBLAS_R_8F_E4M3},
    {"CUDA_R_8F_E5M2", mblasDataTypeEnum::MBLAS_R_8F_E5M2},
    {"CUDA_R_8F_UE4M3", mblasDataTypeEnum::MBLAS_R_8F_UE4M3},
    {"CUDA_R_8F_UE8M0", mblasDataTypeEnum::MBLAS_R_8F_UE8M0},
    {"CUDA_R_6F_E2M3", mblasDataTypeEnum::MBLAS_R_6F_E2M3},
    {"CUDA_R_6F_E3M2", mblasDataTypeEnum::MBLAS_R_6F_E3M2},
    {"CUDA_R_4F_E2M1", mblasDataTypeEnum::MBLAS_R_4F_E2M1},
    // HIP
    {"HIP_R_16F",  mblasDataTypeEnum::MBLAS_R_16F},
    {"HIP_C_16F",  mblasDataTypeEnum::MBLAS_C_16F},
    {"HIP_R_16BF", mblasDataTypeEnum::MBLAS_R_16BF},
    {"HIP_C_16BF", mblasDataTypeEnum::MBLAS_C_16BF},
    {"HIP_R_32F",  mblasDataTypeEnum::MBLAS_R_32F},
    {"HIP_C_32F",  mblasDataTypeEnum::MBLAS_C_32F},
    {"HIP_R_64F",  mblasDataTypeEnum::MBLAS_R_64F},
    {"HIP_C_64F",  mblasDataTypeEnum::MBLAS_C_64F},
    {"HIP_R_4I",   mblasDataTypeEnum::MBLAS_R_4I},
    {"HIP_C_4I",   mblasDataTypeEnum::MBLAS_C_4I},
    {"HIP_R_4U",   mblasDataTypeEnum::MBLAS_R_4U},
    {"HIP_C_4U",   mblasDataTypeEnum::MBLAS_C_4U},
    {"HIP_R_8I",   mblasDataTypeEnum::MBLAS_R_8I},
    {"HIP_C_8I",   mblasDataTypeEnum::MBLAS_C_8I},
    {"HIP_R_8U",   mblasDataTypeEnum::MBLAS_R_8U},
    {"HIP_C_8U",   mblasDataTypeEnum::MBLAS_C_8U},
    {"HIP_R_16I",  mblasDataTypeEnum::MBLAS_R_16I},
    {"HIP_C_16I",  mblasDataTypeEnum::MBLAS_C_16I},
    {"HIP_R_16U",  mblasDataTypeEnum::MBLAS_R_16U},
    {"HIP_C_16U",  mblasDataTypeEnum::MBLAS_C_16U},
    {"HIP_R_32I",  mblasDataTypeEnum::MBLAS_R_32I},
    {"HIP_C_32I",  mblasDataTypeEnum::MBLAS_C_32I},
    {"HIP_R_32U",  mblasDataTypeEnum::MBLAS_R_32U},
    {"HIP_C_32U",  mblasDataTypeEnum::MBLAS_C_32U},
    {"HIP_R_64I",  mblasDataTypeEnum::MBLAS_R_64I},
    {"HIP_C_64I",  mblasDataTypeEnum::MBLAS_C_64I},
    {"HIP_R_64U",  mblasDataTypeEnum::MBLAS_R_64U},
    {"HIP_C_64U",  mblasDataTypeEnum::MBLAS_C_64U},
    {"HIP_R_8F_E4M3", mblasDataTypeEnum::MBLAS_R_8F_E4M3},
    {"HIP_R_8F_E5M2", mblasDataTypeEnum::MBLAS_R_8F_E5M2},
    {"HIP_R_8F_E4M3_FUNZ", mblasDataTypeEnum::MBLAS_R_8F_E4M3},
    {"HIP_R_8F_E5M2_FUNZ", mblasDataTypeEnum::MBLAS_R_8F_E5M2},
};

// Manually defined
bool mblasDataType::operator==(const mblasDataType& other) const {
  return value == other.value;
}

bool mblasDataType::operator<(const mblasDataType& other) const {
  return value < other.value;
}

// Defined based on above
bool mblasDataType::operator!=(const mblasDataType& other) const {
  return !(*this == other);
}

bool mblasDataType::operator>(const mblasDataType& other) const {
  return (!(*this == other)) && (!(*this < other));
}

bool mblasDataType::operator<=(const mblasDataType& other) const {
  return (*this == other) || (*this < other);
}

bool mblasDataType::operator>=(const mblasDataType& other) const {
  return !(*this < other);
}

mblasDataType::mblasDataType(std::string instr) {
  if (precDType.find(instr) != precDType.end())
    value = precDType.at(instr);
  else {
    value = mblasDataType::MBLAS_NULL;
  }
}

std::string mblasDataType::to_string(std::string prefix) const {
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

bool mblasDataType::isReal() const {
  // Uses toString for maintenence reasons, not perf critical
  std::string str = to_string();
  if (str.find("_C_") != std::string::npos) {
    return false;
  }
  // Assume real
  return true;
}

bool mblasDataType::isFp8() const {
  if (value == MBLAS_R_8F_E4M3 || value == MBLAS_R_8F_E5M2) {
    return true;
  }
  return false;
}

bool mblasDataType::isFp4() const {
  if (value == MBLAS_R_4F_E2M1) {
    return true;
  }
  return false;
}

void mblasDataType::set_scalar(std::string scalarstr, mblasDataType precision,
                            mblasComputeType& compute) {
  if (scalarstr != "") {
    set(mblasDataType(scalarstr));
    return;
  } else {
    // Scalar type not specified, setting based on compute type
    for (auto ele : precToCompute) {
      mblasDataType selDtype = mblasDataType(ele.first);
      mblasComputeType selCtype = mblasComputeType(ele.second);
      if (selCtype == compute && precision.isReal() == selDtype.isReal()) {
        set(selDtype);
        return;
      }
    }
    // something terrible has happened
    set(precision);
  }
}

//mblasDataType::mblasDataType() {
//  value = mblasDataTypeEnum::MBLAS_NULL;
//}