#include "mblasOperation.h"
#include <string>
#include <iostream>

const std::map<std::string, mblasOperationEnum> mblasOperation::opDType = {
    {"N",  mblasOperation::MBLAS_OP_N},
    {"T",  mblasOperation::MBLAS_OP_T},
    {"C",  mblasOperation::MBLAS_OP_C},
    {"CONJG",  mblasOperation::MBLAS_OP_CONJG},
    // MBLAS
    {"MBLAS_OP_N",  mblasOperation::MBLAS_OP_N},
    {"MBLAS_OP_T",  mblasOperation::MBLAS_OP_T},
    {"MBLAS_OP_C",  mblasOperation::MBLAS_OP_C},
    {"MBLAS_OP_CONJG",  mblasOperation::MBLAS_OP_CONJG},
    //{"MBLAS_OP_NULL",  mblasOperation::MBLAS_OP_NULL},
    // CUBLAS
    {"CUBLAS_OP_N",  mblasOperation::MBLAS_OP_N},
    {"CUBLAS_OP_T",  mblasOperation::MBLAS_OP_T},
    {"CUBLAS_OP_C",  mblasOperation::MBLAS_OP_C},
    {"CUBLAS_OP_HERMITAN",  mblasOperation::MBLAS_OP_C},
    {"CUBLAS_OP_CONJG",  mblasOperation::MBLAS_OP_CONJG},
    // ROCBLAS
    {"rocblas_operation_none", mblasOperation::MBLAS_OP_N},
    {"rocblas_operation_transpose", mblasOperation::MBLAS_OP_T},
    {"rocblas_operation_conjugate_transpose", mblasOperation::MBLAS_OP_C},
    // HIPBLAS
    {"HIPBLAS_OP_N",  mblasOperation::MBLAS_OP_N},
    {"HIPBLAS_OP_T",  mblasOperation::MBLAS_OP_T},
    {"HIPBLAS_OP_C",  mblasOperation::MBLAS_OP_C},
};

const std::map<mblasOperationEnum, std::string> mblasOperation::opSShort = {
  {mblasOperation::MBLAS_OP_N, "N"},
  {mblasOperation::MBLAS_OP_T, "T"},
  {mblasOperation::MBLAS_OP_C, "C"},
  {mblasOperation::MBLAS_OP_CONJG, "CONJG"},
};

// Manually defined
bool mblasOperation::operator==(const mblasOperation& other) const {
  return value == other.value;
}

bool mblasOperation::operator<(const mblasOperation& other) const {
  return value < other.value;
}

// Defined based on above
bool mblasOperation::operator!=(const mblasOperation& other) const {
  return !(*this == other);
}

bool mblasOperation::operator>(const mblasOperation& other) const {
  return (!(*this == other)) && (!(*this < other));
}

bool mblasOperation::operator<=(const mblasOperation& other) const {
  return (*this == other) || (*this < other);
}

bool mblasOperation::operator>=(const mblasOperation& other) const {
  return !(*this < other);
}

mblasOperation::mblasOperation(std::string instr) {
  if (opDType.find(instr) != opDType.end())
    value = opDType.at(instr);
  else {
    value = mblasOperation::MBLAS_OP_NULL;
  }
}

std::string mblasOperation::to_string(std::string prefix) const {
  for (auto ele : opDType) {
    if (ele.second == value && ele.first.find(prefix) != std::string::npos) {
      return ele.first;
    }
  }
  for (auto ele : opDType) {
    if (ele.second == value) {
      return ele.first;
    }
  }
  return "(Operation name not found)";
}

std::string mblasOperation::to_string_short() {
  try {
    return opSShort.at(value);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse op." << std::endl;
    throw e;
    return "(Operation name not found)";
  }
}