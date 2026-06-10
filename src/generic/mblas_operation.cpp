#include "mblas_operation.h"
#include <string>
#include <iostream>

const std::map<std::string, mblas_operation_enum> mblas_operation::opDType = {
    {"N",  mblas_operation::MBLAS_OP_N},
    {"T",  mblas_operation::MBLAS_OP_T},
    {"C",  mblas_operation::MBLAS_OP_C},
    {"CONJG",  mblas_operation::MBLAS_OP_CONJG},
    // MBLAS
    {"MBLAS_OP_N",  mblas_operation::MBLAS_OP_N},
    {"MBLAS_OP_T",  mblas_operation::MBLAS_OP_T},
    {"MBLAS_OP_C",  mblas_operation::MBLAS_OP_C},
    {"MBLAS_OP_CONJG",  mblas_operation::MBLAS_OP_CONJG},
    //{"MBLAS_OP_NULL",  mblas_operation::MBLAS_OP_NULL},
    // CUBLAS
    {"CUBLAS_OP_N",  mblas_operation::MBLAS_OP_N},
    {"CUBLAS_OP_T",  mblas_operation::MBLAS_OP_T},
    {"CUBLAS_OP_C",  mblas_operation::MBLAS_OP_C},
    {"CUBLAS_OP_HERMITAN",  mblas_operation::MBLAS_OP_C},
    {"CUBLAS_OP_CONJG",  mblas_operation::MBLAS_OP_CONJG},
    // ROCBLAS
    {"rocblas_operation_none", mblas_operation::MBLAS_OP_N},
    {"rocblas_operation_transpose", mblas_operation::MBLAS_OP_T},
    {"rocblas_operation_conjugate_transpose", mblas_operation::MBLAS_OP_C},
    // HIPBLAS
    {"HIPBLAS_OP_N",  mblas_operation::MBLAS_OP_N},
    {"HIPBLAS_OP_T",  mblas_operation::MBLAS_OP_T},
    {"HIPBLAS_OP_C",  mblas_operation::MBLAS_OP_C},
};

const std::map<mblas_operation_enum, std::string> mblas_operation::opSShort = {
  {mblas_operation::MBLAS_OP_N, "N"},
  {mblas_operation::MBLAS_OP_T, "T"},
  {mblas_operation::MBLAS_OP_C, "C"},
  {mblas_operation::MBLAS_OP_CONJG, "CONJG"},
};

// Manually defined
bool mblas_operation::operator==(const mblas_operation& other) const {
  return value == other.value;
}

bool mblas_operation::operator<(const mblas_operation& other) const {
  return value < other.value;
}

// Defined based on above
bool mblas_operation::operator!=(const mblas_operation& other) const {
  return !(*this == other);
}

bool mblas_operation::operator>(const mblas_operation& other) const {
  return (!(*this == other)) && (!(*this < other));
}

bool mblas_operation::operator<=(const mblas_operation& other) const {
  return (*this == other) || (*this < other);
}

bool mblas_operation::operator>=(const mblas_operation& other) const {
  return !(*this < other);
}

mblas_operation::mblas_operation(std::string instr) {
  if (opDType.find(instr) != opDType.end())
    value = opDType.at(instr);
  else {
    value = mblas_operation::MBLAS_OP_NULL;
  }
}

std::string mblas_operation::to_string(std::string prefix) const {
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

std::string mblas_operation::to_string_short() {
  try {
    return opSShort.at(value);
  } catch (std::out_of_range &e) {
    std::cerr << "Failed to parse op." << std::endl;
    throw e;
    return "(Operation name not found)";
  }
}