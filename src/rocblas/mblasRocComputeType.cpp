#include "mblasRocComputeType.h"

#include <iostream>

#include "mblasRocDataType.h"

// Used for converting mblas type to rocblas type
const std::map<mblasComputeTypeEnum, rocblas_computetype> mblasRocComputeType::compute_mappings = {
    {MBLAS_COMPUTE_32F, rocblas_compute_type_f32},
    {MBLAS_COMPUTE_32F_8F_8F, rocblas_compute_type_f8_f8_f32},
    {MBLAS_COMPUTE_32F_8F_8BF, rocblas_compute_type_f8_bf8_f32},
    {MBLAS_COMPUTE_32F_8BF_8F, rocblas_compute_type_bf8_f8_f32},
    {MBLAS_COMPUTE_32F_8BF_8BF, rocblas_compute_type_bf8_bf8_f32,},
    {MBLAS_COMPUTE_NULL, rocblas_compute_type_invalid},
};

const std::vector<std::pair<mblasComputeTypeEnum, mblasDataTypeEnum>> prec_mappings = {
    {MBLAS_COMPUTE_64F, MBLAS_R_64F},
    {MBLAS_COMPUTE_32F, MBLAS_R_32F},
    {MBLAS_COMPUTE_16F, MBLAS_R_16F},
    {MBLAS_COMPUTE_32I, MBLAS_R_32I},
    {MBLAS_COMPUTE_64F, MBLAS_C_64F},
    {MBLAS_COMPUTE_32F, MBLAS_C_32F},
};

rocblas_computetype mblasRocComputeType::convertToRocm(mblasRocComputeType data)  { return convertToRocm(&data); }

rocblas_computetype mblasRocComputeType::convertToRocm(const mblasRocComputeType *data) {
  try {
    return compute_mappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to rocBLAS Compute Type " << data->toString() << std::endl;
    throw e;
    return rocblas_compute_type_f32;
  }
}

mblasRocComputeType::operator rocblas_computetype() const {
  return convertToRocm(this);
}

rocblas_datatype mblasRocComputeType::convertToRocmData(const mblasRocComputeType *data) {
  for (auto ele : prec_mappings) {
    mblasRocDataType rdata = mblasRocDataType(ele.second);
    if (ele.first == *data && rdata.isReal() == data->rocIsReal) {
      return rocblas_datatype(rdata);
    }
  }
  std::cout << "Failed to convert to rocBLAS Data Type " << data->toString() << std::endl;
  throw std::out_of_range("Value not found in list");
}

mblasRocComputeType::operator rocblas_datatype() const {
  return convertToRocmData(this);
}

// void mblasRocComputeType::operator = (const rocblas_computetype cudt) {
//   for (auto ele : compute_mappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblasRocComputeType & mblasRocComputeType::operator = (const mblasRocComputeType& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}

void mblasRocComputeType::set_compute(std::string computestr, mblasDataType& precision) {
  mblasComputeType::set_compute(computestr, precision);
  rocIsReal = precision.isReal();
}
