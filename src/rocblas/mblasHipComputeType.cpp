#include "mblasHipComputeType.h"
#include <iostream>

// Used for converting mblas type to cublas type
const std::map<mblasComputeTypeEnum, hipblasComputeType_t> mblasHipComputeType::compute_mappings = {
    {MBLAS_COMPUTE_16F, HIPBLAS_COMPUTE_16F},
    {MBLAS_COMPUTE_16F_PEDANTIC, HIPBLAS_COMPUTE_16F_PEDANTIC},
    {MBLAS_COMPUTE_32F, HIPBLAS_COMPUTE_32F},
    {MBLAS_COMPUTE_32F_PEDANTIC, HIPBLAS_COMPUTE_32F_PEDANTIC},
    {MBLAS_COMPUTE_32F_FAST_16F, HIPBLAS_COMPUTE_32F_FAST_16F},
    {MBLAS_COMPUTE_32F_FAST_16BF, HIPBLAS_COMPUTE_32F_FAST_16BF},
    {MBLAS_COMPUTE_32F_FAST_TF32, HIPBLAS_COMPUTE_32F_FAST_TF32},
    {MBLAS_COMPUTE_64F, HIPBLAS_COMPUTE_64F},
    {MBLAS_COMPUTE_64F_PEDANTIC, HIPBLAS_COMPUTE_64F_PEDANTIC},
    {MBLAS_COMPUTE_32I, HIPBLAS_COMPUTE_32I},
    {MBLAS_COMPUTE_32I_PEDANTIC, HIPBLAS_COMPUTE_32I_PEDANTIC},
};

hipblasComputeType_t mblasHipComputeType::convertToHip(mblasHipComputeType data)  { return convertToHip(&data); }
hipblasComputeType_t mblasHipComputeType::convertToHip(const mblasHipComputeType *data) {
  try {
    return compute_mappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to Hip Compute Type " << data->toString() << std::endl;
    throw e;
    return HIPBLAS_COMPUTE_32F;
  }
}

// void mblasHipComputeType::operator = (const hipblasComputeType_t cudt) {
//   for (auto ele : compute_mappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblasHipComputeType & mblasHipComputeType::operator = (const mblasHipComputeType& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}

mblasHipComputeType::operator hipblasComputeType_t() const {
  return convertToHip(this);
}
