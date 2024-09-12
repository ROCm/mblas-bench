#include "mblasCuComputeType.h"
#include <iostream>

// Used for converting mblas type to cublas type
const std::map<mblasComputeTypeEnum, cublasComputeType_t> mblasCuComputeType::compMappings = {
    {MBLAS_COMPUTE_16F, CUBLAS_COMPUTE_16F},
    {MBLAS_COMPUTE_16F_PEDANTIC, CUBLAS_COMPUTE_16F_PEDANTIC},
    {MBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F},
    {MBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_COMPUTE_32F_PEDANTIC},
    {MBLAS_COMPUTE_32F_FAST_16F, CUBLAS_COMPUTE_32F_FAST_16F},
    {MBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_COMPUTE_32F_FAST_16BF},
    {MBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_COMPUTE_32F_FAST_TF32},
    {MBLAS_COMPUTE_64F, CUBLAS_COMPUTE_64F},
    {MBLAS_COMPUTE_64F_PEDANTIC, CUBLAS_COMPUTE_64F_PEDANTIC},
    {MBLAS_COMPUTE_32I, CUBLAS_COMPUTE_32I},
    {MBLAS_COMPUTE_32I_PEDANTIC, CUBLAS_COMPUTE_32I_PEDANTIC},
};

cublasComputeType_t mblasCuComputeType::convertToCuda(mblasCuComputeType data)  { return convertToCuda(&data); }
cublasComputeType_t mblasCuComputeType::convertToCuda(const mblasCuComputeType *data) {
  try {
    return compMappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to CUDA Compute Type " << data->toString() << std::endl;
    throw e;
    return CUBLAS_COMPUTE_32F;
  }
}

// void mblasCuComputeType::operator = (const cublasComputeType_t cudt) {
//   for (auto ele : compMappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblasCuComputeType & mblasCuComputeType::operator = (const mblasCuComputeType& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}

mblasCuComputeType::operator cublasComputeType_t() const {
  return convertToCuda(this);
}
