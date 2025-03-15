#include "mblasCuDataType.h"
#include <iostream>

cudaDataType mblasCuDataType::convertToCuda(mblasCuDataType data)  { return convertToCuda(&data); }
cudaDataType mblasCuDataType::convertToCuda(const mblasCuDataType *data) {
  try {
    return precMappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to CUDA Datatype " << data->toString() << std::endl;
    throw e;
    return CUDA_R_32F;
  }
}

// void mblasCuDataType::operator = (const cudaDataType cudt) {
//   for (auto ele : precMappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblasCuDataType & mblasCuDataType::operator = (const mblasCuDataType& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}
// 
// mblasCuDataType & mblasCuDataType::operator = (const mblasDataType& mdt) {
//   if (this == &mdt)
//     return *this;
//   // Use parent class default = operator
//   mblasDataType * dest = dynamic_cast<mblasDataType*>(this);
//   *dest = mdt;
//   return *this;
// }

mblasCuDataType::operator cudaDataType() const {
  return convertToCuda(this);
}

mblasCuDataType mblasCuDataType::get_scale_type() {
  if (*this == MBLAS_R_8F_E4M3 || *this == MBLAS_R_8F_E5M2) {
    return mblasDataTypeEnum::MBLAS_R_8F_UE8M0;
  } else if (*this == MBLAS_R_4F_E2M1 ) {
    return mblasDataTypeEnum::MBLAS_R_8F_UE4M3;
  }
  return mblasDataTypeEnum::MBLAS_R_32F;
}

cublasLtMatmulMatrixScale_t mblasCuDataType::get_scale_mode() {
  if (*this == MBLAS_R_8F_E4M3 || *this == MBLAS_R_8F_E5M2) {
    return CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  } else if (*this == MBLAS_R_4F_E2M1 ) {
    return CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  }
  return CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
}

const std::map<mblasDataType, cudaDataType> mblasCuDataType::precMappings = {
    {MBLAS_R_16F,  CUDA_R_16F},
    {MBLAS_C_16F,  CUDA_C_16F},
    {MBLAS_R_16BF, CUDA_R_16BF},
    {MBLAS_C_16BF, CUDA_C_16BF},
    {MBLAS_R_32F,  CUDA_R_32F},
    {MBLAS_C_32F,  CUDA_C_32F},
    {MBLAS_R_64F,  CUDA_R_64F},
    {MBLAS_C_64F,  CUDA_C_64F},
    {MBLAS_R_4I,   CUDA_R_4I},
    {MBLAS_C_4I,   CUDA_C_4I},
    {MBLAS_R_4U,   CUDA_R_4U},
    {MBLAS_C_4U,   CUDA_C_4U},
    {MBLAS_R_8I,   CUDA_R_8I},
    {MBLAS_C_8I,   CUDA_C_8I},
    {MBLAS_R_8U,   CUDA_R_8U},
    {MBLAS_C_8U,   CUDA_C_8U},
    {MBLAS_R_16I,  CUDA_R_16I},
    {MBLAS_C_16I,  CUDA_C_16I},
    {MBLAS_R_16U,  CUDA_R_16U},
    {MBLAS_C_16U,  CUDA_C_16U},
    {MBLAS_R_32I,  CUDA_R_32I},
    {MBLAS_C_32I,  CUDA_C_32I},
    {MBLAS_R_32U,  CUDA_R_32U},
    {MBLAS_C_32U,  CUDA_C_32U},
    {MBLAS_R_64I,  CUDA_R_64I},
    {MBLAS_C_64I,  CUDA_C_64I},
    {MBLAS_R_64U,  CUDA_R_64U},
    {MBLAS_C_64U,  CUDA_C_64U},
    {MBLAS_R_8F_E4M3, CUDA_R_8F_E4M3},
    {MBLAS_R_8F_E5M2, CUDA_R_8F_E5M2},
    {MBLAS_R_8F_UE4M3, CUDA_R_8F_UE4M3},
    {MBLAS_R_8F_UE8M0, CUDA_R_8F_UE8M0},
    {MBLAS_R_6F_E2M3, CUDA_R_6F_E2M3},
    {MBLAS_R_6F_E3M2, CUDA_R_6F_E3M2},
    {MBLAS_R_4F_E2M1, CUDA_R_4F_E2M1},
};
