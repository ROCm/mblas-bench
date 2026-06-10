#include "mblas_cuda_data_type.h"
#include <iostream>

cudaDataType mblas_cuda_data_type::convert_to_cuda(mblas_cuda_data_type data)  { return convert_to_cuda(&data); }
cudaDataType mblas_cuda_data_type::convert_to_cuda(const mblas_cuda_data_type *data) {
  try {
    return prec_mappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to CUDA Datatype " << data->to_string() << std::endl;
    throw e;
    return CUDA_R_32F;
  }
}

// void mblas_cuda_data_type::operator = (const cudaDataType cudt) {
//   for (auto ele : prec_mappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblas_cuda_data_type & mblas_cuda_data_type::operator = (const mblas_cuda_data_type& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}
// 
// mblas_cuda_data_type & mblas_cuda_data_type::operator = (const mblas_data_type& mdt) {
//   if (this == &mdt)
//     return *this;
//   // Use parent class default = operator
//   mblas_data_type * dest = dynamic_cast<mblas_data_type*>(this);
//   *dest = mdt;
//   return *this;
// }

mblas_cuda_data_type::operator cudaDataType() const {
  return convert_to_cuda(this);
}

mblas_cuda_data_type mblas_cuda_data_type::get_scale_type() {
  if (*this == MBLAS_R_8F_E4M3 || *this == MBLAS_R_8F_E5M2) {
    return mblas_data_type_enum::MBLAS_R_8F_UE8M0;
  } else if (*this == MBLAS_R_4F_E2M1 ) {
    return mblas_data_type_enum::MBLAS_R_8F_UE4M3;
  }
  return mblas_data_type_enum::MBLAS_R_32F;
}

#if (ENABLE_CUDA_FP4)
cublasLtMatmulMatrixScale_t mblas_cuda_data_type::get_scale_mode() {
  if (*this == MBLAS_R_8F_E4M3 || *this == MBLAS_R_8F_E5M2) {
    return CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  } else if (*this == MBLAS_R_4F_E2M1 ) {
    return CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  }
  return CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
}
#endif

const std::map<mblas_data_type, cudaDataType> mblas_cuda_data_type::prec_mappings = {
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
#if (ENABLE_CUDA_FP4)
    {MBLAS_R_8F_UE4M3, CUDA_R_8F_UE4M3},
    {MBLAS_R_8F_UE8M0, CUDA_R_8F_UE8M0},
    {MBLAS_R_6F_E2M3, CUDA_R_6F_E2M3},
    {MBLAS_R_6F_E3M2, CUDA_R_6F_E3M2},
    {MBLAS_R_4F_E2M1, CUDA_R_4F_E2M1},
#endif
};
