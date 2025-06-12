#include "mblasRocDataType.h"
#include <iostream>

rocblas_datatype mblasRocDataType::convertToHip(mblasRocDataType data)  { return convertToHip(&data); }

rocblas_datatype mblasRocDataType::convertToHip(const mblasRocDataType *data) {
  try {
    return prec_mappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to Rocblas Datatype " << data->toString() << std::endl;
    throw e;
    return rocblas_datatype_f32_r;
  }
}

mblasRocDataType::operator rocblas_datatype() const {
  return convertToHip(this);
}

// void mblasRocDataType::operator = (const hipDataType cudt) {
//   for (auto ele : prec_mappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblasRocDataType & mblasRocDataType::operator = (const mblasRocDataType& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}
// 
// mblasRocDataType & mblasRocDataType::operator = (const mblasDataType& mdt) {
//   if (this == &mdt)
//     return *this;
//   // Use parent class default = operator
//   mblasDataType * dest = dynamic_cast<mblasDataType*>(this);
//   *dest = mdt;
//   return *this;
// }

const std::map<mblasDataType, rocblas_datatype> mblasRocDataType::prec_mappings = {
    {MBLAS_R_16F,  rocblas_datatype_f16_r},
    {MBLAS_C_16F,  rocblas_datatype_f16_c},
    {MBLAS_R_16BF, rocblas_datatype_bf16_r},
    {MBLAS_C_16BF, rocblas_datatype_bf16_c},
    {MBLAS_R_32F,  rocblas_datatype_f32_r},
    {MBLAS_C_32F,  rocblas_datatype_f32_c},
    {MBLAS_R_64F,  rocblas_datatype_f64_r},
    {MBLAS_C_64F,  rocblas_datatype_f64_c},
    {MBLAS_R_8I,   rocblas_datatype_i8_r},
    {MBLAS_C_8I,   rocblas_datatype_i8_c},
    {MBLAS_R_8U,   rocblas_datatype_u8_r},
    {MBLAS_C_8U,   rocblas_datatype_u8_c},
    {MBLAS_R_32I,  rocblas_datatype_i32_r},
    {MBLAS_C_32I,  rocblas_datatype_i32_c},
    {MBLAS_R_32U,  rocblas_datatype_u32_r},
    {MBLAS_C_32U,  rocblas_datatype_u32_c},
    {MBLAS_R_8F_E4M3, rocblas_datatype_f8_r},
    {MBLAS_R_8F_E5M2, rocblas_datatype_bf8_r},
};
