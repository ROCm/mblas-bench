#include "mblasHipDataType.h"
#include <iostream>

hipDataType mblasHipDataType::convertToHip(mblasHipDataType data)  { return convertToHip(&data); }
hipDataType mblasHipDataType::convertToHip(const mblasHipDataType *data) {
  try {
    return prec_mappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to Hip Datatype " << data->toString() << std::endl;
    throw e;
    return HIP_R_32F;
  }
}

// void mblasHipDataType::operator = (const hipDataType cudt) {
//   for (auto ele : prec_mappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblasHipDataType & mblasHipDataType::operator = (const mblasHipDataType& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}
// 
// mblasHipDataType & mblasHipDataType::operator = (const mblasDataType& mdt) {
//   if (this == &mdt)
//     return *this;
//   // Use parent class default = operator
//   mblasDataType * dest = dynamic_cast<mblasDataType*>(this);
//   *dest = mdt;
//   return *this;
// }

mblasHipDataType::operator hipDataType() const {
  return convertToHip(this);
}

const std::map<mblasDataType, hipDataType> mblasHipDataType::prec_mappings = {
    {MBLAS_R_16F,  HIP_R_16F},
    {MBLAS_C_16F,  HIP_C_16F},
    {MBLAS_R_16BF, HIP_R_16BF},
    {MBLAS_C_16BF, HIP_C_16BF},
    {MBLAS_R_32F,  HIP_R_32F},
    {MBLAS_C_32F,  HIP_C_32F},
    {MBLAS_R_64F,  HIP_R_64F},
    {MBLAS_C_64F,  HIP_C_64F},
    {MBLAS_R_4I,   HIP_R_4I},
    {MBLAS_C_4I,   HIP_C_4I},
    {MBLAS_R_4U,   HIP_R_4U},
    {MBLAS_C_4U,   HIP_C_4U},
    {MBLAS_R_8I,   HIP_R_8I},
    {MBLAS_C_8I,   HIP_C_8I},
    {MBLAS_R_8U,   HIP_R_8U},
    {MBLAS_C_8U,   HIP_C_8U},
    {MBLAS_R_16I,  HIP_R_16I},
    {MBLAS_C_16I,  HIP_C_16I},
    {MBLAS_R_16U,  HIP_R_16U},
    {MBLAS_C_16U,  HIP_C_16U},
    {MBLAS_R_32I,  HIP_R_32I},
    {MBLAS_C_32I,  HIP_C_32I},
    {MBLAS_R_32U,  HIP_R_32U},
    {MBLAS_C_32U,  HIP_C_32U},
    {MBLAS_R_64I,  HIP_R_64I},
    {MBLAS_C_64I,  HIP_C_64I},
    {MBLAS_R_64U,  HIP_R_64U},
    {MBLAS_C_64U,  HIP_C_64U},
    {MBLAS_R_8F_E4M3, HIP_R_8F_E4M3_FNUZ},
    {MBLAS_R_8F_E5M2, HIP_R_8F_E5M2_FNUZ},
};
