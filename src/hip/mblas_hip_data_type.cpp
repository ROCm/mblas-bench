#include "mblas_hip_data_type.h"
#include <iostream>

hipDataType mblas_hip_data_type::convert_to_hip(mblas_hip_data_type data)  { return convert_to_hip(&data); }
hipDataType mblas_hip_data_type::convert_to_hip(const mblas_hip_data_type *data) {
  try {
    return prec_mappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to Hip Datatype " << data->to_string() << std::endl;
    throw e;
    return HIP_R_32F;
  }
}

// void mblas_hip_data_type::operator = (const hipDataType cudt) {
//   for (auto ele : prec_mappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblas_hip_data_type & mblas_hip_data_type::operator = (const mblas_hip_data_type& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}
// 
// mblas_hip_data_type & mblas_hip_data_type::operator = (const mblas_data_type& mdt) {
//   if (this == &mdt)
//     return *this;
//   // Use parent class default = operator
//   mblas_data_type * dest = dynamic_cast<mblas_data_type*>(this);
//   *dest = mdt;
//   return *this;
// }

mblas_hip_data_type::operator hipDataType() const {
  return convert_to_hip(this);
}

const std::map<mblas_data_type, hipDataType> mblas_hip_data_type::prec_mappings = {
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
    {MBLAS_R_8F_E4M3, HIP_R_8F_E4M3},
    {MBLAS_R_8F_E5M2, HIP_R_8F_E5M2},
    #if defined(HIPRT_VERSION) && HIPRT_VERSION >= 70000000
    {MBLAS_R_6F_E2M3, HIP_R_6F_E2M3},
    {MBLAS_R_6F_E3M2, HIP_R_6F_E3M2},
    {MBLAS_R_4F_E2M1, HIP_R_4F_E2M1},
    #endif
};
