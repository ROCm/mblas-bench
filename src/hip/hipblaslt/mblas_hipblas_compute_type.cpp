#include "mblas_hipblas_compute_type.h"
#include <iostream>

// Used for converting mblas type to cublas type
const std::map<mblas_compute_type_enum, hipblasComputeType_t> mblas_hipblas_compute_type::compute_mappings = {
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

hipblasComputeType_t mblas_hipblas_compute_type::convert_to_hip(mblas_hipblas_compute_type data)  { return convert_to_hip(&data); }
hipblasComputeType_t mblas_hipblas_compute_type::convert_to_hip(const mblas_hipblas_compute_type *data) {
  try {
    return compute_mappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to Hip Compute Type " << data->to_string() << std::endl;
    throw e;
    return HIPBLAS_COMPUTE_32F;
  }
}

// void mblas_hipblas_compute_type::operator = (const hipblasComputeType_t cudt) {
//   for (auto ele : compute_mappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblas_hipblas_compute_type & mblas_hipblas_compute_type::operator = (const mblas_hipblas_compute_type& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}

mblas_hipblas_compute_type::operator hipblasComputeType_t() const {
  return convert_to_hip(this);
}
