#include "mblas_rocblas_compute_type.h"

#include <iostream>

#include "mblas_rocblas_data_type.h"

const std::vector<std::pair<mblas_compute_type_enum, mblas_data_type_enum>> mblas_rocblas_compute_type::prec_mappings = {
    {MBLAS_COMPUTE_64F, MBLAS_R_64F},
    {MBLAS_COMPUTE_32F, MBLAS_R_32F},
    {MBLAS_COMPUTE_16F, MBLAS_R_16F},
    {MBLAS_COMPUTE_32I, MBLAS_R_32I},
    {MBLAS_COMPUTE_64F, MBLAS_C_64F},
    {MBLAS_COMPUTE_32F, MBLAS_C_32F},
};

rocblas_datatype mblas_rocblas_compute_type::convert_to_rocm_data(const mblas_rocblas_compute_type *data) {
  for (auto ele : prec_mappings) {
    mblas_rocblas_data_type rdata = mblas_rocblas_data_type(ele.second);
    if (ele.first == *data && rdata.is_real() == data->roc_is_real) {
      return rocblas_datatype(rdata);
    }
  }
  std::cout << "Failed to convert to rocBLAS Data Type " << data->to_string() << std::endl;
  throw std::out_of_range("Value not found in list");
}

mblas_rocblas_compute_type::operator rocblas_datatype() const {
  return convert_to_rocm_data(this);
}

mblas_rocblas_compute_type & mblas_rocblas_compute_type::operator = (const mblas_rocblas_compute_type& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}

void mblas_rocblas_compute_type::set_compute(std::string computestr, mblas_data_type& precision) {
  mblas_compute_type::set_compute(computestr, precision);
  roc_is_real = precision.is_real();
}
