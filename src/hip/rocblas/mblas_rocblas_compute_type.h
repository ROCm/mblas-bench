#pragma once

#include <vector>
#include <rocblas/internal/rocblas-types.h>
#include "mblas_compute_type.h"

class mblas_rocblas_compute_type: public mblas_compute_type {
 private:
  static const std::vector<std::pair<mblas_compute_type_enum, mblas_data_type_enum>> prec_mappings;
  bool roc_is_real;
 public:
  // Constructors
  mblas_rocblas_compute_type(const std::string & instr) : mblas_compute_type(instr) {}
  mblas_rocblas_compute_type() : mblas_compute_type() {}
  mblas_rocblas_compute_type(mblas_compute_type_enum y) : mblas_compute_type(y) {}
  // Conversions
  static rocblas_datatype convert_to_rocm_data(const mblas_rocblas_compute_type *data);
  operator rocblas_datatype() const;
  //mblas_rocblas_compute_type& operator = (const mblas_rocblas_compute_type mdt);
  //mblas_rocblas_compute_type & operator = (const mblas_rocblas_compute_type mdt);
  mblas_rocblas_compute_type & operator = (const mblas_rocblas_compute_type& mdt);
  // mblas_rocblas_compute_type & operator = (const mblas_compute_type& mdt);
  //mblas_rocblas_compute_type(mblas_compute_type_enum& y) : mblas_compute_type(y) {}

  std::string to_string() const override { return mblas_compute_type::to_string("rocblas"); }
  void set_compute(std::string computestr, mblas_data_type& precision) override;
};
