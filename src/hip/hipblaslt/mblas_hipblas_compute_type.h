#pragma once

#include <hipblas/hipblas.h>
#include "mblas_compute_type.h"

class mblas_hipblas_compute_type: public mblas_compute_type {
 private:
  static const std::map<mblas_compute_type_enum, hipblasComputeType_t> compute_mappings;
 public:
  static hipblasComputeType_t convert_to_hip(mblas_hipblas_compute_type data);
  static hipblasComputeType_t convert_to_hip(const mblas_hipblas_compute_type *data);
  // void operator = (const hipblasComputeType_t cudt);
  //mblas_hipblas_compute_type& operator = (const mblas_hipblas_compute_type mdt);
  //mblas_hipblas_compute_type & operator = (const mblas_hipblas_compute_type mdt);
  mblas_hipblas_compute_type & operator = (const mblas_hipblas_compute_type& mdt);
  // mblas_hipblas_compute_type & operator = (const mblas_compute_type& mdt);
  operator hipblasComputeType_t() const;
  mblas_hipblas_compute_type(const std::string & instr) : mblas_compute_type(instr) {}
  mblas_hipblas_compute_type() : mblas_compute_type() {}
  mblas_hipblas_compute_type(mblas_compute_type_enum y) : mblas_compute_type(y) {}
  //mblas_hipblas_compute_type(mblas_compute_type_enum& y) : mblas_compute_type(y) {}

  std::string to_string() const override { return mblas_compute_type::to_string("HIPBLAS"); }
};
