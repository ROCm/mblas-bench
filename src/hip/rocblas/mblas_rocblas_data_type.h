#pragma once

#include <rocblas/internal/rocblas-types.h>
#include "mblas_data_type.h"

class mblas_rocblas_data_type: public mblas_data_type {
 private:
  static const std::map<mblas_data_type, rocblas_datatype> prec_mappings;
 public:
  // Constructors
  mblas_rocblas_data_type(const std::string & instr) : mblas_data_type(instr) {}
  mblas_rocblas_data_type() : mblas_data_type() {}
  mblas_rocblas_data_type(mblas_data_type_enum y) : mblas_data_type(y) {}
  // Conversions
  static rocblas_datatype convert_to_hip(mblas_rocblas_data_type data);
  static rocblas_datatype convert_to_hip(const mblas_rocblas_data_type *data);
  operator rocblas_datatype() const;
  //void operator = (const hipDataType cudt);
  //mblas_rocblas_data_type& operator = (const mblas_rocblas_data_type mdt);
  //mblas_rocblas_data_type & operator = (const mblas_rocblas_data_type mdt);
  // mblas_rocblas_data_type & operator = (const mblas_data_type& mdt);
  mblas_rocblas_data_type & operator = (const mblas_rocblas_data_type& mdt);

  std::string to_string() const override { return mblas_data_type::to_string("rocblas"); }
};
