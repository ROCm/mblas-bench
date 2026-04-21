#pragma once

#include <hip/library_types.h>
#include "mblas_data_type.h"

class mblas_hip_data_type: public mblas_data_type {
 private:
  static const std::map<mblas_data_type, hipDataType> prec_mappings;
 public:
  // Constructors
  mblas_hip_data_type(const std::string & instr) : mblas_data_type(instr) {}
  mblas_hip_data_type() : mblas_data_type() {}
  mblas_hip_data_type(mblas_data_type_enum y) : mblas_data_type(y) {}
  // Conversions
  static hipDataType convert_to_hip(mblas_hip_data_type data);
  static hipDataType convert_to_hip(const mblas_hip_data_type *data);
  operator hipDataType() const;
  //void operator = (const hipDataType cudt);
  //mblas_hip_data_type& operator = (const mblas_hip_data_type mdt);
  //mblas_hip_data_type & operator = (const mblas_hip_data_type mdt);
  // mblas_hip_data_type & operator = (const mblas_data_type& mdt);
  mblas_hip_data_type & operator = (const mblas_hip_data_type& mdt);

  std::string to_string() const override { return mblas_data_type::to_string("HIP"); }
  
  mblas_hip_data_type get_scale_type();
};
