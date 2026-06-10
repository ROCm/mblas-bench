#pragma once

#include <hipblas/hipblas.h>
#include "mblas_operation.h"

class mblas_hipblas_operation: public mblas_operation {
 private:
  static const std::map<mblas_operation, hipblasOperation_t> prec_mappings;
 public:
  static hipblasOperation_t convert_to_hip(mblas_hipblas_operation data);
  static hipblasOperation_t convert_to_hip(const mblas_hipblas_operation *data);
  hipblasOperation_t convert_to_hip();
  //void operator = (const hipblasOperation_t cudt);
  //mblas_hipblas_operation& operator = (const mblas_hipblas_operation mdt);
  //mblas_hipblas_operation & operator = (const mblas_hipblas_operation mdt);
  mblas_hipblas_operation & operator = (const mblas_hipblas_operation& mdt);
  // mblas_hipblas_operation & operator = (const mblas_operation& mdt);

  //operator hipblasOperation_t() const;
  mblas_hipblas_operation(const std::string & instr) : mblas_operation(instr) {}
  mblas_hipblas_operation() : mblas_operation() {}
  mblas_hipblas_operation(mblas_operation_enum y) : mblas_operation(y) {}

  std::string to_string() const override { return mblas_operation::to_string("HIPBLAS"); }
};
