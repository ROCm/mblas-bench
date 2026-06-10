#pragma once

#include <rocblas/internal/rocblas-types.h>
#include "mblas_operation.h"

class mblas_rocblas_operation: public mblas_operation {
 private:
  static const std::map<mblas_operation, rocblas_operation> prec_mappings;
 public:
  // Constructors
  static rocblas_operation convert_to_rocm(mblas_rocblas_operation data);
  static rocblas_operation convert_to_rocm(const mblas_rocblas_operation *data);
  rocblas_operation convert_to_rocm();
  //void operator = (const rocblas_operation cudt);
  //mblas_rocblas_operation& operator = (const mblas_rocblas_operation mdt);
  //mblas_rocblas_operation & operator = (const mblas_rocblas_operation mdt);
  mblas_rocblas_operation & operator = (const mblas_rocblas_operation& mdt);
  // mblas_rocblas_operation & operator = (const mblas_operation& mdt);

  //operator rocblas_operation() const;
  mblas_rocblas_operation(const std::string & instr) : mblas_operation(instr) {}
  mblas_rocblas_operation() : mblas_operation() {}
  mblas_rocblas_operation(mblas_operation_enum y) : mblas_operation(y) {}

  std::string to_string() const override { return mblas_operation::to_string("rocblas"); }
};
