#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mblas_operation.h"

class mblas_cuda_operation: public mblas_operation {
 private:
  static const std::map<mblas_operation, cublasOperation_t> prec_mappings;
 public:
  static cublasOperation_t convert_to_cuda(mblas_cuda_operation data);
  static cublasOperation_t convert_to_cuda(const mblas_cuda_operation *data);
  cublasOperation_t convert_to_cuda();
  //void operator = (const cublasOperation_t cudt);
  //mblas_cuda_operation& operator = (const mblas_cuda_operation mdt);
  //mblas_cuda_operation & operator = (const mblas_cuda_operation mdt);
  mblas_cuda_operation & operator = (const mblas_cuda_operation& mdt);
  // mblas_cuda_operation & operator = (const mblas_operation& mdt);

  //operator cublasOperation_t() const;
  mblas_cuda_operation(const std::string & instr) : mblas_operation(instr) {}
  mblas_cuda_operation() : mblas_operation() {}
  mblas_cuda_operation(mblas_operation_enum y) : mblas_operation(y) {}

  std::string to_string() const override { return mblas_operation::to_string("CUBLAS"); }
};
