#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mblasOperation.h"

class mblasCuOperation: public mblasOperation {
 private:
  static const std::map<mblasOperation, cublasOperation_t> prec_mappings;
 public:
  static cublasOperation_t convert_to_cuda(mblasCuOperation data);
  static cublasOperation_t convert_to_cuda(const mblasCuOperation *data);
  cublasOperation_t convert_to_cuda();
  //void operator = (const cublasOperation_t cudt);
  //mblasCuOperation& operator = (const mblasCuOperation mdt);
  //mblasCuOperation & operator = (const mblasCuOperation mdt);
  mblasCuOperation & operator = (const mblasCuOperation& mdt);
  // mblasCuOperation & operator = (const mblasOperation& mdt);

  //operator cublasOperation_t() const;
  mblasCuOperation(const std::string & instr) : mblasOperation(instr) {}
  mblasCuOperation() : mblasOperation() {}
  mblasCuOperation(mblasOperationEnum y) : mblasOperation(y) {}

  std::string to_string() const override { return mblasOperation::to_string("CUBLAS"); }
};
