#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mblasOperation.h"

class mblasCuOperation: public mblasOperation {
 private:
  static const std::map<mblasOperation, cublasOperation_t> precMappings;
 public:
  static cublasOperation_t convertToCuda(mblasCuOperation data);
  static cublasOperation_t convertToCuda(const mblasCuOperation *data);
  cublasOperation_t convertToCuda();
  //void operator = (const cublasOperation_t cudt);
  //mblasCuOperation& operator = (const mblasCuOperation mdt);
  //mblasCuOperation & operator = (const mblasCuOperation mdt);
  mblasCuOperation & operator = (const mblasCuOperation& mdt);
  // mblasCuOperation & operator = (const mblasOperation& mdt);

  //operator cublasOperation_t() const;
  mblasCuOperation(const std::string & instr) : mblasOperation(instr) {}
  mblasCuOperation() : mblasOperation() {}
  mblasCuOperation(mblasOperationEnum y) : mblasOperation(y) {}

  std::string toString() const override { return mblasOperation::toString("CUBLAS"); }
};
