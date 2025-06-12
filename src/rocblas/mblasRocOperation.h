#pragma once

#include <rocblas/internal/rocblas-types.h>
#include "mblasOperation.h"

class mblasRocOperation: public mblasOperation {
 private:
  static const std::map<mblasOperation, rocblas_operation> prec_mappings;
 public:
  // Constructors
  static rocblas_operation convertToRocm(mblasRocOperation data);
  static rocblas_operation convertToRocm(const mblasRocOperation *data);
  rocblas_operation convertToRocm();
  //void operator = (const rocblas_operation cudt);
  //mblasRocOperation& operator = (const mblasRocOperation mdt);
  //mblasRocOperation & operator = (const mblasRocOperation mdt);
  mblasRocOperation & operator = (const mblasRocOperation& mdt);
  // mblasRocOperation & operator = (const mblasOperation& mdt);

  //operator rocblas_operation() const;
  mblasRocOperation(const std::string & instr) : mblasOperation(instr) {}
  mblasRocOperation() : mblasOperation() {}
  mblasRocOperation(mblasOperationEnum y) : mblasOperation(y) {}

  std::string toString() const override { return mblasOperation::toString("rocblas"); }
};
