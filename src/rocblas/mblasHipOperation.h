#pragma once

#include <hipblas/hipblas.h>
#include "mblasOperation.h"

class mblasHipOperation: public mblasOperation {
 private:
  static const std::map<mblasOperation, hipblasOperation_t> precMappings;
 public:
  static hipblasOperation_t convertToHip(mblasHipOperation data);
  static hipblasOperation_t convertToHip(const mblasHipOperation *data);
  hipblasOperation_t convertToHip();
  //void operator = (const hipblasOperation_t cudt);
  //mblasHipOperation& operator = (const mblasHipOperation mdt);
  //mblasHipOperation & operator = (const mblasHipOperation mdt);
  mblasHipOperation & operator = (const mblasHipOperation& mdt);
  // mblasHipOperation & operator = (const mblasOperation& mdt);

  //operator hipblasOperation_t() const;
  mblasHipOperation(const std::string & instr) : mblasOperation(instr) {}
  mblasHipOperation() : mblasOperation() {}
  mblasHipOperation(mblasOperationEnum y) : mblasOperation(y) {}

  std::string toString() const override { return mblasOperation::toString("HIPBLAS"); }
};
