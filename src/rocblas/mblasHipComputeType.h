#pragma once

#include <hipblas/hipblas.h>
#include "mblasComputeType.h"

class mblasHipComputeType: public mblasComputeType {
 private:
  static const std::map<mblasComputeTypeEnum, hipblasComputeType_t> compute_mappings;
 public:
  static hipblasComputeType_t convertToHip(mblasHipComputeType data);
  static hipblasComputeType_t convertToHip(const mblasHipComputeType *data);
  // void operator = (const hipblasComputeType_t cudt);
  //mblasHipComputeType& operator = (const mblasHipComputeType mdt);
  //mblasHipComputeType & operator = (const mblasHipComputeType mdt);
  mblasHipComputeType & operator = (const mblasHipComputeType& mdt);
  // mblasHipComputeType & operator = (const mblasComputeType& mdt);
  operator hipblasComputeType_t() const;
  mblasHipComputeType(const std::string & instr) : mblasComputeType(instr) {}
  mblasHipComputeType() : mblasComputeType() {}
  mblasHipComputeType(mblasComputeTypeEnum y) : mblasComputeType(y) {}
  //mblasHipComputeType(mblasComputeTypeEnum& y) : mblasComputeType(y) {}

  std::string toString() const override { return mblasComputeType::toString("HIPBLAS"); }
};
