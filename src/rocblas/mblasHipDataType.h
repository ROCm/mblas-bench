#pragma once

#include <hip/library_types.h>
#include "mblasDataType.h"

class mblasHipDataType: public mblasDataType {
 private:
  static const std::map<mblasDataType, hipDataType> prec_mappings;
 public:
  // Constructors
  mblasHipDataType(const std::string & instr) : mblasDataType(instr) {}
  mblasHipDataType() : mblasDataType() {}
  mblasHipDataType(mblasDataTypeEnum y) : mblasDataType(y) {}
  // Conversions
  static hipDataType convertToHip(mblasHipDataType data);
  static hipDataType convertToHip(const mblasHipDataType *data);
  operator hipDataType() const;
  //void operator = (const hipDataType cudt);
  //mblasHipDataType& operator = (const mblasHipDataType mdt);
  //mblasHipDataType & operator = (const mblasHipDataType mdt);
  // mblasHipDataType & operator = (const mblasDataType& mdt);
  mblasHipDataType & operator = (const mblasHipDataType& mdt);

  std::string toString() const override { return mblasDataType::toString("HIP"); }
};
