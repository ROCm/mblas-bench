#pragma once

#include <rocblas/internal/rocblas-types.h>
#include "mblasDataType.h"

class mblasRocDataType: public mblasDataType {
 private:
  static const std::map<mblasDataType, rocblas_datatype> prec_mappings;
 public:
  // Constructors
  mblasRocDataType(const std::string & instr) : mblasDataType(instr) {}
  mblasRocDataType() : mblasDataType() {}
  mblasRocDataType(mblasDataTypeEnum y) : mblasDataType(y) {}
  // Conversions
  static rocblas_datatype convertToHip(mblasRocDataType data);
  static rocblas_datatype convertToHip(const mblasRocDataType *data);
  operator rocblas_datatype() const;
  //void operator = (const hipDataType cudt);
  //mblasRocDataType& operator = (const mblasRocDataType mdt);
  //mblasRocDataType & operator = (const mblasRocDataType mdt);
  // mblasRocDataType & operator = (const mblasDataType& mdt);
  mblasRocDataType & operator = (const mblasRocDataType& mdt);

  std::string toString() const override { return mblasDataType::toString("rocblas"); }
};
