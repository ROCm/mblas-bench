#pragma once

#include <vector>
#include <rocblas/internal/rocblas-types.h>
#include "mblasComputeType.h"

class mblasRocComputeType: public mblasComputeType {
 private:
  static const std::map<mblasComputeTypeEnum, rocblas_computetype> compMappings;
  //static const std::vector<std::pair<mblasComputeTypeEnum, mblasDataTypeEnum>> precMappings;
  bool rocIsReal;
 public:
  // Constructors
  mblasRocComputeType(const std::string & instr) : mblasComputeType(instr) {}
  mblasRocComputeType() : mblasComputeType() {}
  mblasRocComputeType(mblasComputeTypeEnum y) : mblasComputeType(y) {}
  // Conversions
  static rocblas_computetype convertToRocm(mblasRocComputeType data);
  static rocblas_computetype convertToRocm(const mblasRocComputeType *data);
  operator rocblas_computetype() const;
  static rocblas_datatype convertToRocmData(const mblasRocComputeType *data);
  operator rocblas_datatype() const;
  // void operator = (const rocblas_computetype cudt);
  //mblasRocComputeType& operator = (const mblasRocComputeType mdt);
  //mblasRocComputeType & operator = (const mblasRocComputeType mdt);
  mblasRocComputeType & operator = (const mblasRocComputeType& mdt);
  // mblasRocComputeType & operator = (const mblasComputeType& mdt);
  //mblasRocComputeType(mblasComputeTypeEnum& y) : mblasComputeType(y) {}

  std::string toString() const override { return mblasComputeType::toString("rocblas"); }
  void setCompute(std::string computestr, mblasDataType& precision) override;
};
