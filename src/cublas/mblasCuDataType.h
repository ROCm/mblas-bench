#pragma once

#include <cuda_runtime.h>
//#include <cublas.h>
#include <cublasLt.h>
#include "mblasDataType.h"

class mblasCuDataType: public mblasDataType {
 private:
  static const std::map<mblasDataType, cudaDataType> precMappings;
 public:
  static cudaDataType convertToCuda(mblasCuDataType data);
  static cudaDataType convertToCuda(const mblasCuDataType *data);
  //void operator = (const cudaDataType cudt);
  //mblasCuDataType& operator = (const mblasCuDataType mdt);
  //mblasCuDataType & operator = (const mblasCuDataType mdt);
  mblasCuDataType & operator = (const mblasCuDataType& mdt);
  // mblasCuDataType & operator = (const mblasDataType& mdt);
  operator cudaDataType() const;
  mblasCuDataType(const std::string & instr) : mblasDataType(instr) {}
  mblasCuDataType() : mblasDataType() {}
  mblasCuDataType(mblasDataTypeEnum y) : mblasDataType(y) {}

  std::string toString() const override { return mblasDataType::toString("CUDA"); }
  mblasCuDataType get_scale_type();
  cublasLtMatmulMatrixScale_t get_scale_mode();
};
