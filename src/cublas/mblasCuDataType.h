#pragma once

#include <cuda_runtime.h>
//#include <cublas.h>
#include <cublasLt.h>
#include "mblasDataType.h"

class mblasCuDataType: public mblasDataType {
 private:
  static const std::map<mblasDataType, cudaDataType> prec_mappings;
 public:
  static cudaDataType convert_to_cuda(mblasCuDataType data);
  static cudaDataType convert_to_cuda(const mblasCuDataType *data);
  //void operator = (const cudaDataType cudt);
  //mblasCuDataType& operator = (const mblasCuDataType mdt);
  //mblasCuDataType & operator = (const mblasCuDataType mdt);
  mblasCuDataType & operator = (const mblasCuDataType& mdt);
  // mblasCuDataType & operator = (const mblasDataType& mdt);
  operator cudaDataType() const;
  mblasCuDataType(const std::string & instr) : mblasDataType(instr) {}
  mblasCuDataType() : mblasDataType() {}
  mblasCuDataType(mblasDataTypeEnum y) : mblasDataType(y) {}

  std::string to_string() const override { return mblasDataType::to_string("CUDA"); }
  mblasCuDataType get_scale_type();

#if (CUDART_VERSION >= 12080)
  cublasLtMatmulMatrixScale_t get_scale_mode();
#endif
};
