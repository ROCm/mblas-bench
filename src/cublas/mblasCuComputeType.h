#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mblasComputeType.h"

class mblasCuComputeType: public mblasComputeType {
 private:
  static const std::map<mblasComputeTypeEnum, cublasComputeType_t> compute_mappings;
 public:
  static cublasComputeType_t convert_to_cuda(mblasCuComputeType data);
  static cublasComputeType_t convert_to_cuda(const mblasCuComputeType *data);
  // void operator = (const cublasComputeType_t cudt);
  //mblasCuComputeType& operator = (const mblasCuComputeType mdt);
  //mblasCuComputeType & operator = (const mblasCuComputeType mdt);
  mblasCuComputeType & operator = (const mblasCuComputeType& mdt);
  // mblasCuComputeType & operator = (const mblasComputeType& mdt);
  operator cublasComputeType_t() const;
  mblasCuComputeType(const std::string & instr) : mblasComputeType(instr) {}
  mblasCuComputeType() : mblasComputeType() {}
  mblasCuComputeType(mblasComputeTypeEnum y) : mblasComputeType(y) {}
  //mblasCuComputeType(mblasComputeTypeEnum& y) : mblasComputeType(y) {}

  std::string to_string() const override { return mblasComputeType::to_string("CUBLAS"); }
};
