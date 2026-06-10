#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mblas_compute_type.h"

class mblas_cuda_compute_type: public mblas_compute_type {
 private:
  static const std::map<mblas_compute_type_enum, cublasComputeType_t> compute_mappings;
 public:
  static cublasComputeType_t convert_to_cuda(mblas_cuda_compute_type data);
  static cublasComputeType_t convert_to_cuda(const mblas_cuda_compute_type *data);
  // void operator = (const cublasComputeType_t cudt);
  //mblas_cuda_compute_type& operator = (const mblas_cuda_compute_type mdt);
  //mblas_cuda_compute_type & operator = (const mblas_cuda_compute_type mdt);
  mblas_cuda_compute_type & operator = (const mblas_cuda_compute_type& mdt);
  // mblas_cuda_compute_type & operator = (const mblas_compute_type& mdt);
  operator cublasComputeType_t() const;
  mblas_cuda_compute_type(const std::string & instr) : mblas_compute_type(instr) {}
  mblas_cuda_compute_type() : mblas_compute_type() {}
  mblas_cuda_compute_type(mblas_compute_type_enum y) : mblas_compute_type(y) {}
  //mblas_cuda_compute_type(mblas_compute_type_enum& y) : mblas_compute_type(y) {}

  std::string to_string() const override { return mblas_compute_type::to_string("CUBLAS"); }
};
