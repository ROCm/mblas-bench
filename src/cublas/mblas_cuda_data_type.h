#pragma once

#include <cuda_runtime.h>
//#include <cublas.h>
#include <cublasLt.h>
#include "mblas_data_type.h"

class mblas_cuda_data_type: public mblas_data_type {
 private:
  static const std::map<mblas_data_type, cudaDataType> prec_mappings;
 public:
  static cudaDataType convert_to_cuda(mblas_cuda_data_type data);
  static cudaDataType convert_to_cuda(const mblas_cuda_data_type *data);
  //void operator = (const cudaDataType cudt);
  //mblas_cuda_data_type& operator = (const mblas_cuda_data_type mdt);
  //mblas_cuda_data_type & operator = (const mblas_cuda_data_type mdt);
  mblas_cuda_data_type & operator = (const mblas_cuda_data_type& mdt);
  // mblas_cuda_data_type & operator = (const mblas_data_type& mdt);
  operator cudaDataType() const;
  mblas_cuda_data_type(const std::string & instr) : mblas_data_type(instr) {}
  mblas_cuda_data_type() : mblas_data_type() {}
  mblas_cuda_data_type(mblas_data_type_enum y) : mblas_data_type(y) {}

  std::string to_string() const override { return mblas_data_type::to_string("CUDA"); }
  mblas_cuda_data_type get_scale_type();

#if (ENABLE_CUDA_FP4)
  cublasLtMatmulMatrixScale_t get_scale_mode();
#endif
};
