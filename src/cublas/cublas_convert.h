#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#if (ENABLE_CUDA_FP4)
#include <cuda_fp4.h>
#endif
#include "mblas_cuda_data_type.h"

__global__ void float_to_bf16(float *input, size_t num_elements,
                                __nv_bfloat16 *output);

__global__ void float_to_fp16(float *input, size_t num_elements, __half *output);

void copy_and_convert(mblas_cuda_data_type precision, void *host_a, void *devA, long x,
                    long y, int batchsz, long long stride);
// void copy_and_convert(mblas_cuda_data_type precision, void *host_a, void *devA, long x,
//                             long y, int batchsz, long long stride);
void * convert_scalar(mblas_cuda_data_type precision, void *scalar);
void copy_and_convert_scalar(mblas_cuda_data_type scalarPrecision, void *hostScalar,
                          void *devScalar);

__global__ void float_to_fp8(float *input, size_t num_elements,
                           __nv_fp8_storage_t *output, __nv_fp8_interpretation_t interp);


#if (ENABLE_CUDA_FP4)
__global__ void float_to_fp4(float2 *input, size_t num_elements,
                           __nv_fp4x2_storage_t *output);
#endif
