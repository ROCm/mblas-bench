#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#if (CUDART_VERSION >= 12080)
#include <cuda_fp4.h>
#endif
#include <cuda_runtime.h>
#include "mblasCuDataType.h"

__global__ void float_to_bf16(float *input, size_t num_elements,
                                __nv_bfloat16 *output);

__global__ void float_to_fp16(float *input, size_t num_elements, __half *output);

void copy_and_convert(mblasCuDataType precision, void *hostA, void *devA, long x,
                    long y, int batchsz);
void * convert_scalar(mblasCuDataType precision, void *scalar);
void copy_and_convert_scalar(mblasCuDataType scalarPrecision, void *hostScalar,
                          void *devScalar);

__global__ void float_to_fp8(float *input, size_t num_elements,
                           __nv_fp8_storage_t *output, __nv_fp8_interpretation_t interp);


#if (CUDART_VERSION >= 12080)
__global__ void float_to_fp4(float2 *input, size_t num_elements,
                           __nv_fp4x2_storage_t *output);
#endif
