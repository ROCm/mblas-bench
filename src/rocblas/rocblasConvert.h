#pragma once

#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "mblasDataType.h"

__global__ void float_to_bf16(float *input, size_t num_elements,
                                hip_bfloat16 *output);

__global__ void float_to_fp16(float *input, size_t num_elements, __half *output);

__global__ void float_to_fp8(float *input, size_t num_elements,
                           __hip_fp8_storage_t *output, __hip_fp8_interpretation_t interp);
void copy_and_convert(mblasDataType precision, void *host_a, void *devA, int x,
                    int y, int batchsz);
//void copy_and_convert(hipblasDatatype_t precision, void *host_a, void *devA, int x,
//                    int y, int batchsz);
void *convert_scalar(mblasDataType precision, void *scalar);
//void *convert_scalar(hipblasDatatype_t precision, void *scalar);
void copyAndConvertScalar(mblasDataType scalarPrecision, void *hostScalar,
                          void *devScalar);
//void copyAndConvertScalar(hipblasDatatype_t scalarPrecision, void *hostScalar,
//                          void *devScalar);
                          