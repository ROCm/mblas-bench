#pragma once

#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include "mblas_data_type.h"

__global__ void float_to_bf16(float *input, size_t num_elements,
                                hip_bfloat16 *output);

__global__ void float_to_fp16(float *input, size_t num_elements, __half *output);

__global__ void float_to_fp8(float *input, size_t num_elements,
                           __hip_fp8_storage_t *output, __hip_fp8_interpretation_t interp);

#if HIP_VERSION >= 70000000
void float_to_fp6_helper(float *input, size_t num_elements, __hip_fp8_storage_t *output, mblas_fp6_interpretation_t minterp, size_t num_blocks, size_t block_size);
void float_to_fp4_helper(float2 *input, size_t num_elements, __hip_fp8_storage_t *output, size_t num_blocks, size_t block_size);
__global__ void float_to_ue8m0(float *input, size_t num_elements, __hip_fp8_storage_t *output);
#endif

void copy_and_convert(mblas_data_type precision, void *host_a, void *devA, long x,
                    long y, int batchsz);
//void copy_and_convert(hipblasDatatype_t precision, void *host_a, void *devA, int x,
//                    int y, int batchsz);
void *convert_scalar(mblas_data_type precision, void *scalar);
//void *convert_scalar(hipblasDatatype_t precision, void *scalar);
void copyAndConvertScalar(mblas_data_type scalarPrecision, void *hostScalar,
                          void *devScalar);
//void copyAndConvertScalar(hipblasDatatype_t scalarPrecision, void *hostScalar,
//                          void *devScalar);
                          
