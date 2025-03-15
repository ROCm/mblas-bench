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

void copyAndConvert(mblasDataType precision, void *hostA, void *devA, int x,
                    int y, int batchsz);
void copyAndConvert(hipblasDatatype_t precision, void *hostA, void *devA, int x,
                    int y, int batchsz);
void *convertScalar(mblasDataType precision, void *scalar);
void *convertScalar(hipblasDatatype_t precision, void *scalar);
void copyAndConvertScalar(mblasDataType scalarPrecision, void *hostScalar,
                          void *devScalar);
void copyAndConvertScalar(hipblasDatatype_t scalarPrecision, void *hostScalar,
                          void *devScalar);
                          