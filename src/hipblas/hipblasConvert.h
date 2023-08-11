#pragma once

#include <hip/hipblas.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__global__ void floatToBfloat16(float *input, size_t num_elements,
                                hip_bfloat16 *output);

__global__ void floatToFp16(float *input, size_t num_elements, __half *output);

void copyAndConvert(hipDataType precision, void *hostA, void *devA, int x,
                    int y, int batchsz);
void *convertScalar(hipDataType precision, void *scalar);
void copyAndConvertScalar(hipDataType scalarPrecision, void *hostScalar,
                          void *devScalar);
