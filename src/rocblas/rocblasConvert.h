#pragma once

#include <rocblas/rocblas.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__global__ void floatToBfloat16(float *input, size_t num_elements,
                                hip_bfloat16 *output);

__global__ void floatToFp16(float *input, size_t num_elements, __half *output);

void copyAndConvert(rocblas_datatype precision, void *hostA, void *devA, int x,
                    int y, int batchsz, int blockct);
void *convertScalar(rocblas_datatype precision, void *scalar);
void copyAndConvertScalar(rocblas_datatype scalarPrecision, void *hostScalar,
                          void *devScalar);
