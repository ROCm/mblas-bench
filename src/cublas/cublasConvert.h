#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void floatToBfloat16(float *input, size_t num_elements,
                                __nv_bfloat16 *output);

__global__ void floatToFp16(float *input, size_t num_elements, __half *output);

void copyAndConvert(cublasDataType_t precision, void *hostA, void *devA, int x,
                    int y, int batchsz);
void *convertScalar(cublasDataType_t precision, void *scalar);
void copyAndConvertScalar(cublasDataType_t scalarPrecision, void *hostScalar,
                          void *devScalar);
