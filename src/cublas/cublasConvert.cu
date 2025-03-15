#include "cublasConvert.h"

#include <bitset>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cublasCreateAllocate.h"
#include "cudaError.h"
#include "mblasCuDataType.h"
#include "genericSetup.h"

__global__ void float_to_bf16(float *input, size_t num_elements,
                                __nv_bfloat16 *output)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = __float2bfloat16(input[idx]);
  }
}

__global__ void float_to_fp16(float *input, size_t num_elements, __half *output)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = __float2half(input[idx]);
  }
}

__global__ void float_to_fp8(float *input, size_t num_elements,
                           __nv_fp8_storage_t *output, __nv_fp8_interpretation_t interp)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = __nv_cvt_float_to_fp8(input[idx], __NV_SATFINITE, interp);
  }
}

/*
FYI: 

cudaRoundMode
    cudaRoundNearest
    cudaRoundZero
    cudaRoundPosInf
    cudaRoundMinInf
*/
__global__ void float_to_fp4(float2 *input, size_t num_elements,
                           __nv_fp4x2_storage_t *output)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = __nv_cvt_float2_to_fp4x2(input[idx], __NV_E2M1, cudaRoundNearest);
  }
}

__global__ void intToInt8(__int32_t *input, size_t num_elements,
                          __int8_t *output, __nv_fp8_interpretation_t interp)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = (__int8_t)input[idx];
    // output[idx] = dynamic_cast<__int8_t>(input[idx]);
  }
}

void copy_and_convert(mblasCuDataType precision, void *hostA, void *devA, long x, long y, int batchsz)
{

  long hostsz = typeCallHost<sizeofCUDT>(precision);
  long devsz = typeCallDev<sizeofCUDT>(precision);
  if (precision == mblasDataType::MBLAS_C_16F || precision == mblasDataType::MBLAS_R_16F)
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA = allocateHDevArr(precision, x, y, batchsz);
    checkCuda(cudaMemcpy(tmpA, hostA, batchsz * x * y * hostsz,
                         cudaMemcpyHostToDevice));
    int num_elements = batchsz * x * y;
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    float_to_fp16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__half *)devA);
    cudaFree(tmpA);
  }
  else if (precision == mblasDataType::MBLAS_C_16BF || precision == mblasDataType::MBLAS_R_16BF)
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA = allocateHDevArr(precision, x, y, batchsz);
    checkCuda(cudaMemcpy(tmpA, hostA, batchsz * x * y * hostsz,
                         cudaMemcpyHostToDevice));
    int num_elements = batchsz * x * y;
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    float_to_bf16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__nv_bfloat16 *)devA);
    cudaFree(tmpA);
  }
  else if (precision == mblasDataType::MBLAS_R_8F_E4M3 || precision == mblasDataType::MBLAS_R_8F_E5M2 || precision == mblasDataType::MBLAS_R_8F_UE4M3)
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA = allocateHDevArr(precision, x, y, batchsz);
    checkCuda(cudaMemcpy(tmpA, hostA, batchsz * x * y * hostsz,
                         cudaMemcpyHostToDevice));
    int num_elements = batchsz * x * y;
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    __nv_fp8_interpretation_t interp;
    if (precision == mblasDataType::MBLAS_R_8F_E4M3)
    {
      interp = __NV_E4M3;
    }
    if (precision == mblasDataType::MBLAS_R_8F_UE4M3)
    {
      interp = __NV_E4M3;
    }
    else if (precision == mblasDataType::MBLAS_R_8F_E5M2)
    {
      interp = __NV_E5M2;
    }
    float_to_fp8<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__nv_fp8_storage_t *)devA, interp);
    cudaFree(tmpA);
  }
  else if (precision == mblasDataType::MBLAS_R_4F_E2M1)
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA = allocateHDevArr(precision, x, y, batchsz);
    checkCuda(cudaMemcpy(tmpA, hostA, batchsz * x * y * hostsz,
                         cudaMemcpyHostToDevice));
    long num_elements = ceil_division(batchsz * x * y, 2l);
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    float_to_fp4<<<num_blocks, block_size>>>((float2 *)tmpA, num_elements, (__nv_fp4x2_storage_t *)devA);
    cudaFree(tmpA);
  }
  // else if (precision == mblasDataType::MBLAS_C_8I || precision == mblasDataType::MBLAS_R_8I)
  //{
  //   // Allocate memory in the device for host precision (float)
  //   void *tmpA = allocateHDevArr(precision, x, y, batchsz);
  //   checkCuda(cudaMemcpy(tmpA, hostA, batchsz * x * y * hostsz,
  //                        cudaMemcpyHostToDevice));
  //   int num_elements = batchsz * x * y;
  //   int block_size = 256;
  //   int num_blocks = (num_elements + block_size - 1) / block_size;
  //   floatToBfloat16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__nv_bfloat16 *)devA);
  //   cudaFree(tmpA);
  // }
  else
  {
    checkCuda(cudaMemcpy(devA, hostA, (long)batchsz * x * y * hostsz,
                         cudaMemcpyHostToDevice));
  }
}

void *convert_scalar(mblasCuDataType precision, void *scalar)
{

  if (precision == mblasDataType::MBLAS_R_16F)
  {
    float scalarVal = *static_cast<float *>(scalar);
    free(scalar);
    __half *hscalar = (__half *)malloc(sizeof(__half));
    *hscalar = __float2half(scalarVal);
    return (void *)hscalar;
  }
  else if (precision == mblasDataType::MBLAS_C_16F)
  {
    // Implement me...
    return NULL;
  }
  else
  {
    return scalar;
  }
}