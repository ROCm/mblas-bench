#include "rocblasConvert.h"

#include <bitset>
// #include <cuda_fp8.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "rocblasCreateAllocate.h"
#include "rocblasError.h"

__global__ void floatToBfloat16(float *input, size_t num_elements,
                                hip_bfloat16 *output)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = (hip_bfloat16) input[idx];
  }
}

__global__ void floatToFp16(float *input, size_t num_elements, __half *output)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = __float2half(input[idx]);
  }
}

// __global__ void floatToFp8(float *input, size_t num_elements,
//                            __nv_fp8_storage_t *output, __nv_fp8_interpretation_t interp)
// {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < num_elements)
//   {
//     output[idx] = __nv_cvt_float_to_fp8(input[idx], __NV_SATFINITE, interp);
//   }
// }

// __global__ void intToInt8(__int32_t *input, size_t num_elements,
//                           __int8_t *output, __nv_fp8_interpretation_t interp)
// {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < num_elements)
//   {
//     output[idx] = (__int8_t)input[idx];
//     // output[idx] = dynamic_cast<__int8_t>(input[idx]);
//   }
// }

void copyAndConvert(mblasDataType precision, void *hostA, void *devA, int x, int y, int batchsz)
{

  int hostsz = typeCallHost<sizeofCUDT>(precision);
  int devsz = typeCallDev<sizeofCUDT>(precision);
  if (precision == mblasDataType(MBLAS_C_16F) || precision == mblasDataType(MBLAS_R_16F))
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA = allocateHDevArr(precision, x, y, batchsz);
    checkHip(hipMemcpy(tmpA, hostA, batchsz * x * y * hostsz,
                       hipMemcpyHostToDevice));
    long num_elements = batchsz * x * y;
    long block_size = 256;
    long num_blocks = (num_elements + block_size - 1) / block_size;
    floatToFp16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__half *)devA);
    hipFree(tmpA);
  }
  else if (precision == mblasDataType(MBLAS_C_16BF) || precision == mblasDataType(MBLAS_R_16BF))
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA = allocateHDevArr(precision, x, y, batchsz);
    checkHip(hipMemcpy(tmpA, hostA, batchsz * x * y * hostsz,
                       hipMemcpyHostToDevice));
    long num_elements = batchsz * x * y;
    long block_size = 256;
    long num_blocks = (num_elements + block_size - 1) / block_size;
    floatToBfloat16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (hip_bfloat16 *)devA);
    hipFree(tmpA);
  } else {
    checkHip(hipMemcpy(devA, hostA, (long)batchsz * x * y * hostsz,
                       hipMemcpyHostToDevice));
  }
}

void copyAndConvert(hipblasDatatype_t precision, void *hostA, void *devA, int x, int y, int batchsz)
{

  int hostsz = typeCallHost<sizeofCUDT>(precision);
  int devsz = typeCallDev<sizeofCUDT>(precision);
  if (precision == HIPBLAS_R_16F)
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA = allocateHDevArr(precision, x, y, batchsz);
    checkHip(hipMemcpy(tmpA, hostA, batchsz * x * y * hostsz,
                       hipMemcpyHostToDevice));
    long num_elements = batchsz * x * y;
    long block_size = 256;
    long num_blocks = (num_elements + block_size - 1) / block_size;
    floatToFp16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__half *)devA);
    hipFree(tmpA);
  }
  else if (precision == HIPBLAS_R_16B)
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA = allocateHDevArr(precision, x, y, batchsz);
    checkHip(hipMemcpy(tmpA, hostA, batchsz * x * y * hostsz,
                       hipMemcpyHostToDevice));
    long num_elements = batchsz * x * y;
    long block_size = 256;
    long num_blocks = (num_elements + block_size - 1) / block_size;
    floatToBfloat16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (hip_bfloat16 *)devA);
    hipFree(tmpA);
  } else {
    checkHip(hipMemcpy(devA, hostA, (long)batchsz * x * y * hostsz,
                       hipMemcpyHostToDevice));
  }
}

void *convertScalar(mblasDataType precision, void *scalar)
{

  if (precision == mblasDataType(MBLAS_R_16F))
  {
    float scalarVal = *static_cast<float *>(scalar);
    free(scalar);
    __half *hscalar = (__half *)malloc(sizeof(__half));
    *hscalar = __float2half(scalarVal);
    return (void *)hscalar;
  }
  else if (precision == mblasDataType(MBLAS_C_16F))
  {
    // Implement me...
    return NULL;
  }
  else
  {
    return scalar;
  }
}

void *convertScalar(hipblasDatatype_t precision, void *scalar)
{

  if (precision == HIPBLAS_R_16F)
  {
    float scalarVal = *static_cast<float *>(scalar);
    free(scalar);
    __half *hscalar = (__half *)malloc(sizeof(__half));
    *hscalar = __float2half(scalarVal);
    return (void *)hscalar;
  } else {
    return scalar;
  }
}