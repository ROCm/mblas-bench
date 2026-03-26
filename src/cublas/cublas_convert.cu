#include "cublas_convert.h"

#include <bitset>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#if (ENABLE_CUDA_FP4)
#include <cuda_fp4.h>
#endif
#include "cublas_create_allocate.h"
#include "cuda_error.h"
#include "mblas_cuda_data_type.h"
#include "generic_setup.h"

__global__ void float_to_bf16(float *input, size_t num_elements,
                                __nv_bfloat16 *output)
{
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = __float2bfloat16(input[idx]);
  }
}

__global__ void float_to_fp16(float *input, size_t num_elements, __half *output)
{
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = __float2half(input[idx]);
  }
}

__global__ void float_to_fp8(float *input, size_t num_elements,
                           __nv_fp8_storage_t *output, __nv_fp8_interpretation_t interp)
{
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = __nv_cvt_float_to_fp8(input[idx], __NV_SATFINITE, interp);
  }
}

__global__ void float_to_e8m0(float *input, size_t num_elements,
                           __nv_fp8_storage_t *output)
{
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    /*
     Rounding only controls the direction of rounding
     https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html
    */
    output[idx] = __nv_cvt_float_to_e8m0(input[idx], __NV_SATFINITE, cudaRoundZero);
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

#if (ENABLE_CUDA_FP4)
__global__ void float_to_fp4(float2 *input, size_t num_elements,
                           __nv_fp4x2_storage_t *output)
{
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = __nv_cvt_float2_to_fp4x2(input[idx], __NV_E2M1, cudaRoundNearest);
  }
}
#endif

void copy_and_convert(mblas_cuda_data_type precision, void *host_a, void *devA, long x,
                      long y, int batchsz, long long stride)
{
  if (batchsz * x * y == 0)
  {
    // Matrix not used, don't copy
    return;
  }
  long hostsz = type_call_host<sizeofCUDT>(precision);
  long devsz = type_call_dev<sizeofCUDT>(precision);
  long long base = x * y;
  long long total_elements =
      (batchsz > 1 && stride > base) ? (stride * (batchsz - 1) + base)
                                     : (base * batchsz);
  if (precision == mblas_data_type::MBLAS_C_16F || precision == mblas_data_type::MBLAS_R_16F)
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA;
    check_cuda(cudaMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_cuda(cudaMemcpy(tmpA, host_a, total_elements * hostsz, cudaMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size = 256;
    long long num_blocks = (num_elements + block_size - 1) / block_size;
    float_to_fp16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__half *)devA);
    check_cuda(cudaGetLastError());
    cudaFree(tmpA);
  }
  else if (precision == mblas_data_type::MBLAS_C_16BF || precision == mblas_data_type::MBLAS_R_16BF)
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA;
    check_cuda(cudaMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_cuda(cudaMemcpy(tmpA, host_a, total_elements * hostsz, cudaMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size = 256;
    long long num_blocks = (num_elements + block_size - 1) / block_size;
    float_to_bf16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__nv_bfloat16 *)devA);
    check_cuda(cudaGetLastError());
    cudaFree(tmpA);
  }
  else if (precision == mblas_data_type::MBLAS_R_8F_E4M3 ||
           precision == mblas_data_type::MBLAS_R_8F_E5M2 ||
           precision == mblas_data_type::MBLAS_R_8F_UE4M3)
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA;
    check_cuda(cudaMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_cuda(cudaMemcpy(tmpA, host_a, total_elements * hostsz, cudaMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size = 256;
    long long num_blocks = (num_elements + block_size - 1) / block_size;
    __nv_fp8_interpretation_t interp;
    if (precision == mblas_data_type::MBLAS_R_8F_E4M3)
    {
      interp = __NV_E4M3;
    }
    else if (precision == mblas_data_type::MBLAS_R_8F_UE4M3)
    {
      interp = __NV_E4M3;
    }
    else if (precision == mblas_data_type::MBLAS_R_8F_E5M2)
    {
      interp = __NV_E5M2;
    }
    float_to_fp8<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__nv_fp8_storage_t *)devA, interp);
    check_cuda(cudaGetLastError());
    cudaFree(tmpA);
  }
  else if (precision == mblas_data_type::MBLAS_R_8F_UE8M0)
  {
#if (ENABLE_CUDA_FP4)
    // Allocate memory in the device for host precision (float)
    void *tmpA;
    check_cuda(cudaMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_cuda(cudaMemcpy(tmpA, host_a, total_elements * hostsz, cudaMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size = 256;
    long long num_blocks = (num_elements + block_size - 1) / block_size;
    float_to_e8m0<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__nv_fp8_storage_t *)devA);
    check_cuda(cudaGetLastError());
    cudaFree(tmpA);
#endif
  }
  else if (precision == mblas_data_type::MBLAS_R_4F_E2M1)
  {
#if (ENABLE_CUDA_FP4)
    // Allocate memory in the device for host precision (float)
    void *tmpA;
    check_cuda(cudaMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_cuda(cudaMemcpy(tmpA, host_a, total_elements * hostsz, cudaMemcpyHostToDevice));
    long long num_elements = ceil_division(total_elements, 2l);
    long long block_size = 256;
    long long num_blocks = (num_elements + block_size - 1) / block_size;
    float_to_fp4<<<num_blocks, block_size>>>((float2 *)tmpA, num_elements, (__nv_fp4x2_storage_t *)devA);
    check_cuda(cudaGetLastError());
    cudaFree(tmpA);
#endif
  }
  // else if (precision == mblas_data_type::MBLAS_C_8I || precision == mblas_data_type::MBLAS_R_8I)
  //{
  //   // Allocate memory in the device for host precision (float)
  //   void *tmpA = allocate_host_dev_array(precision, x, y, batchsz);
  //   check_cuda(cudaMemcpy(tmpA, host_a, batchsz * x * y * hostsz,
  //                        cudaMemcpyHostToDevice));
  //   int num_elements = batchsz * x * y;
  //   int block_size = 256;
  //   int num_blocks = (num_elements + block_size - 1) / block_size;
  //   floatToBfloat16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__nv_bfloat16 *)devA);
  //   cudaFree(tmpA);
  // }
  else
  {
    check_cuda(cudaMemcpy(
        devA, host_a,
        static_cast<size_t>(get_malloc_size_host(precision, x, y, batchsz, stride)),
        cudaMemcpyHostToDevice));
  }
}

void *convert_scalar(mblas_cuda_data_type precision, void *scalar)
{
  if (precision == mblas_data_type::MBLAS_R_16F)
  {
    // Read float value, convert to __half, write in-place
    float scalarVal = *static_cast<float *>(scalar);
    __half *hscalar = (__half *)scalar;
    *hscalar = __float2half(scalarVal);
    return scalar;
  }
  else if (precision == mblas_data_type::MBLAS_C_16F)
  {
    // Read complex<float>, convert to complex<__half>, write in-place
    std::complex<float> *cFloat = static_cast<std::complex<float> *>(scalar);
    float realVal = cFloat->real();
    float imagVal = cFloat->imag();
    
    std::complex<__half> *cHalf = (std::complex<__half> *)scalar;
    *cHalf = std::complex<__half>(__float2half(realVal), __float2half(imagVal));
    return scalar;
  }
  else
  {
    return scalar;
  }
}
