#include "hip_convert.h"

#include <bitset>
// #include <cuda_fp8.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "hip_create_allocate.h"
#include "hip_error.h"

__global__ void float_to_bf16(float *input, size_t num_elements,
                                hip_bfloat16 *output)
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
                           __hip_fp8_storage_t *output, __hip_fp8_interpretation_t interp)
{
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    output[idx] = __hip_cvt_float_to_fp8(input[idx], __HIP_SATFINITE, interp);
  }
}

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

void copy_and_convert(mblas_data_type precision, void *hostA, void *devA, long x, long y, int batchsz)
{

  long hostsz = type_call_host<sizeofCUDT>(precision);
  long devsz = type_call_dev<sizeofCUDT>(precision);
  if (precision == mblas_data_type(MBLAS_C_16F) || precision == mblas_data_type(MBLAS_R_16F))
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA;
    check_hip(hipMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz)));
    check_hip(hipMemcpy(tmpA, hostA, batchsz * x * y * hostsz,
                       hipMemcpyHostToDevice));
    long num_elements = batchsz * x * y;
    long block_size = 256;
    long num_blocks = (num_elements + block_size - 1) / block_size;
    float_to_fp16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__half *)devA);
    check_hip(hipGetLastError());
    hipFree(tmpA);
  }
  else if (precision == mblas_data_type(MBLAS_C_16BF) || precision == mblas_data_type(MBLAS_R_16BF))
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA;
    check_hip(hipMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz)));
    check_hip(hipMemcpy(tmpA, hostA, batchsz * x * y * hostsz,
                       hipMemcpyHostToDevice));
    long num_elements = batchsz * x * y;
    long block_size = 256;
    long num_blocks = (num_elements + block_size - 1) / block_size;
    float_to_bf16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (hip_bfloat16 *)devA);
    check_hip(hipGetLastError());
    hipFree(tmpA);
  } 
  else if (precision == mblas_data_type::MBLAS_R_8F_E4M3 || precision == mblas_data_type::MBLAS_R_8F_E5M2 )
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA;
    check_hip(hipMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz)));
    check_hip(hipMemcpy(tmpA, hostA, batchsz * x * y * hostsz,
                         hipMemcpyHostToDevice));
    long num_elements = batchsz * x * y;
    long block_size = 256;
    long num_blocks = (num_elements + block_size - 1) / block_size;
    __hip_fp8_interpretation_t interp;
    if (precision == mblas_data_type::MBLAS_R_8F_E4M3)
    {
      interp = __HIP_E4M3;
    }
    else if (precision == mblas_data_type::MBLAS_R_8F_E5M2)
    {
      interp = __HIP_E5M2;
    }
    float_to_fp8<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__hip_fp8_storage_t *)devA, interp);
    check_hip(hipGetLastError());
    hipFree(tmpA);
  }
  else 
  {
    check_hip(hipMemcpy(devA, hostA, (long)batchsz * x * y * hostsz,
                       hipMemcpyHostToDevice));
  }
}

void *convert_scalar(mblas_data_type precision, void *scalar)
{
  if (precision == mblas_data_type(MBLAS_R_16F))
  {
    // Read float value, convert to __half, write in-place
    float scalarVal = *static_cast<float *>(scalar);
    __half *hscalar = (__half *)scalar;
    *hscalar = __float2half(scalarVal);
    return scalar;
  }
  else if (precision == mblas_data_type(MBLAS_C_16F))
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
