#include "hip_convert.h"

#include <bitset>
// #include <cuda_fp8.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#if HIP_VERSION >= 70000000
//#include <hip/hip_fp4.h>
#include <hip/hip_fp6.h>
#endif
#include "hip_create_allocate.h"
#include "hip_error.h"
#include "generic_setup.h"

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

#if HIP_VERSION >= 70000000
// Convert float to OCP UE8M0 (8-bit unsigned exponent, no mantissa, bias 127).
// Mirrors cuda's __nv_cvt_float_to_e8m0(x, __NV_SATFINITE, cudaRoundNearest):
// abs(x) -> nearest representable 2^e with round-to-nearest-even, sat-to-finite,
// NaN propagates as 0xFF. Sign is dropped because UE8M0 is unsigned.
__global__ void float_to_ue8m0(float *input, size_t num_elements,
                                __hip_fp8_storage_t *output)
{
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements) return;

  const uint32_t bits = __float_as_uint(input[idx]);
  const uint32_t abs_bits = bits & 0x7FFFFFFFu;
  const uint32_t exp_field = (abs_bits >> 23) & 0xFFu;
  const uint32_t mant_field = abs_bits & 0x7FFFFFu;

  uint32_t result;
  if (exp_field == 0xFFu) {
    result = (mant_field != 0u) ? 0xFFu : 0xFEu;
  } else if (exp_field == 0u) {
    // Zero or float subnormal -> smallest representable scale (no zero in UE8M0).
    result = 0x00u;
  } else {
    // Linear-distance midpoint between 2^e and 2^(e+1) is at mantissa == 0x400000.
    uint32_t rounded = exp_field;
    if (mant_field > 0x400000u) {
      rounded += 1u;
    } else if (mant_field == 0x400000u) {
      rounded += (exp_field & 1u);
    }
    if (rounded > 0xFEu) rounded = 0xFEu;
    result = rounded;
  }

  output[idx] = (__hip_fp8_storage_t)result;
}
#endif

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

void copy_and_convert(mblas_data_type precision, void *hostA, void *devA, long x, long y,
                      int batchsz, long long stride)
{
  if (batchsz * x * y == 0)
  {
    return;
  }
  long long base = x * y;
  long long total_elements = stride * (batchsz - 1) + base;
  long hostsz = type_call_host<sizeofCUDT>(precision);
  if (precision == mblas_data_type(MBLAS_C_16F) || precision == mblas_data_type(MBLAS_R_16F))
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA;
    check_hip(hipMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_hip(hipMemcpy(tmpA, hostA, total_elements * hostsz, hipMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size = 256;
    long long num_blocks = (num_elements + block_size - 1) / block_size;
    float_to_fp16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__half *)devA);
    check_hip(hipGetLastError());
    hipFree(tmpA);
  }
  else if (precision == mblas_data_type(MBLAS_C_16BF) || precision == mblas_data_type(MBLAS_R_16BF))
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA;
    check_hip(hipMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_hip(hipMemcpy(tmpA, hostA, total_elements * hostsz, hipMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size = 256;
    long long num_blocks = (num_elements + block_size - 1) / block_size;
    float_to_bf16<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (hip_bfloat16 *)devA);
    check_hip(hipGetLastError());
    hipFree(tmpA);
  }
  else if (precision == mblas_data_type::MBLAS_R_8F_E4M3 ||
           precision == mblas_data_type::MBLAS_R_8F_E5M2)
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA;
    check_hip(hipMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_hip(hipMemcpy(tmpA, hostA, total_elements * hostsz, hipMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size = 256;
    long long num_blocks = (num_elements + block_size - 1) / block_size;
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
#if HIP_VERSION >= 70000000
  else if (precision == mblas_data_type::MBLAS_R_4F_E2M1)
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA;
    check_hip(hipMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_hip(hipMemcpy(tmpA, hostA, total_elements * hostsz,
                         hipMemcpyHostToDevice));
    long long num_elements = ceil_division(total_elements, 2ll);
    long long block_size = 256;
    long long num_blocks = (num_elements + block_size - 1) / block_size;
    float_to_fp4_helper((float2 *)tmpA, num_elements, (__hip_fp8_storage_t *)devA, num_blocks, block_size);
    //float_to_fp4<<<num_blocks, block_size>>>((float2 *)tmpA, num_elements, 
    //                                           (__hip_fp8_storage_t *)devA);
    check_hip(hipGetLastError());
    hipFree(tmpA);
  }
  else if (precision == mblas_data_type::MBLAS_R_6F_E2M3 || precision == mblas_data_type::MBLAS_R_6F_E3M2)
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA;
    check_hip(hipMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_hip(hipMemcpy(tmpA, hostA, total_elements * hostsz,
                         hipMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size = 256;
    long long num_blocks = (num_elements + block_size - 1) / block_size;
    mblas_fp6_interpretation_t interp;
    if (precision == mblas_data_type::MBLAS_R_6F_E2M3)
    {
      interp = MBLAS_F6_E2M3;
    }
    else if (precision == mblas_data_type::MBLAS_R_6F_E3M2)
    {
      interp = MBLAS_F6_E3M2;
    }
    float_to_fp6_helper((float *)tmpA, num_elements, (__hip_fp8_storage_t *)devA, interp, num_blocks, block_size);
    // float_to_fp6<<<num_blocks, block_size>>>((float *)tmpA, num_elements,
    //                                            (__hip_fp8_storage_t *)devA, interp);
    check_hip(hipGetLastError());
    hipFree(tmpA);
  }
  else if (precision == mblas_data_type::MBLAS_R_8F_UE8M0)
  {
    // Allocate memory in the device for host precision (float)
    void *tmpA;
    check_hip(hipMalloc(&tmpA, get_malloc_size_host(precision, x, y, batchsz, stride)));
    check_hip(hipMemcpy(tmpA, hostA, total_elements * hostsz,
                         hipMemcpyHostToDevice));
    long long num_elements = total_elements;
    long long block_size = 256;
    long long num_blocks = (num_elements + block_size - 1) / block_size;
    float_to_ue8m0<<<num_blocks, block_size>>>((float *)tmpA, num_elements, (__hip_fp8_storage_t *)devA);
    check_hip(hipGetLastError());
    hipFree(tmpA);
  }
#endif
  else 
  {
    check_hip(hipMemcpy(devA, hostA,
                        get_malloc_size_host(precision, x, y, batchsz, stride),
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
