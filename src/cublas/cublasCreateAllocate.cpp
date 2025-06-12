#include "cublasCreateAllocate.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <cuda/std/complex>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "cudaError.h"
#include "genericInit.h"
#include "genericSetup.h"

void *allocate_host_array(mblasDataType type, long x, long y, int batch) {
  int typesize = type_call_host<sizeofCUDT>(type);
  void *data = (void *)malloc(x * y * batch * typesize);
  return data;
}

void *allocate_dev_array(mblasDataType type, long x, long y, int batch) {
  int typesize = type_call_dev<sizeofCUDT>(type);
  long packing_count = get_packing_count(type);
  // Compensate for packing of 4 bit dtypes
  long malloc_size = ceil_division(x * y * batch * typesize, packing_count);
  void *data;
  check_cuda(cudaMalloc(&data, malloc_size));
  return data;
}

void *allocate_host_dev_array(mblasDataType type, long x, long y, int batch) {
  int typesize = type_call_host<sizeofCUDT>(type);
  void *data;
  check_cuda(cudaMalloc(&data, x * y * batch * typesize));
  return data;
}

void batched_pointer_magic_generic(void **hptr, void *dAr, int batch_count, long x, long y, int flush_batch_count, long total_block_size, mblasDataType type) {
  //T **host = reinterpret_cast<T **>(hptr);
  //T *device_array = static_cast<T *>(dAr);
  long type_size = type_call_host<sizeofCUDT>(type);
  long packing_count = get_packing_count(type);
  for (int j = 0; j < flush_batch_count; j++) {
    // Offset to the next block if using cache flushing
    int flush_offset = j*total_block_size;
    for (int i = 0; i < batch_count; i++) {
      hptr[j*batch_count + i] = (char*) dAr + flush_offset + ceil_division(i * x * y * type_size, packing_count);
    }
  }
}


int get_packing_count(mblasDataType type) {
  if (type == mblasDataType::MBLAS_R_4F_E2M1) {
    // Two 4-bit FP4 floats per byte
    return 2;
  } else {
    return 1;
  }

}

