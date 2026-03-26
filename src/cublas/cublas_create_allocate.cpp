#include "cublas_create_allocate.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "cuda_error.h"
#include "generic_init.h"
#include "generic_setup.h"

// DEPRECATED: Disabled due to cross-library malloc/free issues
// Use malloc(get_malloc_size_host(...)) instead
// void *allocate_host_array(mblas_data_type type, long x, long y, int batch) {
//   int typesize = type_call_host<sizeofCUDT>(type);
//   void *data = (void *)malloc(x * y * batch * typesize);
//   return data;
// }

// DEPRECATED: Disabled due to cross-library malloc/free issues
// Use cudaMalloc(&ptr, get_malloc_size_dev(...)) instead
// void *allocate_dev_array(mblas_data_type type, long x, long y, int batch) {
//   int typesize = type_call_dev<sizeofCUDT>(type);
//   long packing_count = get_packing_count(type);
//   // Compensate for packing of 4 bit dtypes
//   long malloc_size = ceil_division(x * y * batch * typesize, packing_count);
//   void *data;
//   check_cuda(cudaMalloc(&data, malloc_size));
//   return data;
// }

// DEPRECATED: Disabled due to cross-library malloc/free issues
// Use cudaMalloc(&ptr, get_malloc_size_host(...)) instead
// void *allocate_host_dev_array(mblas_data_type type, long x, long y, int batch) {
//   int typesize = type_call_host<sizeofCUDT>(type);
//   void *data;
//   check_cuda(cudaMalloc(&data, x * y * batch * typesize));
//   return data;
// }

void batched_pointer_magic_generic(void **hptr, void *dAr, int batch_count, long x, long y, int flush_batch_count, long total_block_size, mblas_data_type type) {
  //T **host = reinterpret_cast<T **>(hptr);
  //T *device_array = static_cast<T *>(dAr);
  long type_size = type_call_host<sizeofCUDT>(type);
  long packing_count = type.get_packing_count();
  for (int j = 0; j < flush_batch_count; j++) {
    // Offset to the next block if using cache flushing
    int flush_offset = j*total_block_size;
    for (int i = 0; i < batch_count; i++) {
      hptr[j*batch_count + i] = (char*) dAr + flush_offset + ceil_division(i * x * y * type_size, packing_count);
    }
  }
}

long get_malloc_size_scalar(mblas_data_type type) {
  return type_call_host<sizeofCUDT>(type);
}

long long get_malloc_size_host(mblas_data_type type, long x, long y, int batch, long long stride) {
  int typesize = type_call_host<sizeofCUDT>(type);
  long long base = x * y;
  long long total_elements;
  if (batch > 1 && stride > base) {
    total_elements = stride * (batch - 1) + base;
  } else {
    total_elements = base * batch;
  }
  return total_elements * typesize;
}

long long get_malloc_size_dev(mblas_data_type type, long x, long y, int batch, long long stride) {
  int typesize = type_call_dev<sizeofCUDT>(type);
  long long packing_count = type.get_packing_count();
  long long base = x * y;
  long long total_elements;
  if (batch > 1 && stride > base) {
    total_elements = stride * (batch - 1) + base;
  } else {
    total_elements = base * batch;
  }
  return ceil_division(total_elements * typesize, packing_count);
}
