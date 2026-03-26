#include "hip_create_allocate.h"

#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
// #include <cuda_fp8.h>
#include <hip/hip_runtime.h>
//#include <omp.h>

// #include <cuda/std/complex>
// #include <hip/hip_complex.h>
#include <iostream>
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "hip_error.h"
#include "generic_init.h"

// using cuda::std::complex;
using std::string;

// DEPRECATED: Disabled due to cross-library malloc/free issues
// Use malloc(get_malloc_size_host(...)) instead
// void *allocate_host_array(mblas_data_type type, long x, long y, int batch) {
//   int typesize = type_call_host<sizeofCUDT>(type);
//   void *data = (void *)malloc(x * y * batch * typesize);
//   return data;
// }

// DEPRECATED: Disabled due to cross-library malloc/free issues
// Use hipMalloc(&ptr, get_malloc_size_dev(...)) instead
// void *allocate_dev_array(mblas_data_type type, long x, long y, int batch) {
//   int typesize = type_call_dev<sizeofCUDT>(type);
//   void *data;
//   check_hip(hipMalloc(&data, x * y * batch * typesize));
//   return data;
// }

// DEPRECATED: Disabled due to cross-library malloc/free issues
// Use hipMalloc(&ptr, get_malloc_size_host(...)) instead
// void *allocate_host_dev_array(mblas_data_type type, long x, long y, int batch) {
//   int typesize = type_call_host<sizeofCUDT>(type);
//   void *data;
//   check_hip(hipMalloc(&data, x * y * batch * typesize));
//   return data;
// }

long get_malloc_size_host(mblas_data_type type, long x, long y, int batch,
                          long long stride) {
  int typesize = type_call_host<sizeofCUDT>(type);
  long packing_count = type.get_packing_count();
  long base = x * y;
  long total_elements;
  if (batch > 1 && stride > base) {
    total_elements = stride * (batch - 1) + base;
  } else {
    total_elements = base * batch;
  }
  return total_elements * typesize * packing_count;
}

long get_malloc_size_dev(mblas_data_type type, long x, long y, int batch,
                         long long stride) {
  int typesize = type_call_dev<sizeofCUDT>(type);
  long packing_count = type.get_packing_count();
  long base = x * y;
  long total_elements;
  if (batch > 1 && stride > base) {
    total_elements = stride * (batch - 1) + base;
  } else {
    total_elements = base * batch;
  }
  return total_elements * typesize * packing_count;
}

long get_malloc_size_scalar(mblas_data_type type) {
  return type_call_host<sizeofCUDT>(type);
}
