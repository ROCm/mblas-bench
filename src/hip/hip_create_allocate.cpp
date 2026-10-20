#include "hip_create_allocate.h"

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
#include "generic_setup.h"

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

long long get_malloc_size(mblas_data_type type, long x, long y, int batch,
                          long long stride, bool use_dev_type) {
  // use_dev_type == false: host staging size (float element size, no packing).
  // use_dev_type == true : packed device size (native element size / packing).
  int typesize = use_dev_type ? type_call_dev<sizeofCUDT>(type)
                              : type_call_host<sizeofCUDT>(type);
  long long packing_count = use_dev_type ? type.get_packing_count() : 1;
  long long base = x * y;
  long long total_elements = stride * (batch - 1) + base;
  return ceil_division(total_elements * typesize, packing_count);
}

long get_malloc_size_scalar(mblas_data_type type) {
  return type_call_host<sizeofCUDT>(type);
}
