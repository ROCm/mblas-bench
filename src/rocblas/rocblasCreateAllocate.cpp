#include "rocblasCreateAllocate.h"

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

#include "rocblasError.h"
#include "genericInit.h"

// using cuda::std::complex;
using std::string;

void *allocate_host_array(mblasDataType type, long x, long y, int batch) {
  int typesize = type_call_host<sizeofCUDT>(type);
  void *data = (void *)malloc(x * y * batch * typesize);
  return data;
}

void *allocate_dev_array(mblasDataType type, long x, long y, int batch) {
  int typesize = type_call_dev<sizeofCUDT>(type);
  void *data;
  check_hip(hipMalloc(&data, x * y * batch * typesize));
  return data;
}

void *allocate_host_dev_array(mblasDataType type, long x, long y, int batch) {
  int typesize = type_call_host<sizeofCUDT>(type);
  void *data;
  check_hip(hipMalloc(&data, x * y * batch * typesize));
  return data;
}

int get_packing_count(mblasDataType type) {
  if (type == mblasDataType::MBLAS_R_4F_E2M1) {
    // Two 4-bit FP4 floats per byte
    return 2;
  } else {
    return 1;
  }

}
