#pragma once
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

// #include <cuda/std/complex>
// #include <hip/hip_complex.h>
#include <complex>
#include <random>
#include <sstream>
#include <string>

#include "generic_init.h"
#include "mblas_data_type.h"

// DEPRECATED: These functions are disabled due to cross-library malloc/free issues.
// Use malloc(get_malloc_size_host(...)) and hipMalloc(..., get_malloc_size_dev(...)) instead.
// void *allocate_host_array(mblas_data_type type, long x, long y, int batch = 1);
// void *allocate_dev_array(mblas_data_type type, long x, long y, int batch = 1);
// void *allocate_host_dev_array(mblas_data_type type, long x, long y, int batch = 1);

long get_malloc_size_host(mblas_data_type type, long x, long y, int batch);
long get_malloc_size_dev(mblas_data_type type, long x, long y, int batch);

// void initHostH(mblas_data_type precision, std::string initialization, void *ptr,
//                int rows_A, int cols_A, int ld, int batch, long long int stride,
//                float constant = 0.f, bool alternating = false);

template <typename T>
struct sizeofCUDT {
  int operator()();
};

template <typename T>
struct sizeofCUDTP {
  int operator()();
};

template <typename T>
struct batchedPtrMagic {
  void operator()(void **hptr, void **dptr, void *hArr, int batch_count, int x,
                  int y);
};

// DEPRECATED: allocSetScalar performs malloc in the wrong library context.
// Use malloc + set_scalar instead to keep allocation in the calling library.
//template <typename T>
//struct allocSetScalar {
//  void *operator()(std::string, std::string);
//};

template <typename T>
struct set_scalar {
  void operator()(void *ptr, std::string, std::string);
};

template <template <typename> class tFunc, class... Args>
auto type_call_host(mblas_data_type type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type;

template <template <typename> class tFunc, class... Args>
auto type_call_dev(mblas_data_type type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type;

long get_malloc_size_scalar(mblas_data_type type);

template <typename T>
void set_scalar_val(void *ptr, std::string sval, std::string sval2, T dummy) {
  // Only for real numbers, no need to worry about contents from sval2
  //void *ptr = (void *)malloc(sizeof(T));
  T *data = (T *)ptr;
  std::istringstream iss(sval.c_str());
  iss >> *data;
  //return ptr;
}

template <typename T>
void set_scalar_val(void *ptr, std::string sval, std::string sval2,
                         std::complex<T> dummy) {
  // Complex numbers, do something about sval2
  //void *ptr = (void *)malloc(sizeof(std::complex<T>));
  std::complex<T> *data = (std::complex<T> *)ptr;
  T val;
  std::istringstream iss(sval.c_str());
  iss >> val;
  data->real(val);
  std::istringstream iss2(sval2.c_str());
  iss2 >> val;
  data->imag(val);
  //return ptr;
}

// DEPRECATED: alloc_set_scalar_val performs malloc in the wrong library context.
// Use malloc + set_scalar_val instead to keep allocation in the calling library.
//template <typename T>
//void *alloc_set_scalar_val(std::string sval, std::string sval2, T dummy) {
//  // Only for real numbers, no need to worry about contents from sval2
//  void *ptr = (void *)malloc(sizeof(T));
//  T *data = (T *)ptr;
//  std::istringstream iss(sval.c_str());
//  iss >> *data;
//  return ptr;
//}
//
//template <typename T>
//void *alloc_set_scalar_val(std::string sval, std::string sval2,
//                         std::complex<T> dummy) {
//  // Complex numbers, do something about sval2
//  void *ptr = (void *)malloc(sizeof(std::complex<T>));
//  std::complex<T> *data = (std::complex<T> *)ptr;
//  T val;
//  std::istringstream iss(sval.c_str());
//  iss >> val;
//  data->real(val);
//  std::istringstream iss2(sval2.c_str());
//  iss2 >> val;
//  data->imag(val);
//  return ptr;
//}

template <typename T>
int sizeofCUDT<T>::operator()() {
  return sizeof(T);
}

template <typename T>
int sizeofCUDTP<T>::operator()() {
  return sizeof(T *);
}

// DEPRECATED: allocSetScalar implementation commented out
//template <typename T>
//void *allocSetScalar<T>::operator()(std::string sval1, std::string sval2) {
//  T dummy;
//  return alloc_set_scalar_val(sval1, sval2, std::forward<T>(dummy));
//}

template <typename T>
void set_scalar<T>::operator()(void *ptr, std::string sval1, std::string sval2) {
  T dummy;
  return set_scalar_val(ptr, sval1, sval2, std::forward<T>(dummy));
}

template <typename T>
void batchedPtrMagic<T>::operator()(void **hptr, void **dptr, void *dAr,
                                    int batch_count, int x, int y) {
  T **host = reinterpret_cast<T **>(hptr);
  T *device_array = static_cast<T *>(dAr);
  for (int i = 0; i < batch_count; i++) {
    host[i] = device_array + (i * x * y);
  }
  // check_cuda(cudaMalloc(&dptr, batch_count * sizeof(T *)));
  // hptr = reinterpret_cast<void **>(host);
  // check_cuda(
  hipMemcpy(dptr, hptr, batch_count * sizeof(T *), hipMemcpyHostToDevice);
}

// template <typename T>
// void fill_rand_host_blasgemm(void *ptr, int rows_A, int cols_A, int ld, int batch,
//                           long long int stride);
// template <typename T>
// void fill_rand_host_constant(void *ptr, int rows_A, int cols_A, int ld, int batch,
//                           long long int stride, float constant);
// 
// template <typename T>
// void fill_rand_host_csv(void *ptr, int rows_A, int cols_A, int ld, int batch,
//                          long long int stride, std::string filename);
// 
// template <typename T>
// void fill_rand_host_rand_int_alternating(void *ptr, int rows_A, int cols_A, int ld, int batch,
//                            long long int stride, bool alternating);
// 
// template <typename T>
// void fill_rand_host_trig_float(void *ptr, int rows_A, int cols_A, int ld, int batch,
//                            long long int stride, bool isSin);
// 
// template <typename T>
// void fill_rand_host_normal_float(void *ptr, int rows_A, int cols_A, int ld, int batch,
//                              long long int stride);

template <template <typename> class tFunc, class... Args>
auto type_call_host(mblas_data_type type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type {
  // At runtime, determine which typed implementation to use and call it
  switch (type) {
    case MBLAS_R_64F:
      return tFunc<double>()(args...);
    case MBLAS_C_64F:
      return tFunc<std::complex<double>>()(args...);
    case MBLAS_R_32F:
      return tFunc<float>()(args...);
    case MBLAS_C_32F:
      return tFunc<std::complex<float>>()(args...);
    case MBLAS_R_16BF:
      return tFunc<float>()(args...);
    case MBLAS_R_16F:
      return tFunc<float>()(args...);
    case mblas_data_type::MBLAS_R_8F_E4M3:
      return tFunc<float>()(args...);
    case mblas_data_type::MBLAS_R_8F_E5M2:
      return tFunc<float>()(args...);
    case MBLAS_R_8I:
      return tFunc<__int8_t>()(args...);
    case MBLAS_R_8U:
      return tFunc<__uint8_t>()(args...);
    case MBLAS_R_32I:
      return tFunc<__int32_t>()(args...);
    default:
      return tFunc<double>()(args...);
  }
}

//template <template <typename> class tFunc, class... Args>
//auto type_call_host(hipblasDatatype_t type, Args... args) ->
//    typename std::result_of<tFunc<double>(Args...)>::type {
//  // At runtime, determine which typed implementation to use and call it
//  switch (type) {
//    case HIPBLAS_R_64F:
//      return tFunc<double>()(args...);
//    // case HIPBLAS_C_64F:
//    //   return tFunc<hipDoubleComplex>()(args...);
//    case HIPBLAS_R_32F:
//      return tFunc<float>()(args...);
//    // case HIPBLAS_C_32F:
//    //   return tFunc<hipComplex>()(args...);
//    case HIPBLAS_R_16B:
//      return tFunc<float>()(args...);
//    // case HIPBLAS_C_16B:
//    //   return tFunc<hipComplex>()(args...);
//    case HIPBLAS_R_16F:
//      return tFunc<float>()(args...);
//    // case HIPBLAS_C_16F:
//    //   return tFunc<hipComplex>()(args...);
//    case HIPBLAS_R_8I:
//      return tFunc<__int8_t>()(args...);
//    case HIPBLAS_R_8U:
//      return tFunc<__uint8_t>()(args...);
//    case HIPBLAS_R_32I:
//      return tFunc<__int32_t>()(args...);
//    default:
//      return tFunc<double>()(args...);
//  }
//}

template <template <typename> class tFunc, class... Args>
auto type_call_dev(mblas_data_type type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type {
  // At runtime, determine which typed implementation to use and call it
  switch (type) {
    case MBLAS_R_64F:
      return tFunc<double>()(args...);
    case MBLAS_C_64F:
      return tFunc<std::complex<double>>()(args...);
    case MBLAS_R_32F:
      return tFunc<float>()(args...);
    case MBLAS_C_32F:
      return tFunc<std::complex<float>>()(args...);
    case MBLAS_R_16BF:
      return tFunc<hip_bfloat16>()(args...);
    // case hipDataType_bf16_c:
    //   return tFunc<hipComplex>()(args...);
    case MBLAS_R_16F:
      return tFunc<__half>()(args...);
    // case hipDataType_f16_c:
    //   return tFunc<hipComplex>()(args...);
    // case mblas_data_type::MBLAS_R_8F_E4M3:
    //   return tFunc<__hip_fp8_e4m3>()(args...);
    // case mblas_data_type::MBLAS_R_8F_E5M2:
    //   return tFunc<__hip_fp8_e5m2>()(args...);
    case mblas_data_type::MBLAS_R_8F_E4M3:
      return tFunc<__hip_fp8_storage_t>()(args...);
    case mblas_data_type::MBLAS_R_8F_E5M2:
      return tFunc<__hip_fp8_storage_t>()(args...);
    case MBLAS_R_8I:
      return tFunc<__int8_t>()(args...);
    case MBLAS_R_8U:
      return tFunc<__uint8_t>()(args...);
    case MBLAS_R_32I:
      return tFunc<__int32_t>()(args...);
    default:
      return tFunc<double>()(args...);
  }
}

//template <template <typename> class tFunc, class... Args>
//auto type_call_dev(hipblasDatatype_t type, Args... args) ->
//    typename std::result_of<tFunc<double>(Args...)>::type {
//  // At runtime, determine which typed implementation to use and call it
//  switch (type) {
//    case HIPBLAS_R_64F:
//      return tFunc<double>()(args...);
//    // case HIPBLAS_C_64F:
//    //   return tFunc<hipDoubleComplex>()(args...);
//    case HIPBLAS_R_32F:
//      return tFunc<float>()(args...);
//    // case HIPBLAS_C_32F:
//    //   return tFunc<hipComplex>()(args...);
//    case HIPBLAS_R_16B:
//      return tFunc<hip_bfloat16>()(args...);
//    // case HIPBLAS_C_16B:
//    //   return tFunc<hipComplex>()(args...);
//    case HIPBLAS_R_16F:
//      return tFunc<__half>()(args...);
//    // case HIPBLAS_C_16F:
//    //   return tFunc<hipComplex>()(args...);
//    case HIPBLAS_R_8I:
//      return tFunc<__int8_t>()(args...);
//    case HIPBLAS_R_8U:
//      return tFunc<__uint8_t>()(args...);
//    case HIPBLAS_R_32I:
//      return tFunc<__int32_t>()(args...);
//    default:
//      return tFunc<double>()(args...);
//  }
//}

// template <typename T>
// inline T rand_int_gen(std::uniform_int_distribution<int> &idist,
//                     std::mt19937 &gen, T &dummy);
// 
// // template <typename T>
// // inline cuda::std::complex<T> rand_int_gen(
// //     std::uniform_int_distribution<int> &idist, std::mt19937 &gen,
// //     cuda::std::complex<T> &dummy);
// 
// template <typename T>
// inline T rand_int_gen_negative(std::uniform_int_distribution<int> &idist,
//                      std::mt19937 &gen, T &dummy);
// 
// // template <typename T>
// // inline cuda::std::complex<T> rand_int_gen_negative(
// //     std::uniform_int_distribution<int> &idist, std::mt19937 &gen,
// //     cuda::std::complex<T> &dummy);
// 
// template <typename T>
// inline T normal_float_gen(std::normal_distribution<double> &ndist,
//                         std::mt19937 &gen, T &dummy);

// void dummy2() {
//  // This function forces the compiler to generate the needed templated
//  variants
//  // of each function. It is never called
//  void *h_A;
//  type_call_host<fill_rand_host_blasgemm>(CUDA_R_64F, h_A, 10, 10, 10, 1, 0);
//  type_call_host<sizeofCUDTP>(CUDA_R_64F);
//  // DEPRECATED: allocSetScalar no longer used
//  //type_call_host<allocSetScalar>(CUDA_R_64F, "1", "0");
//  type_call_dev<batchedPtrMagic>(CUDA_R_64F, (void **)NULL, (void **)NULL,
//                               (void *)NULL, 10, 10, 10);
//}