#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#if (CUDART_VERSION >= 12800)
#include <cuda_fp4.h>
#endif

#include <complex>
#include <random>
#include <sstream>
#include <string>

#include "genericInit.h"
#include "mblasDataType.h"

// int sizeof_cudt_host(mblasDataType type);
void *allocateHostArr(mblasDataType type, long x, long y, int batch = 1);
void *allocateDevArr(mblasDataType type, long x, long y, int batch = 1);
void *allocateHDevArr(mblasDataType type, long x, long y, int batch = 1);
// void initHostH(mblasDataType precision, std::string initialization, void *ptr,
//                int rows_A, int cols_A, int ld, int batch, long long int stride,
//                float constant = 0.f, bool control = false, std::string filename = "");

template <typename T>
struct sizeofCUDT {
  int operator()();
};

template <typename T>
struct sizeofCUDTP {
  int operator()();
};

int get_packing_count(mblasDataType type);

//template <typename T>
//struct batchedPtrCopy {
//  void operator()(void **dptr, void *hArr, int batch_count, int x,
//                  int y, int flush_batch_count = 1, long total_block_size = 0);
//};

template <typename T>
struct batchedPtrMagic {
  void operator()(void **hptr, void *hArr, int batch_count, int x,
                  int y, int flush_batch_count = 1, long total_block_size = 0);
};

template <typename T>
struct allocSetScalar {
  void *operator()(std::string, std::string);
};

template <template <typename> class tFunc, class... Args>
auto typeCallHost(mblasDataType type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type;

template <template <typename> class tFunc, class... Args>
auto typeCallDev(mblasDataType type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type;

template <typename T>
void *allocSetScalarFunc(std::string sval, std::string sval2, T dummy) {
  // Only for real numbers, no need to worry about contents from sval2
  void *ptr = (void *)malloc(sizeof(T));
  T *data = (T *)ptr;
  std::istringstream iss(sval.c_str());
  iss >> *data;
  return ptr;
}

template <typename T>
void *allocSetScalarFunc(std::string sval, std::string sval2,
                         std::complex<T> dummy) {
  // Complex numbers, do something about sval2
  void *ptr = (void *)malloc(sizeof(std::complex<T>));
  std::complex<T> *data = (std::complex<T> *)ptr;
  T val;
  std::istringstream iss(sval.c_str());
  iss >> val;
  data->real(val);
  std::istringstream iss2(sval2.c_str());
  iss2 >> val;
  data->imag(val);
  return ptr;
}

template <typename T>
int sizeofCUDT<T>::operator()() {
  return sizeof(T);
}

template <typename T>
int sizeofCUDTP<T>::operator()() {
  return sizeof(T *);
}

template <typename T>
void *allocSetScalar<T>::operator()(std::string sval1, std::string sval2) {
  T dummy;
  return allocSetScalarFunc(sval1, sval2, std::forward<T>(dummy));
}

//template <typename T>
//void batchedPtrMagic<T>::operator()(void **hptr, void **dptr, void *dAr,
//                                    int batch_count, int x, int y) {
//  T **host = reinterpret_cast<T **>(hptr);
//  T *device_array = static_cast<T *>(dAr);
//  for (int i = 0; i < batch_count; i++) {
//    host[i] = device_array + (i * x * y);
//  }
//  // checkCuda(cudaMalloc(&dptr, batch_count * sizeof(T *)));
//  // hptr = reinterpret_cast<void **>(host);
//  // checkCuda(
//  cudaMemcpy(dptr, hptr, batch_count * sizeof(T *), cudaMemcpyHostToDevice);
//}

template <typename T>
void batchedPtrMagic<T>::operator()(void **hptr, void *dAr,
                                    int batch_count, int x, int y, int flush_batch_count, long total_block_size) {
  T **host = reinterpret_cast<T **>(hptr);
  T *device_array = static_cast<T *>(dAr);
  for (int j = 0; j < flush_batch_count; j++) {
    // Offset to the next block if using cache flushing
    int flush_offset = j*total_block_size;
    for (int i = 0; i < batch_count; i++) {
      host[j*batch_count + i] = device_array + flush_offset + (i * x * y);
    }
  }
}

void batchedPtrMagicGeneric(void **hptr, void *dAr, int batch_count, long x, long y, int flush_batch_count, long total_block_size, mblasDataType type);

//template <typename T>
//void batchedPtrCopy<T>::operator()(void **dptr, void *dAr,
//                                    int batch_count, int x, int y, int flush_batch_count = 1, long total_block_size = 0) {
//  void **hptr = (void **)malloc(batch_count * flush_batch_count * sizeof(T *));
//  checkCuda(cudaMalloc(dptr, batch_count * flush_batch_count * sizeof(T *)));
//  batchedPtrMagic<T>::operator()(hptr, dAr, batch_count, x, y, flush_batch_count, total_block_size);
//  cudaMemcpy(dptr, hptr, batch_count * sizeof(T *), cudaMemcpyHostToDevice);
//  free(hptr);
//}

template <template <typename> class tFunc, class... Args>
auto typeCallHost(mblasDataType type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type {
  // At runtime, determine which typed implementation to use and call it
  switch (type) {
    case mblasDataType::MBLAS_R_64F:
      return tFunc<double>()(args...);
    case mblasDataType::MBLAS_C_64F:
      return tFunc<std::complex<double>>()(args...);
    case mblasDataType::MBLAS_R_32F:
      return tFunc<float>()(args...);
    case mblasDataType::MBLAS_C_32F:
      return tFunc<std::complex<float>>()(args...);
    case mblasDataType::MBLAS_R_16BF:
      return tFunc<float>()(args...);
    case mblasDataType::MBLAS_C_16BF:
      return tFunc<std::complex<float>>()(args...);
    case mblasDataType::MBLAS_R_16F:
      return tFunc<float>()(args...);
    case mblasDataType::MBLAS_C_16F:
      return tFunc<std::complex<float>>()(args...);
    case mblasDataType::MBLAS_R_8F_E4M3:
      return tFunc<float>()(args...);
    //case mblasDataType::MBLAS_R_8F_UE4M3:
    //  return tFunc<float>()(args...);
    case mblasDataType::MBLAS_R_8F_E5M2:
      return tFunc<float>()(args...);
    case mblasDataType::MBLAS_R_4F_E2M1:
      return tFunc<float>()(args...);
    case mblasDataType::MBLAS_R_8I:
      return tFunc<__int8_t>()(args...);
    case mblasDataType::MBLAS_C_8I:
      return tFunc<std::complex<__int8_t>>()(args...);
    case mblasDataType::MBLAS_R_8U:
      return tFunc<__uint8_t>()(args...);
    case mblasDataType::MBLAS_C_8U:
      return tFunc<std::complex<__uint8_t>>()(args...);
    case mblasDataType::MBLAS_R_32I:
      return tFunc<__int32_t>()(args...);
    case mblasDataType::MBLAS_C_32I:
      return tFunc<std::complex<__int32_t>>()(args...);
    default:
      return tFunc<double>()(args...);
  }
}

template <template <typename> class tFunc, class... Args>
auto typeCallDev(mblasDataType type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type {
  // At runtime, determine which typed implementation to use and call it
  switch (type) {
    case mblasDataType::MBLAS_R_64F:
      return tFunc<double>()(args...);
    case mblasDataType::MBLAS_C_64F:
      return tFunc<std::complex<double>>()(args...);
    case mblasDataType::MBLAS_R_32F:
      return tFunc<float>()(args...);
    case mblasDataType::MBLAS_C_32F:
      return tFunc<std::complex<float>>()(args...);
    case mblasDataType::MBLAS_R_16BF:
      return tFunc<__nv_bfloat16>()(args...);
    case mblasDataType::MBLAS_C_16BF:
      return tFunc<std::complex<__nv_bfloat16>>()(args...);
    case mblasDataType::MBLAS_R_16F:
      return tFunc<__half>()(args...);
    case mblasDataType::MBLAS_C_16F:
      return tFunc<std::complex<__half>>()(args...);
    case mblasDataType::MBLAS_R_8F_E4M3:
      return tFunc<float>()(args...);
    //case mblasDataType::MBLAS_R_8F_UE4M3:
    //  return tFunc<__nv_fp8_e4m3>()(args...);
    case mblasDataType::MBLAS_R_8F_E5M2:
      return tFunc<__nv_fp8_e5m2>()(args...);
#if (CUDART_VERSION >= 12800)
    case mblasDataType::MBLAS_R_4F_E2M1:
      return tFunc<__nv_fp4x2_e2m1>()(args...);
#endif
    case mblasDataType::MBLAS_R_8I:
      return tFunc<__int8_t>()(args...);
    case mblasDataType::MBLAS_C_8I:
      return tFunc<std::complex<__int8_t>>()(args...);
    case mblasDataType::MBLAS_R_8U:
      return tFunc<__uint8_t>()(args...);
    case mblasDataType::MBLAS_C_8U:
      return tFunc<std::complex<__uint8_t>>()(args...);
    case mblasDataType::MBLAS_R_32I:
      return tFunc<__int32_t>()(args...);
    case mblasDataType::MBLAS_C_32I:
      return tFunc<std::complex<__int32_t>>()(args...);
    default:
      return tFunc<double>()(args...);
  }
}