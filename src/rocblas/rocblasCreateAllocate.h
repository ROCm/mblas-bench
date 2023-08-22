#pragma once
#include <rocblas/rocblas.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
// #include <cuda_fp8.h>
#include <hip/hip_runtime.h>

// #include <cuda/std/complex>
#include <hip/hip_complex.h>
#include <random>
#include <sstream>
#include <string>

// int sizeof_cudt_host(rocblas_datatype type);
void *allocateHostArr(rocblas_datatype type, long x, long y, int batch = 1);
void *allocateDevArr(rocblas_datatype type, long x, long y, int batch = 1);
void *allocateHDevArr(rocblas_datatype type, long x, long y, int batch = 1);

void initHostH(rocblas_datatype precision, std::string initialization, void *ptr,
               int rows_A, int cols_A, int ld, int batch, long long int stride,
               float constant = 0.f, bool alternating = false);

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
  void operator()(void **hptr, void **dptr, void *hArr, int batchct, int x,
                  int y);
};

template <typename T>
struct allocSetScalar {
  void *operator()(std::string, std::string);
};
//
// template <typename T>
// void *allocSetScalarFunc(std::string, std::string, T);
//
// template <typename T>
// void *allocSetScalarFunc(std::string, std::string, cuda::std::complex<T>);

template <template <typename> class tFunc, class... Args>
auto typeCallHost(rocblas_datatype type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type;

template <template <typename> class tFunc, class... Args>
auto typeCallDev(rocblas_datatype type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type;

template <typename T>
struct initHost {
  void operator()(std::string initialization, void *ptr, int rows_A, int cols_A,
                  int ld, int batch, long long int stride, bool control = false,
                  float constant = 0.f, std::string filename = "");
};
template <typename T>
void *allocSetScalarFunc(std::string sval, std::string sval2, T dummy) {
  // Only for real numbers, no need to worry about contents from sval2
  void *ptr = (void *)malloc(sizeof(T));
  T *data = (T *)ptr;
  std::istringstream iss(sval.c_str());
  iss >> *data;
  return ptr;
}

// template <>
// void *allocSetScalarFunc(std::string sval, std::string sval2,
//                          hipComplex dummy) {
//   // Complex numbers, do something about sval2
//   void *ptr = (void *)malloc(sizeof(hipComplex));
//   hipComplex *data = (hipComplex *)ptr;
//   float val;
//   std::istringstream iss(sval.c_str());
//   iss >> val;
//   data->x = val;
//   std::istringstream iss2(sval2.c_str());
//   iss2 >> val;
//   data->y = val;
//   return ptr;
// }

// template <>
// void *allocSetScalarFunc(std::string sval, std::string sval2,
//                          hipDoubleComplex dummy) {
//   // Complex numbers, do something about sval2
//   void *ptr = (void *)malloc(sizeof(hipComplex));
//   hipDoubleComplex *data = (hipDoubleComplex *)ptr;
//   double val;
//   std::istringstream iss(sval.c_str());
//   iss >> val;
//   data->x = val;
//   std::istringstream iss2(sval2.c_str());
//   iss2 >> val;
//   data->y = val;
//   return ptr;
// }

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

template <typename T>
void batchedPtrMagic<T>::operator()(void **hptr, void **dptr, void *dAr,
                                    int batchct, int x, int y) {
  T **host = reinterpret_cast<T **>(hptr);
  T *device_array = static_cast<T *>(dAr);
  for (int i = 0; i < batchct; i++) {
    host[i] = device_array + (i * x * y);
  }
  // checkCuda(cudaMalloc(&dptr, batchct * sizeof(T *)));
  // hptr = reinterpret_cast<void **>(host);
  // checkCuda(
  hipMemcpy(dptr, hptr, batchct * sizeof(T *), hipMemcpyHostToDevice);
}

template <typename T>
void fillRandHostBlasgemm(void *ptr, int rows_A, int cols_A, int ld, int batch,
                          long long int stride);
template <typename T>
void fillRandHostConstant(void *ptr, int rows_A, int cols_A, int ld, int batch,
                          long long int stride, float constant);

template <typename T>
void fillRandHostFromCSV(void *ptr, int rows_A, int cols_A, int ld, int batch,
                         long long int stride, std::string filename);

template <typename T>
void fillRandHostRandIntAS(void *ptr, int rows_A, int cols_A, int ld, int batch,
                           long long int stride, bool alternating);

template <typename T>
void fillRandHostTrigFloat(void *ptr, int rows_A, int cols_A, int ld, int batch,
                           long long int stride, bool isSin);

template <typename T>
void fillRandHostNormalFloat(void *ptr, int rows_A, int cols_A, int ld, int batch,
                             long long int stride);

template <template <typename> class tFunc, class... Args>
auto typeCallHost(rocblas_datatype type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type {
  // At runtime, determine which typed implementation to use and call it
  switch (type) {
    case rocblas_datatype_f64_r:
      return tFunc<double>()(args...);
    // case rocblas_datatype_f64_c:
    //   return tFunc<hipDoubleComplex>()(args...);
    case rocblas_datatype_f32_r:
      return tFunc<float>()(args...);
    // case rocblas_datatype_f32_c:
    //   return tFunc<hipComplex>()(args...);
    case rocblas_datatype_bf16_r:
      return tFunc<float>()(args...);
    // case rocblas_datatype_bf16_c:
    //   return tFunc<hipComplex>()(args...);
    case rocblas_datatype_f16_r:
      return tFunc<float>()(args...);
    // case rocblas_datatype_f16_c:
    //   return tFunc<hipComplex>()(args...);
    case rocblas_datatype_i8_r:
      return tFunc<__int8_t>()(args...);
    case rocblas_datatype_u8_r:
      return tFunc<__uint8_t>()(args...);
    case rocblas_datatype_i32_r:
      return tFunc<__int32_t>()(args...);
    default:
      return tFunc<double>()(args...);
  }
}

template <template <typename> class tFunc, class... Args>
auto typeCallDev(rocblas_datatype type, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type {
  // At runtime, determine which typed implementation to use and call it
  switch (type) {
    case rocblas_datatype_f64_r:
      return tFunc<double>()(args...);
    // case rocblas_datatype_f64_c:
    //   return tFunc<hipDoubleComplex>()(args...);
    case rocblas_datatype_f32_r:
      return tFunc<float>()(args...);
    // case rocblas_datatype_f32_c:
    //   return tFunc<hipComplex>()(args...);
    case rocblas_datatype_bf16_r:
      return tFunc<hip_bfloat16>()(args...);
    // case rocblas_datatype_bf16_c:
    //   return tFunc<hipComplex>()(args...);
    case rocblas_datatype_f16_r:
      return tFunc<__half>()(args...);
    // case rocblas_datatype_f16_c:
    //   return tFunc<hipComplex>()(args...);
    case rocblas_datatype_i8_r:
      return tFunc<__int8_t>()(args...);
    case rocblas_datatype_u8_r:
      return tFunc<__uint8_t>()(args...);
    case rocblas_datatype_i32_r:
      return tFunc<__int32_t>()(args...);
    default:
      return tFunc<double>()(args...);
  }
}

template <typename T>
inline T randIntGen(std::uniform_int_distribution<int> &idist,
                    std::mt19937 &gen, T &dummy);

// template <typename T>
// inline cuda::std::complex<T> randIntGen(
//     std::uniform_int_distribution<int> &idist, std::mt19937 &gen,
//     cuda::std::complex<T> &dummy);

template <typename T>
inline T randIntGenN(std::uniform_int_distribution<int> &idist,
                     std::mt19937 &gen, T &dummy);

// template <typename T>
// inline cuda::std::complex<T> randIntGenN(
//     std::uniform_int_distribution<int> &idist, std::mt19937 &gen,
//     cuda::std::complex<T> &dummy);

template <typename T>
inline T normalFloatGen(std::normal_distribution<double> &ndist,
                        std::mt19937 &gen, T &dummy);

// void dummy2() {
//  // This function forces the compiler to generate the needed templated
//  variants
//  // of each function. It is never called
//  void *h_A;
//  typeCallHost<fillRandHostBlasgemm>(CUDA_R_64F, h_A, 10, 10, 10, 1, 0);
//  typeCallHost<sizeofCUDTP>(CUDA_R_64F);
//  typeCallHost<allocSetScalar>(CUDA_R_64F, "1", "0");
//  typeCallDev<batchedPtrMagic>(CUDA_R_64F, (void **)NULL, (void **)NULL,
//                               (void *)NULL, 10, 10, 10);
//  // template void *allocSetScalar<double>::operator()(string);
//}