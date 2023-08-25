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

using cuda::std::complex;
using std::string;

// struct sizeofCUDTHost
// {
//     int operator()()
//     {

//     }
// };

// template void *allocSetScalar<double>::operator()(string);
// template void *allocSetScalar<float>::operator()(string);

// template <>
// void *allocSetScalar<complex<float>>::operator()(string sval) {
//  return NULL;
//}
//
// template <>
// void *allocSetScalar<complex<double>>::operator()(string sval) {
//  return NULL;
//}

void *allocateHostArr(cudaDataType_t type, long x, long y, int batch) {
  int typesize = typeCallHost<sizeofCUDT>(type);
  void *data = (void *)malloc(x * y * batch * typesize);
  return data;
}

void *allocateDevArr(cudaDataType_t type, long x, long y, int batch) {
  int typesize = typeCallDev<sizeofCUDT>(type);
  void *data;
  checkCuda(cudaMalloc(&data, x * y * batch * typesize));
  return data;
}

void *allocateHDevArr(cudaDataType_t type, long x, long y, int batch) {
  int typesize = typeCallHost<sizeofCUDT>(type);
  void *data;
  checkCuda(cudaMalloc(&data, x * y * batch * typesize));
  return data;
}

// void *allocateScalar(cudaDataType_t type) {
//  int typesize = typeCallDev<sizeofCUDT>(type);
//  void *scalar = (void *)
//}

void dummy() {
  // This function forces the compiler to generate the needed templated variants
  // of each function. It is never called
  void *h_A;
  typeCallHost<sizeofCUDTP>(CUDA_R_64F);
  typeCallHost<allocSetScalar>(CUDA_R_64F, "1", "0");
  typeCallDev<batchedPtrMagic>(CUDA_R_64F, (void **)NULL, (void **)NULL,
                               (void *)NULL, 10, 10, 10);
  // template void *allocSetScalar<double>::operator()(string);
}

void initHostH(cudaDataType_t precision, std::string initialization, void *ptr,
               int rows_A, int cols_A, int ld, int batch, long long int stride,
               float constant, bool alternating) {
  typeCallHost<initHost>(precision, initialization, ptr, rows_A, cols_A, ld,
                         batch, stride, constant, alternating);
}

template <typename T>
void initHost<T>::operator()(std::string initialization, void *ptr, int rows_A,
                             int cols_A, int ld, int batch,
                             long long int stride, bool control,
                             float constant) {
  if (initialization == "rand_int") {
    fillRandHostRandIntAS<T>(ptr, rows_A, cols_A, ld, batch, stride, control);
  } else if (initialization == "trig_float") {
    fillRandHostTrigFloat<T>(ptr, rows_A, cols_A, ld, batch, stride, control);
  } else if (initialization == "hpl") {
  } else if (initialization == "blasgemm") {
    fillRandHostBlasgemm<T>(ptr, rows_A, cols_A, ld, batch, stride);
  } else if (initialization == "constant") {
    fillRandHostConstant<T>(ptr, rows_A, cols_A, ld, batch, stride, constant);
  }
}

template void initHost<double>::operator()(std::string, void *, int, int, int,
                                           int, long long int, bool, float);
template void initHost<complex<double>>::operator()(std::string, void *, int,
                                                    int, int, int,
                                                    long long int, bool, float);
template void initHost<float>::operator()(std::string, void *, int, int, int,
                                          int, long long int, bool, float);
template void initHost<complex<float>>::operator()(std::string, void *, int,
                                                   int, int, int, long long int,
                                                   bool, float);
template void initHost<__int8_t>::operator()(std::string, void *, int, int, int,
                                             int, long long int, bool, float);
template void initHost<complex<__int8_t>>::operator()(std::string, void *, int,
                                                      int, int, int,
                                                      long long int, bool,
                                                      float);
template void initHost<__uint8_t>::operator()(std::string, void *, int, int,
                                              int, int, long long int, bool,
                                              float);
template void initHost<complex<__uint8_t>>::operator()(std::string, void *, int,
                                                       int, int, int,
                                                       long long int, bool,
                                                       float);
template void initHost<__int32_t>::operator()(std::string, void *, int, int,
                                              int, int, long long int, bool,
                                              float);
template void initHost<complex<__int32_t>>::operator()(std::string, void *, int,
                                                       int, int, int,
                                                       long long int, bool,
                                                       float);

// Instances of the following functions should be defined implicitly by defining
// those of initHost
template <typename T>
void fillRandHostBlasgemm(void *ptr, int rows_A, int cols_A, int ld, int batch,
                          long long int stride) {
  int a = 1;
  T *A = (T *)ptr;
  for (size_t i = 0; i < rows_A * cols_A * batch; i++) {
    A[i] = (T)rand() / (T)(RAND_MAX / a);
    // if(i < 10)
    //      std::cout << *((double *)ptr+i)  << std::endl;
  }
}

template <typename T>
void fillRandHostConstant(void *ptr, int rows_A, int cols_A, int ld, int batch,
                          long long int stride, float constant) {
  int a = 1;
  T *A = (T *)ptr;
  for (size_t i = 0; i < rows_A * cols_A * batch; i++) {
    A[i] = (T)(constant);
  }
}

template <typename T>
void fillRandHostRandIntAS(void *ptr, int rows_A, int cols_A, int ld, int batch,
                           long long int stride, bool alternating) {
  std::random_device r;
  std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> uniform_dist(1, 10);
  T *A = (T *)ptr;
  T dummy;
  for (size_t i_batch = 0; i_batch < batch; i_batch++) {
    for (size_t j = 0; j < cols_A; ++j) {
      size_t offset = j * ld + i_batch * stride;
      for (size_t i = 0; i < rows_A; ++i) {
        if ((!alternating) || (j % 2 ^ i % 2)) {
          A[i + offset] = randIntGen(uniform_dist, gen, dummy);
        } else {
          A[i + offset] = randIntGenN(uniform_dist, gen, dummy);
        }
      }
    }
  }
}

template <typename T>
void fillRandHostTrigFloat(void *ptr, int rows_A, int cols_A, int ld, int batch,
                           long long int stride, bool isSin) {
  T *A = (T *)ptr;
  for (size_t i_batch = 0; i_batch < batch; i_batch++) {
    for (size_t j = 0; j < cols_A; ++j) {
      size_t offset = j * ld + i_batch * stride;
      for (size_t i = 0; i < rows_A; ++i) {
        if (isSin) {
          A[i + offset] = T(sin(i + offset));
        } else {
          A[i + offset] = T(cos(i + offset));
        }
      }
    }
  }
}

template <typename T>
inline T randIntGen(std::uniform_int_distribution<int> &idist,
                    std::mt19937 &gen, T &dummy) {
  return T(idist(gen));
}

template <typename T>
inline complex<T> randIntGen(std::uniform_int_distribution<int> &idist,
                             std::mt19937 &gen, complex<T> &dummy) {
  return {T(idist(gen)), T(idist(gen))};
}

template <typename T>
inline T randIntGenN(std::uniform_int_distribution<int> &idist,
                     std::mt19937 &gen, T &dummy) {
  return -T(idist(gen));
}

template <typename T>
inline complex<T> randIntGenN(std::uniform_int_distribution<int> &idist,
                              std::mt19937 &gen, complex<T> &dummy) {
  return {-T(idist(gen)), -T(idist(gen))};
}

// template <typename T>
// inline T randIntGen(std::uniform_int_distribution<int> &idist,
//                    std::mt19937 &gen, T &dummy) {
//  return T(rand() % 10 + 1);
//}
//
// template <typename T>
// inline complex<T> randIntGen(std::uniform_int_distribution<int> &idist,
//                             std::mt19937 &gen, complex<T> &dummy) {
//  return {T(rand() % 10 + 1), T(rand() % 10 + 1)};
//}
//
// template <typename T>
// inline T randIntGenN(std::uniform_int_distribution<int> &idist,
//                     std::mt19937 &gen, T &dummy) {
//  return -T(rand() % 10 + 1);
//}
//
// template <typename T>
// inline complex<T> randIntGenN(std::uniform_int_distribution<int> &idist,
//                              std::mt19937 &gen, complex<T> &dummy) {
//  return {-T(rand() % 10 + 1), -T(rand() % 10 + 1)};
//}

// template <typename T>
// inline T randIntGen(std::uniform_int_distribution<int> &idist,
//                    std::mt19937 &gen, T &dummy) {
//  return T(rand());
//}
//
// template <typename T>
// inline complex<T> randIntGen(std::uniform_int_distribution<int> &idist,
//                             std::mt19937 &gen, complex<T> &dummy) {
//  return {T(rand()), T(rand())};
//}
//
// template <typename T>
// inline T randIntGenN(std::uniform_int_distribution<int> &idist,
//                     std::mt19937 &gen, T &dummy) {
//  return -T(rand());
//}
//
// template <typename T>
// inline complex<T> randIntGenN(std::uniform_int_distribution<int> &idist,
//                              std::mt19937 &gen, complex<T> &dummy) {
//  return {-T(rand()), -T(rand())};
//}
// int sizeof_cudt_host(cudaDataType_t type) {
//     int size = 0;
//     complex<double> z1(1,1.5);
//     switch(type) {
//         case CUDA_R_64F:
//             size = sizeof(double);
//             break;
//         case CUDA_C_64F:
//             size = sizeof(complex<double>);
//             break;
//         case CUDA_R_32F:
//             size = sizeof(float);
//             break;
//         case CUDA_C_32F:
//             size = sizeof(complex<float>);
//             break;
//         case CUDA_R_16BF:
//             size = sizeof(float);
//             break;
//         case CUDA_C_16BF:
//             size = sizeof(complex<float>);
//             break;
//         case CUDA_R_16F:
//             size = sizeof(float);
//             break;
//         case CUDA_C_16F:
//             size = sizeof(complex<float>);
//             break;
//         case CUDA_R_8F_E4M3:
//             size = sizeof(float);
//             break;
//         case CUDA_R_8F_E5M2:
//             size = sizeof(complex<float>);
//             break;
//         case CUDA_R_8I:
//             size = sizeof(__int8_t);
//             break;
//         case CUDA_C_8I:
//             size = sizeof(complex<__int8_t>);
//             break;
//         case CUDA_R_8U:
//             size = sizeof(__uint8_t);
//             break;
//         case CUDA_C_8U:
//             size = sizeof(complex<__uint8_t>);
//             break;
//         case CUDA_R_32I:
//             size = sizeof(__int32_t);
//             break;
//         case CUDA_C_32I:
//             size = sizeof(complex<__int32_t>);
//             break;
//         default:
//             size = sizeof(float);
//     }
//     return size;
// }
