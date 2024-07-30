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

void *allocateHostArr(mblasDataType type, long x, long y, int batch) {
  int typesize = typeCallHost<sizeofCUDT>(type);
  void *data = (void *)malloc(x * y * batch * typesize);
  return data;
}

void *allocateHostArr(hipblasDatatype_t type, long x, long y, int batch) {
  int typesize = typeCallHost<sizeofCUDT>(type);
  void *data = (void *)malloc(x * y * batch * typesize);
  return data;
}

void *allocateDevArr(mblasDataType type, long x, long y, int batch) {
  int typesize = typeCallDev<sizeofCUDT>(type);
  void *data;
  checkHip(hipMalloc(&data, x * y * batch * typesize));
  return data;
}

void *allocateDevArr(hipblasDatatype_t type, long x, long y, int batch) {
  int typesize = typeCallDev<sizeofCUDT>(type);
  void *data;
  checkHip(hipMalloc(&data, x * y * batch * typesize));
  return data;
}

void *allocateHDevArr(mblasDataType type, long x, long y, int batch) {
  int typesize = typeCallHost<sizeofCUDT>(type);
  void *data;
  checkHip(hipMalloc(&data, x * y * batch * typesize));
  return data;
}

void *allocateHDevArr(hipblasDatatype_t type, long x, long y, int batch) {
  int typesize = typeCallHost<sizeofCUDT>(type);
  void *data;
  checkHip(hipMalloc(&data, x * y * batch * typesize));
  return data;
}

// void *allocateScalar(hipDataType type) {
//  int typesize = typeCallDev<sizeofCUDT>(type);
//  void *scalar = (void *)
//}

// void dummy() {
//   // This function forces the compiler to generate the needed templated variants
//   // of each function. It is never called
//   void *h_A;
//   typeCallHost<sizeofCUDTP>(hipDataType_f64_r);
//   typeCallHost<allocSetScalar>(hipDataType_f64_r, "1", "0");
//   typeCallDev<batchedPtrMagic>(hipDataType_f64_r, (void **)NULL, (void **)NULL,
//                                (void *)NULL, 10, 10, 10);
//   typeCallHost<sizeofCUDTP>(HIPBLAS_R_32F);
//   typeCallHost<allocSetScalar>(HIPBLAS_R_32F, "1", "0");
//   typeCallDev<batchedPtrMagic>(HIPBLAS_R_32F, (void **)NULL, (void **)NULL,
//                                (void *)NULL, 10, 10, 10);
//   // template void *allocSetScalar<double>::operator()(string);
// }

// void initHostH(hipDataType precision, std::string initialization, void *ptr,
//                int rows_A, int cols_A, int ld, int batch, long long int stride,
//                float constant, bool alternating) {
//   typeCallHost<initHost>(precision, initialization, ptr, rows_A, cols_A, ld,
//                          batch, stride, constant, alternating);
// }
// 
// void initHostH(hipblasDatatype_t precision, std::string initialization, void *ptr,
//                int rows_A, int cols_A, int ld, int batch, long long int stride,
//                float constant, bool alternating) {
//   typeCallHost<initHost>(precision, initialization, ptr, rows_A, cols_A, ld,
//                          batch, stride, constant, alternating);
// }
// 
// template <typename T>
// void initHost<T>::operator()(std::string initialization, void *ptr, int rows_A,
//                              int cols_A, int ld, int batch,
//                              long long int stride, bool control,
//                              float constant, std::string filename) {
//   if (!filename.empty()) {
//     fillRandHostFromCSV<T>(ptr, rows_A, cols_A, ld, batch, stride, filename);
//   } else if (initialization == "rand_int") {
//     fillRandHostRandIntAS<T>(ptr, rows_A, cols_A, ld, batch, stride, control);
//   } else if (initialization == "trig_float") {
//     fillRandHostTrigFloat<T>(ptr, rows_A, cols_A, ld, batch, stride, control);
//   } else if (initialization == "normal_float") {
//     fillRandHostNormalFloat<T>(ptr, rows_A, cols_A, ld, batch, stride);
//   } else if (initialization == "hpl") {
//   } else if (initialization == "blasgemm") {
//     fillRandHostBlasgemm<T>(ptr, rows_A, cols_A, ld, batch, stride);
//   } else if (initialization == "constant") {
//     fillRandHostConstant<T>(ptr, rows_A, cols_A, ld, batch, stride, constant);
//   }
// }

// template void initHost<double>::operator()(std::string, void *, int, int, int,
//                                            int, long long int, bool, float, std::string);
// template void initHost<hipDoubleComplex>::operator()(std::string, void *, int,
//                                                     int, int, int,
//                                                     long long int, int, bool, float, std::string);
// template void initHost<float>::operator()(std::string, void *, int, int, int,
//                                           int, long long int, bool, float, std::string);
// template void initHost<hipComplex>::operator()(std::string, void *, int,
//                                                    int, int, int, long long int,
//                                                    bool, float, std::string);
// template void initHost<__int8_t>::operator()(std::string, void *, int, int, int,
//                                              int, long long int, bool, float, std::string);
// template void initHost<complex<__int8_t>>::operator()(std::string, void *, int,
//                                                       int, int, int,
//                                                       long long int, int, bool,
//                                                       float, std::string);
// template void initHost<__uint8_t>::operator()(std::string, void *, int, int,
//                                               int, int, long long int, bool,
//                                               float, std::string);
// template void initHost<complex<__uint8_t>>::operator()(std::string, void *, int,
//                                                        int, int, int,
//                                                        long long int, int, bool,
//                                                        float, std::string);
// template void initHost<__int32_t>::operator()(std::string, void *, int, int,
//                                               int, int, long long int, bool,
//                                               float, std::string);
// template void initHost<complex<__int32_t>>::operator()(std::string, void *, int,
//                                                        int, int, int,
//                                                        long long int, int, bool,
//                                                        float, std::string);

// Instances of the following functions should be defined implicitly by defining
// those of initHost
// template <typename T>
// void fillRandHostBlasgemm(void *ptr, int rows_A, int cols_A, int ld, int batch,
//                           long long int stride) {
//   int a = 1;
//   T *A = (T *)ptr;
//   for (size_t i = 0; i < rows_A * cols_A * batch; i++) {
//     A[i] = (T)rand() / (T)(RAND_MAX / a);
//   }
// }
// 
// template <typename T>
// void fillRandHostConstant(void *ptr, int rows_A, int cols_A, int ld, int batch,
//                           long long int stride, float constant) {
//   int a = 1;
//   T *A = (T *)ptr;
//   for (size_t i = 0; i < rows_A * cols_A * batch; i++) {
//     A[i] = (T)(constant);
//   }
// }
// 
// template <typename T>
// void fillRandHostFromCSV(void *ptr, int rows_A, int cols_A, int ld, int batch,
//                          long long int stride, std::string filename) {
//   std::ifstream file(filename);
//   std::vector<std::vector<T>> result;
// 
//   for (string line; getline(file, line, '\n'); ) {
//     result.push_back(std::vector<T>());
//     std::istringstream ss(line);
//     for (string field; getline(ss, field, ','); ) {
//       result.back().push_back((T)std::stod(field));
//     }
//     if (result.back().empty()) {
//       std::cout << "warning: empty row in csv file" << std::endl;
//     }
//   }
//   if (result.empty()) {
//     std::cout << "warning: csv file is empty" << std::endl;
//   }
// 
//   T *A = (T *)ptr;
//   size_t n_rows = result.size();
//   for (size_t i_batch = 0; i_batch < batch; i_batch++) {
//     for (size_t j = 0; j < cols_A; ++j) {
//       size_t offset = j * ld + i_batch * stride;
//       for (size_t i = 0; i < rows_A; ++i) {
//         size_t n_cols = result[i % n_rows].size();
//         A[i + offset] = result[i % n_rows][j % n_cols];
//       }
//     }
//   }
// }
// 
// template <typename T>
// void fillRandHostRandIntAS(void *ptr, int rows_A, int cols_A, int ld, int batch,
//                            long long int stride, bool alternating) {
//   std::random_device r;
//   std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
//   std::mt19937 gen(seed);
//   std::uniform_int_distribution<int> uniform_dist(1, 10);
//   T *A = (T *)ptr;
//   T dummy;
//   for (size_t i_batch = 0; i_batch < batch; i_batch++) {
//     for (size_t j = 0; j < cols_A; ++j) {
//       size_t offset = j * ld + i_batch * stride;
//       for (size_t i = 0; i < rows_A; ++i) {
//         if ((!alternating) || (j % 2 ^ i % 2)) {
//           A[i + offset] = randIntGen(uniform_dist, gen, dummy);
//         } else {
//           A[i + offset] = randIntGenN(uniform_dist, gen, dummy);
//         }
//       }
//     }
//   }
// }
// 
// template <typename T>
// void fillRandHostTrigFloat(void *ptr, int rows_A, int cols_A, int ld, int batch,
//                            long long int stride, bool isSin) {
//   T *A = (T *)ptr;
//   for (size_t i_batch = 0; i_batch < batch; i_batch++) {
//     for (size_t j = 0; j < cols_A; ++j) {
//       size_t offset = j * ld + i_batch * stride;
//       for (size_t i = 0; i < rows_A; ++i) {
//         if (isSin) {
//           A[i + offset] = T(sin(i + offset));
//         } else {
//           A[i + offset] = T(cos(i + offset));
//         }
//       }
//     }
//   }
// }
// 
// template <typename T>
// void fillRandHostNormalFloat(void *ptr, int rows_A, int cols_A, int ld, int batch,
//                              long long int stride) {
//   std::random_device r;
//   std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
//   std::mt19937 gen(seed);
//   std::normal_distribution<double> normal_dist(5.0, 2.0);
//   T *A = (T *)ptr;
//   T dummy;
//   for (size_t i_batch = 0; i_batch < batch; i_batch++) {
//     for (size_t j = 0; j < cols_A; ++j) {
//       size_t offset = j * ld + i_batch * stride;
//       for (size_t i = 0; i < rows_A; ++i) {
//         A[i + offset] = normalFloatGen(normal_dist, gen, dummy);
//       }
//     }
//   }
// }

// template <typename T>
// inline T randIntGen(std::uniform_int_distribution<int> &idist,
//                     std::mt19937 &gen, T &dummy) {
//   return T(idist(gen));
// }

// template <typename T>
// inline complex<T> randIntGen(std::uniform_int_distribution<int> &idist,
//                              std::mt19937 &gen, complex<T> &dummy) {
//   return {T(idist(gen)), T(idist(gen))};
// }

// template <typename T>
// inline T randIntGenN(std::uniform_int_distribution<int> &idist,
//                      std::mt19937 &gen, T &dummy) {
//   return -T(idist(gen));
// }

// template <typename T>
// inline complex<T> randIntGenN(std::uniform_int_distribution<int> &idist,
//                               std::mt19937 &gen, complex<T> &dummy) {
//   return {-T(idist(gen)), -T(idist(gen))};
// }

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

// template <typename T>
// inline T normalFloatGen(std::normal_distribution<double> &ndist,
//                         std::mt19937 &gen, T &dummy) {
//   return T(ndist(gen));
// }