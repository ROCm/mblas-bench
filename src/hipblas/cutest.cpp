#include <assert.h>
#include <cublas_v2.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cxxabi.h>
#include <stdlib.h>
#include <unistd.h>

#include <bitset>
#include <iostream>
#include <sstream>
// #include "fp16_conversion.h"
//#include "error_handling.h"
#include "third_party/cxxopts.hpp"
//#include "create-allocate.h"

using std::cout;
using std::endl;

template <typename T>
void t(int a, int b) {
  T c = 10.5;
  cout << a << b << c << endl;
}

void h(int a, int b) { cout << a << b << endl; }

template <class T>
struct allocateHost {
  float operator()() {
    int a = 1;
    int b = 2;
    T c = 10.5;
    cout << a << b << c << endl;
    return c;
  }
};

template <template <typename> class tFunc, class... Args>
auto typeCall(bool dbl, Args... args) ->
    typename std::result_of<tFunc<double>(Args...)>::type {
  // At runtime, determine which typed implementation to use and call it
  if (dbl)
    return tFunc<double>()(args...);
  else
    return tFunc<int>()(args...);
}

// template<typename F, F f, class... Args>
// template<typename ReturnType, typename Args...,ReturnType (*FuncPtr)(Args...)
//>
// template<typename F, std::function<F> Func, class... Args>
// template<typename... Args>
// void f(void (*wrappee)(Args...),bool dbl, Args... args){
//     if (dbl) {
//         wrappee<double>(args...);
//     } else {
//         wrappee<int>(args...);
//     }

// }

int main(int argc, char **argv) {
  /*
  cudaDataType_t test = CUDA_R_16F;
  cudaBfloat16 a = 0.1;
  typedef test gaming = 0.1;
  std::cout << gaming << std::endl;
  */
  // void * h_data1 = allocate_host_arr_single<double>(10,10);
  // void * h_data4 = allocate_host_arr_single<long>(10,10);
  // void * h_data2 = allocate_host_arr_single<float>(10,10);
  //__float16 asdf = 0.1;
  // std::bitset<16> a(asdf);
  // std::cout << a << std::endl;
  // void * h_data3 = allocate_host_arr_single<int>(10,10);
  // cout << sizeof_cudt_host(CUDA_R_64F) << endl;
  // cout << sizeof(__nv_fp8_storage_t) << endl;
  // f
  // f<void (int, int), t>(10, 10)
  // f(t, false, 10, 10);

  std::cout << abi::__cxa_demangle(typeid(cublasGemmEx_64).name(), nullptr,
                                   nullptr, nullptr)
            << std::endl;
  std::cout << abi::__cxa_demangle(typeid(cublasGemmBatchedEx_64).name(),
                                   nullptr, nullptr, nullptr)
            << std::endl;
  std::cout << abi::__cxa_demangle(typeid(cublasGemmStridedBatchedEx_64).name(),
                                   nullptr, nullptr, nullptr)
            << std::endl;
  int n1 = 1;
  int n2 = 2;
  typeCall<allocateHost>(false);  // n3 == 3
  typeCall<allocateHost>(true);
  std::istringstream iss("0.15");
  // cout << n3 << endl;
  return 0;
  // wrapper<decltype(&t), t>(10, 10);
  // auto asdf = test_arr();
}