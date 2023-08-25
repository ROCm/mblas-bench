#include <assert.h>
#include <cxxabi.h>
#include <stdlib.h>
#include <unistd.h>

#include <cctype>
#include <iostream>


#include "genericGemm.h"
#include "rocblasGemm.h"
#include "cublasGemm.h"
#include "cublasLtGemm.h"

#include "third_party/cxxopts.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::flush;
using std::string;

std::string sToLower(std::string data) {
  std::transform(data.begin(), data.end(), data.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return data;
}

int main(int argc, char **argv) {
  // print device info
  // int num_devices;
  // cudaGetDeviceCount(&num_devices);
  // for (int i = 0; i < num_devices; i++) {
  //   cudaDeviceProp prop;
  //   cudaGetDeviceProperties(&prop, i);
  //   std::cout << "Device " << i << ": " << prop.name << ", "
  //             << prop.clockRate / 1000 << " MHZ"
  //             << ", " << prop.memoryClockRate / 1000 << " MHZ" << std::endl;
  // }
  // parse input arguments
  cxxopts::Options options("rocblas_bench", "Benchmark rocBLAS");
  string supPrec = "h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r";
  auto opp_adder = options.add_options();
  opp_adder("m,sizem", "Specific matrix size",
            cxxopts::value<int>()->default_value("128"));
  opp_adder("n,sizen", "Specific matrix size",
            cxxopts::value<int>()->default_value("128"));
  opp_adder("k,sizek", "Specific matrix size",
            cxxopts::value<int>()->default_value("128"));
  opp_adder("f,function", "BLAS function to test", cxxopts::value<string>());
  opp_adder("r,precision", "Precision. Options: " + supPrec + " ",
            cxxopts::value<string>()->default_value("f32_r"));
  opp_adder("transposeA", "transposeA",
            cxxopts::value<string>()->default_value("N"));
  opp_adder("transposeB", "transposeB",
            cxxopts::value<string>()->default_value("N"));
  opp_adder("alpha", "specifies the scalar alpha  (Default value is: 1)",
            cxxopts::value<string>()->default_value("1"));
  opp_adder(
      "alphai",
      "specifies the imaginary part of the scalar alpha  (Default value is: 0)",
      cxxopts::value<string>()->default_value("0"));
  opp_adder("beta", "specifies the scalar beta  (Default value is: 0)",
            cxxopts::value<string>()->default_value("0"));
  opp_adder(
      "betai",
      "specifies the imaginary part of the scalar beta  (Default value is: 0)",
      cxxopts::value<string>()->default_value("0"));
  opp_adder("lda",
            "Leading dimension of matrix A, is only applicable to BLAS-2 & "
            "BLAS-3.  (Default value is based on size of A if not specified)",
            cxxopts::value<string>()->default_value(""));
  opp_adder("ldb",
            "Leading dimension of matrix B, is only applicable to BLAS-2 & "
            "BLAS-3.  (Default value is based on size of B if not specified)",
            cxxopts::value<string>()->default_value(""));
  opp_adder("ldc",
            "Leading dimension of matrix C, is only applicable to BLAS-2 & "
            "BLAS-3.  (Default value is based on size of C if not specified)",
            cxxopts::value<string>()->default_value(""));
  opp_adder("ldd",
            "Leading dimension of matrix D, is only applicable to BLAS-EX.  "
            "(Default value is based on size of D if not specified)",
            cxxopts::value<string>()->default_value(""));
  opp_adder("stride_a",
            "Specific stride of strided_batched matrix A, is only applicable "
            "to strided batchedBLAS-2 and BLAS-3: second dimension * leading "
            "dimension.  (Default value is: 16384)",
            cxxopts::value<long long int>()->default_value("16384"));
  opp_adder("stride_b",
            "Specific stride of strided_batched matrix B, is only applicable "
            "to strided batchedBLAS-2 and BLAS-3: second dimension * leading "
            "dimension.  (Default value is: 16384)",
            cxxopts::value<long long int>()->default_value("16384"));
  opp_adder("stride_c",
            "Specific stride of strided_batched matrix C, is only applicable "
            "to strided batchedBLAS-2 and BLAS-3: second dimension * leading "
            "dimension.  (Default value is: 16384)",
            cxxopts::value<long long int>()->default_value("16384"));
  opp_adder("stride_d",
            "Specific stride of strided_batched matrix D, is only applicable "
            "to strided batchedBLAS_EX: second dimension * leading dimension.  "
            "(Default value is: 16384)",
            cxxopts::value<long long int>()->default_value("16384"));
  opp_adder("a_type",
            "Precision of matrix A. Options:" + supPrec + ". " +
                "Defaults to the value of -r/--precision",
            cxxopts::value<string>()->default_value(""));
  opp_adder("b_type",
            "Precision of matrix B. Options:" + supPrec + ". " +
                "Defaults to the value of -r/--precision",
            cxxopts::value<string>()->default_value(""));
  opp_adder("c_type",
            "Precision of matrix C. Options:" + supPrec + ". " +
                "Defaults to the value of -r/--precision",
            cxxopts::value<string>()->default_value(""));
  opp_adder("d_type",
            "Precision of matrix D. Options:" + supPrec + ". " +
                "Defaults to the value of -r/--precision",
            cxxopts::value<string>()->default_value(""));
  opp_adder("compute_type",
            "What gemm kernel to use for the gemmEx family of functions"
            "Defaults to a value based on -r/--precision when not specified",
            cxxopts::value<string>()->default_value(""));
  opp_adder("scalar_type",
            "What scalar type to use "
            "Defaults to a value based on -r/--precision when not specified",
            cxxopts::value<string>()->default_value(""));
  opp_adder("batch_count",
            "Number of matrices. Only applicable to batched and "
            "strided_batched routines  (Default value is: 1)",
            cxxopts::value<int>()->default_value("1"));
  opp_adder("num_blocks",
            "Number of memory blocks for arrays. Each benchmarking iteration "
            "will use the next block of memory (or loop to the first block)",
            cxxopts::value<int>()->default_value("1"));
  opp_adder("device", "GPU device(s) to run on",
            cxxopts::value<string>()->default_value("0"));
  opp_adder("instances", "Number of instances to run on each GPU",
            cxxopts::value<int>()->default_value("1"));
  opp_adder("initialization",
            "Intialize with random integers, trig functions sin and cos, or "
            "hpl-like input. Options: rand_int, trig_float, normal_float, "
            "hpl, blasgemm",
            cxxopts::value<string>()->default_value("rand_int"));
  opp_adder("filenameA",
            "Intialize matrix A with contents of a csv file",
            cxxopts::value<string>()->default_value(""));
  opp_adder("filenameB",
            "Intialize matrix B with contents of a csv file",
            cxxopts::value<string>()->default_value(""));
  opp_adder("filenameC",
            "Intialize matrix C with contents of a csv file",
            cxxopts::value<string>()->default_value(""));
  opp_adder("i,iters",
            "Iterations to run inside timing loop  (Default value is: 10)",
            cxxopts::value<int>()->default_value("10"));
  opp_adder("j,cold_iters",
            " Cold Iterations to run before entering the timing loop ",
            cxxopts::value<int>()->default_value("2"));
  opp_adder("driver", "Backend to run the GEMM test with",
            cxxopts::value<string>()->default_value("rocblas"));
  opp_adder("h,help", "Print Usage");

  cxxopts::ParseResult result = options.parse(argc, argv);

  if (result.count("help")) {
    cout << options.help() << endl;
    exit(0);
  }

  genericGemm *gemm;

  // Select backend implementation
  string driver = sToLower(result["driver"].as<string>());
  string function = sToLower(result["function"].as<string>());

  if (driver == "cublaslt" || (driver == "cublas" && function == "matmul")) {
    // Since regular cublas has no matmul, we can safely assume the user means
    // cublaslt
    gemm = new cublasLtGemm(result);
  } else if (driver == "cublas-bench" || driver == "cublas") {
    gemm = new cublasGemm(result);
  // if (driver == "hipblaslt" || (driver == "rocblas" && function == "matmul")) {
  //   // Since regular rocblas has no matmul, we can safely assume the user means
  //   // hipblaslt
  //   gemm = new cublasLtGemm(result);
  // } else 
  } else if (driver == "rocblas-bench" || driver == "rocblas") {
    gemm = new rocblasGemm(result);
  } else {
    cerr << "Driver \"" << driver << "\" not supported" << endl;
    return 1;
  }

  string header = gemm->prepareArray();
  cout << header << flush;
  gemm->test();
  cout << std::fixed;

  string results = gemm->getResultString();
  cout << results << flush;

  gemm->freeMem();
  delete gemm;

  return 0;
}
