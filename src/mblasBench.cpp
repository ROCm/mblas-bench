#include <assert.h>
#include <cxxabi.h>
#include <stdlib.h>
#include <unistd.h>

#include <cctype>
#include <iostream>


//#include "genericGemm.h"
//#include "rocblasGemm.h"
//#include "hipblasLtGemm.h"
//#include "cublasGemm.h"
//#include "cublasLtGemm.h"
#include <genericGemmFactory.h>
#include <rocblasGemmFactory.h>
#include <hipblasLtGemmFactory.h>
#include <cublasGemmFactory.h>
#include <cublasLtGemmFactory.h>

#include "third_party/cxxopts.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::flush;
using std::string;

std::string s_to_lower(std::string data) {
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
  opp_adder("transposeA,transA", "transposeA",
            cxxopts::value<string>()->default_value("N"));
  opp_adder("transposeB,transB", "transposeB",
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
  opp_adder("composite_compute_type",
            "rocblas-bench compatibility variable, maps directly to --compute_type",
            cxxopts::value<string>()->default_value(""));
  opp_adder("scalar_type",
            "What scalar type to use "
            "Defaults to a value based on -r/--precision when not specified",
            cxxopts::value<string>()->default_value(""));
  opp_adder("batch_count",
            "Number of matrices. Only applicable to batched and "
            "strided_batched routines",
            cxxopts::value<int>()->default_value("1"));
  opp_adder("flush_batch_count",
            "number of copies of arrays to allocate for cache flushing in timing code."
            "Functions are called iters times in a timing loop. If the problem memory "
            "footprint is small enough, then arrays will be cached. If you specify "
            "flush_batch_count you cannot also specify flush_memory_size or rotating",
            cxxopts::value<int>()->default_value("1"));
  opp_adder("flush_memory_size,rotating",
            "bytes of memory that will be occupied by arrays. Used only in "
            "timing code for cache flushing. Set to greater than cache size so arrays "
            "are flushed from cache before they are reused."
            "If you specify flush_memory_size or rotating you cannot also specify flush_batch_count",
            cxxopts::value<int>()->default_value("0"));
  // opp_adder("block_count",
  //           "Number of memory blocks for arrays. Each benchmarking iteration "
  //           "will use the next block of memory (or loop to the first block)",
  //           cxxopts::value<int>()->default_value("1"));
  opp_adder("device", "GPU device(s) to run on",
            cxxopts::value<string>()->default_value("0"));
  opp_adder("instances", "Number of instances to run on each GPU",
            cxxopts::value<int>()->default_value("1"));
  opp_adder("initialization",
            "Intialize with random integers, trig functions sin and cos, or "
            "hpl-like input. Options: rand_int, trig_float, normal_float, "
            "hpl, blasgemm",
            cxxopts::value<string>()->default_value("rand_int"));
  opp_adder("filename_a",
            "Intialize matrix A with contents of a csv file",
            cxxopts::value<string>()->default_value(""));
  opp_adder("filename_b",
            "Intialize matrix B with contents of a csv file",
            cxxopts::value<string>()->default_value(""));
  opp_adder("filename_c",
            "Intialize matrix C with contents of a csv file",
            cxxopts::value<string>()->default_value(""));
  opp_adder("constant_a",
            "Constant value used for the A matrix."
            "In constant init mode, this value is used as the value to initialized"
            "In trig_float init mode, this value is used as a scale factor",
            cxxopts::value<float>()->default_value("1"));
  opp_adder("constant_b",
            "Constant value used for the B matrix."
            "In constant init mode, this value is used as the value to initialized"
            "In trig_float init mode, this value is used as a scale factor",
            cxxopts::value<float>()->default_value("1"));
  opp_adder("constant_c",
            "Constant value used for the C matrix."
            "In constant init mode, this value is used as the value to initialized"
            "In trig_float init mode, this value is used as a scale factor",
            cxxopts::value<float>()->default_value("1"));
  opp_adder("constant_d",
            "Constant value used for the D matrix."
            "In constant init mode, this value is used as the value to initialized"
            "In trig_float init mode, this value is used as a scale factor",
            cxxopts::value<float>()->default_value("1"));
  opp_adder("scale_mode_a,scaleA",
            "Scale mode for A matrix.  Accepts both numeric and text input"
            "0 = none, 1 = scalar, 2 = vector, 3 = block",
            cxxopts::value<float>()->default_value("0"));
  opp_adder("scale_mode_b,scaleB",
            "Scale mode for B matrix.  Accepts both numeric and text input"
            "0 = none, 1 = scalar, 2 = vector, 3 = block",
            cxxopts::value<float>()->default_value("0"));
  opp_adder("scale_mode_c,scaleC",
            "Scale mode for C matrix.  Accepts both numeric and text input"
            "0 = none, 1 = scalar, 2 = vector, 3 = block",
            cxxopts::value<float>()->default_value("0"));
  opp_adder("scale_mode_d,scaleD",
            "Scale mode for D matrix.  Accepts both numeric and text input"
            "0 = none, 1 = scalar, 2 = vector, 3 = block",
            cxxopts::value<float>()->default_value("0"));
  opp_adder("scale_factor_a",
            "Scale factor for A matrix.",
            cxxopts::value<float>()->default_value("1"));
  opp_adder("scale_factor_b",
            "Scale factor for B matrix.",
            cxxopts::value<float>()->default_value("1"));
  opp_adder("scale_factor_c",
            "Scale factor for C matrix.",
            cxxopts::value<float>()->default_value("1"));
  opp_adder("scale_factor_d",
            "Scale factor for D matrix.",
            cxxopts::value<float>()->default_value("1"));
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

  genericGemmFactory *gemm;
  // Select backend implementation
  string driver = s_to_lower(result["driver"].as<string>());
  string function = s_to_lower(result["function"].as<string>());

  if (driver == "cublaslt" || (driver == "cublas" && function == "matmul")) {
    // Since regular cublas has no matmul, we can safely assume the user means
    // cublaslt
    gemm = new cublasLtGemmFactory();
  } else if (driver == "cublas-bench" || driver == "cublas") {
    gemm = new cublasGemmFactory();
  } else if (driver == "hipblaslt" || (driver == "rocblas" && function == "matmul")) {
    // Since regular rocblas has no matmul, we can safely assume the user means
    // hipblaslt
    // gemm = new hipblasLtGemm(result);
    gemm = new hipblasLtGemmFactory();
  } else if (driver == "rocblas-bench" || driver == "rocblas") {
    gemm = new rocblasGemmFactory();
  } else {
    cerr << "Driver \"" << driver << "\" not supported" << endl;
    return 1;
  }

  gemm->create_gemm(result);
  string header = gemm->prepare_array();
  cout << header << flush;
  gemm->test();
  cout << std::fixed;

  string results = gemm->get_result_string();
  cout << results << flush;

  gemm->free_mem();
  delete gemm;

  return 0;
}
