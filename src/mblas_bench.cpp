#include <assert.h>
#include <cxxabi.h>
#include <stdlib.h>
#include <unistd.h>

#include <cctype>
#include <iostream>
#include <fstream>


//#include "generic_gemm.h"
//#include "rocblas_gemm.h"
//#include "hipblaslt_gemm.h"
//#include "cublas_gemm.h"
//#include "cublaslt_gemm.h"
#include <generic_gemm_factory.h>
#include <rocblas_gemm_factory.h>
#include <hipblaslt_gemm_factory.h>
#include <cublas_gemm_factory.h>
#include <cublaslt_gemm_factory.h>

#include "third_party/cxxopts.hpp"
#include <yaml-cpp/yaml.h>

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


std::vector<cxxopts::ParseResult> parse_yaml_file(const std::string& filename, const cxxopts::Options& opts, int argc, char** argv)
{
    // Read the YAML file and parse each line
    // For each parsed line, convert it to command-line arguments and parse with cxxopts so that the original code can remain unchanged

    std::vector<cxxopts::ParseResult> results;
    YAML::Node config = YAML::LoadFile(filename);
    for (const auto& node : config) {
        if (node.IsMap()) {
            std::vector<std::string> args;
            for (int i = 0; i < argc; ++i) {
                args.push_back(argv[i]);
            }
            for (const auto& it : node) {
                std::string key = it.first.as<std::string>();
                std::string value = it.second.as<std::string>();

                // m,n,k can be uppercase in YAML, but should be lowercase in command-line arguments
                if (key == "M") {
                    key = "m";
                } else if (key == "N") {
                    key = "n";
                } else if (key == "K") {
                    key = "k";
                }

                const std::string arg_key = key.size() == 1 ? "-" + key : "--" + key;
                args.push_back(arg_key);
                args.push_back(value);
            }
            std::vector<const char*> cstr_args;
            for (const auto& arg: args) {
                cstr_args.push_back(arg.c_str());
            }
            // Copy opts just in case member variables are modified during parsing
            cxxopts::Options opts_copy = opts;
            opts_copy = opts_copy.allow_unrecognised_options();
            auto result = opts_copy.parse(static_cast<int>(cstr_args.size()), cstr_args.data());
            results.push_back(result);

            // Print unmatched arguments
            const auto &unmatched = result.unmatched();
            const std::string YELLOW = "\033[33m";
            const std::string RESET = "\033[0m";
            if (!unmatched.empty()) {
              std::cout << YELLOW << "Ignored (unmatched) arguments:";
              for (const auto &arg: unmatched) {
                std::cout << " " << arg;
              }
              std::cout << RESET << std::endl;
            }
            // For debugging. All parsed arguments from YAML
            if (false) {
                std::cout << "Parsed arguments from YAML:" << std::endl;
                std::cout << result.arguments_string() << std::endl;
            }
        }
    }
    return results;
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
            "to strided batched gemm and matmuls: second dimension * leading "
            "dimension.  A value of 0 means automatic based on memory size",
            cxxopts::value<long long int>()->default_value("0"));
  opp_adder("stride_b",
            "Specific stride of strided_batched matrix B, is only applicable "
            "to strided batched gemm and matmuls: second dimension * leading "
            "dimension.  A value of 0 means automatic based on memory size",
            cxxopts::value<long long int>()->default_value("0"));
  opp_adder("stride_c",
            "Specific stride of strided_batched matrix C, is only applicable "
            "to strided batched gemm and matmuls: second dimension * leading "
            "dimension.  A value of 0 means automatic based on memory size",
            cxxopts::value<long long int>()->default_value("0"));
  opp_adder("stride_d",
            "Specific stride of strided_batched matrix D, is only applicable "
            "to strided batched gemm and matmuls: second dimension * leading "
            "dimension.  A value of 0 means automatic based on memory size",
            cxxopts::value<long long int>()->default_value("0"));
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
  opp_adder("mx_init",
            "Initialize any MX datatypes with this initialization strategy."
            "Defaults to the value of initialization"
            "Options: rand_int, trig_float, normal_float, hpl, blasgemm",
            cxxopts::value<string>()->default_value(""));
  opp_adder("scale_init",
            "Initialize the block scaling factors with a particular distribution "
            "Options: constant, normal_float",
            cxxopts::value<string>()->default_value("constant"));
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
  opp_adder("yaml",
            "Use YAML file as problem input. Command line options will be overridden by YAML file input",
            cxxopts::value<string>()->default_value(""));
  opp_adder("h,help", "Print Usage");

  cxxopts::ParseResult result = options.parse(argc, argv);

  if (result.count("help")) {
    cout << options.help() << endl;
    exit(0);
  }

  
  std::vector<cxxopts::ParseResult> input_problems;
  if (result.count("yaml")) {
    string yaml_file = result["yaml"].as<string>();
    input_problems = parse_yaml_file(yaml_file, options, argc, argv);
    if (input_problems.size() == 0) {
      cerr << "No valid problems found in YAML file" << endl;
      return 1;
    }
  } else {
    input_problems.push_back(result);
  }

  for (const auto &result: input_problems)
  {

    generic_gemm_factory *gemm;
    // Select backend implementation
    string driver = s_to_lower(result["driver"].as<string>());
    string function = s_to_lower(result["function"].as<string>());

    if (driver == "cublaslt" || (driver == "cublas" && function == "matmul")) {
      // Since regular cublas has no matmul, we can safely assume the user means
      // cublaslt
      gemm = new cublaslt_gemm_factory();
    } else if (driver == "cublas-bench" || driver == "cublas") {
      gemm = new cublas_gemm_factory();
    } else if (driver == "hipblaslt" || (driver == "rocblas" && function == "matmul")) {
      // Since regular rocblas has no matmul, we can safely assume the user means
      // hipblaslt
      // gemm = new hipblaslt_gemm(result);
      gemm = new hipblaslt_gemm_factory();
    } else if (driver == "rocblas-bench" || driver == "rocblas") {
      gemm = new rocblas_gemm_factory();
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
  }


  return 0;
}
