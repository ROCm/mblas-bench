#include "generic_gemm.h"

#include <string>
#include <utility>

#include "third_party/cxxopts.hpp"
#include "generic_setup.h"

using std::string;

generic_gemm::generic_gemm(cxxopts::ParseResult result) {
  // Parse basic information
  m = result["m"].as<int>();
  n = result["n"].as<int>();
  k = result["k"].as<int>();

  string ldaS = result["lda"].as<string>();
  string ldbS = result["ldb"].as<string>();
  string ldcS = result["ldc"].as<string>();
  string lddS = result["ldd"].as<string>();

  // We may need these for LDX, but parse them in child implementation
  std::string tA = result["transposeA"].as<std::string>();
  std::string tB = result["transposeB"].as<std::string>();

  // Select a default LD based on OP.  See documentation here:
  // https://netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html
  lda = set_ld(ldaS, tA, m, k);
  ldb = set_ld(ldbS, tB, k, n);
  // LDC (and LDD) are always max( 1, m), so use that
  ldc = set_ld(ldcS, "N", m, 0);
  ldd = set_ld(lddS, "N", m, 0);

  // Set matrix dimensions
  std::tie(a_desc.rows, a_desc.cols) = set_row_col(tA, m, k);
  std::tie(b_desc.rows, b_desc.cols) = set_row_col(tB, k, n);
  std::tie(c_desc.rows, c_desc.cols) = set_row_col("N", m, n);
  std::tie(d_desc.rows, d_desc.cols) = set_row_col("N", m, n);

  // Set memory dimensions
  a_desc.rows_mem = lda;
  b_desc.rows_mem = ldb;
  c_desc.rows_mem = ldc;
  d_desc.rows_mem = ldd;
  std::tie(std::ignore, a_desc.cols_mem) = set_row_col(tA, m, k);
  std::tie(std::ignore, b_desc.cols_mem) = set_row_col(tB, k, n);
  std::tie(std::ignore, c_desc.cols_mem) = set_row_col("N", m, n);
  std::tie(std::ignore, d_desc.cols_mem) = set_row_col("N", m, n);

  strided = false;
  batched = false;
  function = result["function"].as<string>();

  iters = result["iters"].as<int>();
  cold_iters = result["cold_iters"].as<int>();

  batch_count = result["batch_count"].as<int>();
  if (function.find("Batched") != string::npos || function.find("batched") != string::npos || batch_count > 1 ) {
    batched = true;
    pure_batched = true;
  }
  if (function.find("Strided") != string::npos || function.find("strided") != string::npos || (function.find("matmul") != string::npos && batch_count > 1)) {
    // all batched matmuls are strided
    strided = true;
    batched = true;
    pure_batched = false;
  }
  //stride_a = result["stride_a"].as<long long int>();
  //stride_b = result["stride_b"].as<long long int>();
  //stride_c = result["stride_c"].as<long long int>();
  //stride_d = result["stride_d"].as<long long int>();
  if (strided) {
    stride_a = fix_stride(result["stride_a"].as<long long int>(), rows_mem_a, cols_mem_a, "A");
    stride_b = fix_stride(result["stride_b"].as<long long int>(), rows_mem_b, cols_mem_b, "B");
    stride_c = fix_stride(result["stride_c"].as<long long int>(), rows_mem_c, cols_mem_c, "C");
    stride_d = fix_stride(result["stride_d"].as<long long int>(), rows_mem_d, cols_mem_d, "D");
  } 

  flush_batch_count = result["flush_batch_count"].as<int>();
  flush_memory_size = result["flush_memory_size"].as<int>();

  initialization = result["initialization"].as<string>();
  scale_init = result["scale_init"].as<string>();
  filename_a = result["filename_a"].as<string>();
  filename_b = result["filename_b"].as<string>();
  filename_c = result["filename_c"].as<string>();

  constant_a = result["constant_a"].as<float>();
  constant_b = result["constant_b"].as<float>();
  constant_c = result["constant_c"].as<float>();
  constant_d = result["constant_d"].as<float>();

  scale_mode_a = set_scale_mode(result["scale_mode_a"].as<string>());
  scale_mode_b = set_scale_mode(result["scale_mode_c"].as<string>());
  scale_mode_c = set_scale_mode(result["scale_mode_b"].as<string>());
  scale_mode_d = set_scale_mode(result["scale_mode_d"].as<string>());

  scale_factor_a = result["scale_factor_a"].as<float>();
  scale_factor_b = result["scale_factor_b"].as<float>();
  scale_factor_c = result["scale_factor_c"].as<float>();
  scale_factor_d = result["scale_factor_d"].as<float>();

  a_desc.init = set_init(a_desc, result["initialization"].as<string>(), result["mx_init"].as<string>());
  b_desc.init = set_init(c_desc, result["initialization"].as<string>(), result["mx_init"].as<string>());
  c_desc.init = set_init(b_desc, result["initialization"].as<string>(), result["mx_init"].as<string>());
  d_desc.init = set_init(d_desc, result["initialization"].as<string>(), result["mx_init"].as<string>());

  // Set init control information
  if (initialization == "rand_int") {
    control_b = true;
  } else if (initialization == "trig_float") {
    control_a = true;
  }
}

int generic_gemm::set_ld(std::string ld, std::string OP, int x, int y) {
  // Use user specified value
  if (ld != "") {
    return stoi(ld);
  }
  if (OP == "N") {
    return x;
  } else {
    return y;
  }
}

long long int generic_gemm::fix_stride(long long int stride, long rows_x, long cols_x, std::string matrix_id) {
  long long rows_x_long = rows_x;
  long long cols_x_long = cols_x;
  long long stride_x = rows_x * cols_x;
  if (stride == 0) {
    std::cout << "Note: Matrix " << matrix_id << "'s stride automatically set to " << stride_x << std::endl;
    return stride_x;
  } else if (stride < stride_x) {
    std::cout << "Note: Matrix " << matrix_id << "'s stride of " << stride << " is too small, overridden to " << stride_x << std::endl;
    return stride_x;
  }
  return stride;

}

std::pair<int, int> generic_gemm::set_row_col(std::string OP, int d1, int d2) {
  if (OP == "N") {
    return std::pair<int, int>(d1, d2);
  } else {
    return std::pair<int, int>(d2, d1);
  }
}

void generic_gemm::set_flush_batch_count(
                      int a_type_size,  int b_type_size, int c_type_size, int d_type_size, 
                      int a_type_packing,  int b_type_packing, int c_type_packing, int d_type_packing,
                      bool inplace) {
  // test
  uint64_t single_block_size = calculate_offsets(rows_mem_a, cols_mem_a, rows_mem_b, cols_mem_b, rows_mem_c, cols_mem_c, rows_mem_d, cols_mem_d, 
                    a_type_size, b_type_size, c_type_size, d_type_size,
                    a_type_packing, b_type_packing, c_type_packing, d_type_packing, batch_count, inplace);
  uint64_t flush_memory_size_bytes = (uint64_t)flush_memory_size * 1024 * 1024;
  if (flush_memory_size == 0) {
    // Not specified, return
    return;
  } 

  int new_flush_batch_count = flush_memory_size_bytes / single_block_size;
  if (new_flush_batch_count == 0) {
    std::cerr << "Note: Unable to set flush_batch_count from flush_memory_size (rotating). "
    "Problem does not fit into memory size of " << flush_memory_size << "MiB" << std::endl;
  } else if (new_flush_batch_count > std::max(cold_iters, iters)) {
    flush_batch_count = std::max(cold_iters, iters);
    std::cout << "Note: flush_batch_count reduced from " << new_flush_batch_count << " to " << flush_batch_count << " to avoid excessive memory allocation." << std::endl;
  } else {
    flush_batch_count = new_flush_batch_count;
  } 
  std::cout << "Using flush_batch_count = " << flush_batch_count << std::endl;
}


scaling_type generic_gemm::set_scale_mode(string value) {
  // Is this a digit or a word?
  bool is_number = std::find_if(value.begin(), value.end(), ::isdigit) != value.end();
  if (is_number) {
    switch (std::stoi(value)) {
      //0 = none, 1 = scalar, 2 = vector, 3 = block,
      case 0:
        return scaling_type::None;
        break;
      case 1: 
        return scaling_type::Scalar;
        break;
      case 2:
        return scaling_type::Vector;
        break;
      case 3:
        return scaling_type::Block;
        break;
    }
  } else {
    string lower_val;
    std::transform(value.begin(), value.end(), lower_val.begin(),
    [](unsigned char c){ return std::tolower(c); });
    if (lower_val == "none") {
      return scaling_type::None;
    } else if (lower_val == "scalar") {
      return scaling_type::Scalar;
    } else if (lower_val == "vector") {
      return scaling_type::Vector;
    } else if (lower_val == "block") {
      return scaling_type::Block;
    }
  }
  return scaling_type::None;
}


std::string generic_gemm::set_init(matrix_desc desc, std::string init, std::string mx_init) {
  // Set init if datatype is using
  if (mx_init != "" || desc.scale_mode != scaling_type::Block) {
    // Default to regular init if mx_init isn't specified or the scaling mode isn't block
    return init;
  }
  return mx_init;

}
//void generic_gemm::set_init_params(){
//  if (initialization == "rand_int") {
//    control_b = true;
//  } else if (initialization == "trig_float") {
//    control_a = true;
//    if ()
//  }
//}
