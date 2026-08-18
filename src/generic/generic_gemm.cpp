#include "generic_gemm.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
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
  std::tie(a_props.rows, a_props.cols) = set_row_col(tA, m, k);
  std::tie(b_props.rows, b_props.cols) = set_row_col(tB, k, n);
  std::tie(c_props.rows, c_props.cols) = set_row_col("N", m, n);
  std::tie(d_props.rows, d_props.cols) = set_row_col("N", m, n);

  // Set memory dimensions
  a_props.rows_mem = lda;
  b_props.rows_mem = ldb;
  c_props.rows_mem = ldc;
  d_props.rows_mem = ldd;
  std::tie(std::ignore, a_props.cols_mem) = set_row_col(tA, m, k);
  std::tie(std::ignore, b_props.cols_mem) = set_row_col(tB, k, n);
  std::tie(std::ignore, c_props.cols_mem) = set_row_col("N", m, n);
  std::tie(std::ignore, d_props.cols_mem) = set_row_col("N", m, n);

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

  // Determine whether this GEMM accumulates into C (beta != 0). We parse
  // the beta strings here once so every backend (and set_flush_batch_count)
  // can read the resulting `accumulate` member without duplicating the
  // logic. Empty / unparseable strings are treated as 0.
  auto safe_stod = [](const string & s) -> double {
    if (s.empty()) return 0.0;
    try { return std::stod(s); } catch (...) { return 0.0; }
  };
  accumulate = (safe_stod(result["beta"].as<string>())  != 0.0) ||
               (safe_stod(result["betai"].as<string>()) != 0.0);

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
  scale_mode_b = set_scale_mode(result["scale_mode_b"].as<string>());
  scale_mode_c = set_scale_mode(result["scale_mode_c"].as<string>());
  scale_mode_d = set_scale_mode(result["scale_mode_d"].as<string>());

  scale_factor_a = result["scale_factor_a"].as<float>();
  scale_factor_b = result["scale_factor_b"].as<float>();
  scale_factor_c = result["scale_factor_c"].as<float>();
  scale_factor_d = result["scale_factor_d"].as<float>();

  a_props.init = set_init(a_props, result["initialization"].as<string>(), result["mx_init"].as<string>());
  b_props.init = set_init(b_props, result["initialization"].as<string>(), result["mx_init"].as<string>());
  c_props.init = set_init(c_props, result["initialization"].as<string>(), result["mx_init"].as<string>());
  d_props.init = set_init(d_props, result["initialization"].as<string>(), result["mx_init"].as<string>());

  requested_solution_num = result["requested_solution_num"].as<int>();
  if (requested_solution_num == 0 || requested_solution_num < -1) {
    throw std::invalid_argument("Invalid --requested_solution_num. Must be -1 (all) or a positive integer.");
  }

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
  long long stride_x = rows_x_long * cols_x_long;
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
                      uint64_t a_scale_bytes, uint64_t b_scale_bytes,
                      uint64_t c_scale_bytes, uint64_t d_scale_bytes,
                      bool inplace) {
  // Compute the rotating buffer's per-block memory footprint. Every
  // rotating block holds A, B, optionally C (only when beta != 0, just
  // like hipblaslt-bench), D (skipped when inplace shares memory with C),
  // and each backend's per-matrix scale tensors. All sizes are padded to
  // a 16-byte boundary so allocations land on aligned addresses.
  auto round16 = [](uint64_t v) -> uint64_t { return (v + 15ULL) & ~15ULL; };

  uint64_t a_sz = round16(ceil_division(
      (uint64_t)rows_mem_a * cols_mem_a * batch_count * a_type_size,
      uint64_t(a_type_packing)));
  uint64_t b_sz = round16(ceil_division(
      (uint64_t)rows_mem_b * cols_mem_b * batch_count * b_type_size,
      uint64_t(b_type_packing)));
  uint64_t c_sz = accumulate
      ? round16(ceil_division(
          (uint64_t)rows_mem_c * cols_mem_c * batch_count * c_type_size,
          uint64_t(c_type_packing)))
      : 0;
  uint64_t d_sz = inplace
      ? 0
      : round16(ceil_division(
          (uint64_t)rows_mem_d * cols_mem_d * batch_count * d_type_size,
          uint64_t(d_type_packing)));
  uint64_t scale_sz = round16(a_scale_bytes) + round16(b_scale_bytes)
                    + round16(c_scale_bytes) + round16(d_scale_bytes);

  uint64_t single_block_size = a_sz + b_sz + c_sz + d_sz + scale_sz;

  if (flush_memory_size == 0) {
    // Not specified, leave flush_batch_count at its user-provided value.
    return;
  }
  if (single_block_size == 0) {
    // Degenerate problem; nothing to flush.
    return;
  }

  uint64_t flush_memory_size_bytes = (uint64_t)flush_memory_size * 1024 * 1024;
  // Ceil division matches hipblaslt-bench so that a partial fit still
  // yields at least one rotating block instead of rounding down to zero.
  uint64_t new_flush_batch_count = ceil_division(flush_memory_size_bytes, single_block_size);
  int max_iters = std::max(cold_iters, iters);

  if (flush_memory_size_bytes < single_block_size) {
    std::cerr << "Note: Problem does not fit into memory size of "
              << flush_memory_size << "MiB; clamping flush_batch_count to 1" << std::endl;
    flush_batch_count = 1;
  } else if ((int64_t)new_flush_batch_count > max_iters) {
    flush_batch_count = max_iters;
    std::cout << "Note: flush_batch_count reduced from " << new_flush_batch_count
              << " to " << flush_batch_count
              << " to avoid excessive memory allocation." << std::endl;
  } else {
    flush_batch_count = (int)new_flush_batch_count;
  }
  std::cout << "Using flush_batch_count = " << flush_batch_count << std::endl;
}


scaling_type generic_gemm::set_scale_mode(string value) {
  // Is this a digit or a word?
  bool is_number = std::find_if(value.begin(), value.end(), ::isdigit) != value.end();
  scaling_type out = scaling_type::None;
  if (is_number) {
    switch (std::stoi(value)) {
      //0 = none, 1 = scalar, 2 = vector, 3 = block,
      case 0:
        out = scaling_type::None;
        break;
      case 1: 
        out = scaling_type::Scalar;
        break;
      case 2:
        out = scaling_type::Vector;
        break;
      case 3:
        out = scaling_type::Block;
        break;
      // 1001 matches hipblaslt-bench's --scaleA/--scaleB value for the gfx950
      // pre-swizzled MX block scale layout (Block_32_UE8M0_32_8_EXT).
      case 1001:
        out = scaling_type::BlockSwizzled;
        break;
    }
  } else {
    string lower_val = value;
    //std::transform(value.begin(), value.end(), lower_val.begin(),
    //[](unsigned char c){ return std::tolower(c); });
    std::transform(lower_val.begin(), lower_val.end(), lower_val.begin(), ::tolower);
    if (lower_val == "none") {
      out = scaling_type::None;
    } else if (lower_val == "scalar") {
      out = scaling_type::Scalar;
    } else if (lower_val == "vector") {
      out = scaling_type::Vector;
    } else if (lower_val == "block") {
      out = scaling_type::Block;
    } else if (lower_val == "block_swizzled") {
      // Note: the text alias must not contain digits, since set_scale_mode
      // routes any value containing a digit through the numeric (stoi) path.
      out = scaling_type::BlockSwizzled;
    }
  }
  return out;
}


std::string generic_gemm::set_init(matrix_desc desc, std::string init, std::string mx_init) {
  // Set init if datatype is using
  if (mx_init == "" ||
      (desc.scale_mode != scaling_type::Block &&
       desc.scale_mode != scaling_type::BlockSwizzled)) {
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

void generic_gemm::run_solutions() {
  string header = prepare_array();
  std::cout << header << std::flush;
  for (int i = 0; i < total_solution_count; i++) {
    current_solution_index = i;
    test(i);
    std::cout << std::fixed;
    std::string results = get_result_string();
    std::cout << results << std::flush;
  }
}

std::string scaling_string(scaling_type input){
  if (input == scaling_type::None) {
    return "None";
  } else if (input == scaling_type::Scalar) {
    return "Scalar";
  } else if (input == scaling_type::Vector) {
    return "Vector";
  } else if (input == scaling_type::Block) {
    return "Block";
  } else if (input == scaling_type::BlockSwizzled) {
    return "BlockSwizzled";
  }
  return "None";
}
