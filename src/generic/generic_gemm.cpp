#include "generic_gemm.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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

  set_timing_config(result);
  sum_devices = result["sum_devices"].as<bool>();

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

  a_props.skip_convert = (a_props.init == "uniform_native");
  b_props.skip_convert = (b_props.init == "uniform_native");
  c_props.skip_convert = (c_props.init == "uniform_native");
  d_props.skip_convert = (d_props.init == "uniform_native");

  requested_solution_num = result["requested_solution_num"].as<int>();
  if (requested_solution_num == 0 || requested_solution_num < -1) {
    throw std::invalid_argument("Invalid --requested_solution_num. Must be -1 (all) or a positive integer.");
  }

  // Set init control information
  if (initialization == "rand_int") {
    control_b = true;
  } else if (initialization == "trig_float") {
    control_a = true;
  } else if (initialization == "uniform_trig") {
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
  // Blocks past the iteration count go unused. Adaptive mode has no fixed count,
  // so it caps on adaptive_max_iters instead (0 => uncapped).
  int max_iters = std::max({1, timing_cfg.cold_iters, timing_cfg.iters});
  if (timing_cfg.adaptive) {
    max_iters = timing_cfg.max_iters > 0 ? timing_cfg.max_iters : std::numeric_limits<int>::max();
  }

  if (flush_memory_size_bytes < single_block_size) {
    std::cerr << "Note: Problem does not fit into memory size of "
              << flush_memory_size << "MiB; clamping flush_batch_count to 1" << std::endl;
    flush_batch_count = 1;
  } else if (new_flush_batch_count > (uint64_t)max_iters) {
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
  // Lowercase the value first, so the check is case insensitive.
  string lower_val = value;
  std::transform(lower_val.begin(), lower_val.end(), lower_val.begin(), ::tolower);

  // The value is a number only when every character is a digit. A block name
  // such as "b32_ue8m0" has digits but is not a number.
  bool is_number = !lower_val.empty() &&
                   std::all_of(lower_val.begin(), lower_val.end(), ::isdigit);

  if (is_number) {
    // Numbers use the hipblaslt-bench client map.
    switch (std::stoi(lower_val)) {
      case 0: return scaling_type::None;
      case 1: return scaling_type::Scalar;
      case 2: return scaling_type::Vector;
      case 3: return scaling_type::Block_32_UE8M0;
      case 4: return scaling_type::Block_16_UE8M0;
      case 5: return scaling_type::Block_32_UE4M3;
      case 6: return scaling_type::Block_16_UE4M3;
      case 7: return scaling_type::Block_32_UE5M3;
      case 8: return scaling_type::Block_16_UE5M3;
      // gfx950 pre-swizzled MX layout (BLK32_UE8M0_32_8_EXT).
      case 1001: return scaling_type::Block_32_UE8M0_Swizzle;
    }
  } else {
    // The word "block" keeps the generic mode. Each backend then picks the
    // scale mode. An explicit name locks the format with no auto selection.
    if (lower_val == "none")   return scaling_type::None;
    if (lower_val == "scalar") return scaling_type::Scalar;
    if (lower_val == "vector") return scaling_type::Vector;
    if (lower_val == "block")  return scaling_type::Block;
    if (lower_val == "block_32_ue8m0" || lower_val == "b32_ue8m0") return scaling_type::Block_32_UE8M0;
    if (lower_val == "block_16_ue8m0" || lower_val == "b16_ue8m0") return scaling_type::Block_16_UE8M0;
    if (lower_val == "block_32_ue4m3" || lower_val == "b32_ue4m3") return scaling_type::Block_32_UE4M3;
    if (lower_val == "block_16_ue4m3" || lower_val == "b16_ue4m3") return scaling_type::Block_16_UE4M3;
    if (lower_val == "block_32_ue5m3" || lower_val == "b32_ue5m3") return scaling_type::Block_32_UE5M3;
    if (lower_val == "block_16_ue5m3" || lower_val == "b16_ue5m3") return scaling_type::Block_16_UE5M3;
    if (lower_val == "block_32_ue8m0_k4" || lower_val == "b32_ue8m0_k4") return scaling_type::Block_32_UE8M0_K4;
    if (lower_val == "block_128_ue8m0_k4" || lower_val == "b128_ue8m0_k4") return scaling_type::Block_128_UE8M0_K4;
    if (lower_val == "block_32_ue8m0_swizzle" || lower_val == "b32_ue8m0_swizzle") {
      return scaling_type::Block_32_UE8M0_Swizzle;
    }
  }

  throw std::invalid_argument(
      "Unknown scale mode \"" + value + "\". "
      "Use a number (0 none, 1 scalar, 2 vector, 3 to 8 block, 1001 gfx950 swizzled block), "
      "the word block, or a block name such as Block_32_UE8M0 or B16_UE4M3.");
}


std::string generic_gemm::set_init(matrix_desc desc, std::string init, std::string mx_init) {
  // Set init if datatype is using
  if (mx_init == "" || !is_block_scaling(desc.scale_mode)) {
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

void generic_gemm::set_timing_config(const cxxopts::ParseResult &result) {
  timing_cfg.adaptive = result["adaptive"].as<bool>();
  timing_cfg.iters = result["iters"].as<int>();
  timing_cfg.cold_iters = result["cold_iters"].as<int>();

  if (!timing_cfg.adaptive) {
    for (const char *opt : {"adaptive_warmup_time", "adaptive_sample_time",
                            "adaptive_measure_time", "adaptive_max_measure_time",
                            "adaptive_min_iters", "adaptive_max_iters",
                            "adaptive_noise_threshold", "adaptive_stability_threshold",
                            "adaptive_stability_window", "adaptive_stability_interval"}) {
      if (result.count(opt)) {
        throw std::invalid_argument(string("--") + opt + " requires --adaptive");
      }
    }
    return;
  }

  // Warmup and iters_per_sample are sized adaptively, so a fixed count would be ignored.
  for (const char *opt : {"iters", "cold_iters"}) {
    if (result.count(opt)) {
      throw std::invalid_argument(string("--") + opt +
                                  " cannot be used with --adaptive (use --adaptive_* options)");
    }
  }
  timing_cfg.warmup_time = result["adaptive_warmup_time"].as<float>();
  timing_cfg.sample_time = result["adaptive_sample_time"].as<float>();
  timing_cfg.measure_time = result["adaptive_measure_time"].as<float>();
  timing_cfg.max_measure_time = result["adaptive_max_measure_time"].as<float>();
  timing_cfg.min_iters = result["adaptive_min_iters"].as<int>();
  timing_cfg.max_iters = result["adaptive_max_iters"].as<int>();
  timing_cfg.noise_threshold = result["adaptive_noise_threshold"].as<float>();
  timing_cfg.stability_threshold = result["adaptive_stability_threshold"].as<float>();
  timing_cfg.stability_window = result["adaptive_stability_window"].as<int>();
  timing_cfg.stability_interval = result["adaptive_stability_interval"].as<int>();
  if (const auto err = validate_adaptive_config(timing_cfg); !err.empty()) {
    throw std::invalid_argument("Invalid adaptive timing config: " + err);
  }
}

void generic_gemm::set_fom_sizes(int a_type_size, int a_type_packing, int b_type_size,
                                 int b_type_packing, int d_type_size, int d_type_packing,
                                 bool is_real) {
  elem_bytes_a = static_cast<double>(a_type_size) / static_cast<double>(a_type_packing);
  elem_bytes_b = static_cast<double>(b_type_size) / static_cast<double>(b_type_packing);
  elem_bytes_d = static_cast<double>(d_type_size) / static_cast<double>(d_type_packing);
  flop_per_mac = is_real ? 2 : 8;
}

std::tuple<double, double, double> generic_gemm::figure_of_merit(double time_us) const {
  const double dm = m, dn = n, dk = k;
  const double gbytes = (elem_bytes_a * dm * dk + elem_bytes_b * dk * dn + elem_bytes_d * dn * dm) / 1e9;
  const double gflops = static_cast<double>(flop_per_mac) * (dm * dn * dk) / 1e9;
  // A non-positive time would divide into inf/nan; report zero throughput instead.
  if (time_us <= 0.0) {
    return {0.0, 0.0, time_us};
  }
  const double time_s = time_us / 1e6;
  return {gflops * batch_count / time_s, gbytes * batch_count / time_s, time_us};
}

std::string generic_gemm::timing_header() const {
  if (!timing_cfg.adaptive) return "";
  return "iters_per_sample,samples,hot_iters,mean_us,min_us,cv,rel_iqr,status,";
}

std::string generic_gemm::timing_values() const {
  if (!timing_cfg.adaptive) return "";
  std::ostringstream oss;
  oss << std::setprecision(7);
  oss << timing.iters_per_sample << ',' << timing.samples << ',' << timing.hot_iters << ','
      << timing.mean_us << ',' << timing.min_us << ',' << timing.cv << ','
      << timing.rel_iqr << ',' << timing_status(timing) << ',';
  return oss.str();
}

void generic_gemm::run_solutions() {
  if (timing_cfg.adaptive) {
    std::cout << format_adaptive_timing_summary(timing_cfg) << std::endl;
  }
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
  switch (input) {
    case scaling_type::None:           return "None";
    case scaling_type::Scalar:         return "Scalar";
    case scaling_type::Vector:         return "Vector";
    case scaling_type::Block:          return "Block";
    case scaling_type::Block_32_UE8M0: return "Block_32_UE8M0";
    case scaling_type::Block_16_UE8M0: return "Block_16_UE8M0";
    case scaling_type::Block_32_UE4M3: return "Block_32_UE4M3";
    case scaling_type::Block_16_UE4M3: return "Block_16_UE4M3";
    case scaling_type::Block_32_UE5M3: return "Block_32_UE5M3";
    case scaling_type::Block_16_UE5M3: return "Block_16_UE5M3";
    case scaling_type::Block_32_UE8M0_K4:  return "Block_32_UE8M0_K4";
    case scaling_type::Block_128_UE8M0_K4: return "Block_128_UE8M0_K4";
    case scaling_type::Block_32_UE8M0_Swizzle: return "Block_32_UE8M0_Swizzle";
  }
  return "Unknown";
}

bool is_block_scaling(scaling_type input) {
  switch (input) {
    case scaling_type::Block:
    case scaling_type::Block_32_UE8M0:
    case scaling_type::Block_16_UE8M0:
    case scaling_type::Block_32_UE4M3:
    case scaling_type::Block_16_UE4M3:
    case scaling_type::Block_32_UE5M3:
    case scaling_type::Block_16_UE5M3:
    case scaling_type::Block_32_UE8M0_K4:
    case scaling_type::Block_128_UE8M0_K4:
    case scaling_type::Block_32_UE8M0_Swizzle:
      return true;
    default:
      return false;
  }
}
