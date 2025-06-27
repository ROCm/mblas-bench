#include "genericGemm.h"

#include <string>
#include <utility>

#include "third_party/cxxopts.hpp"
#include "genericSetup.h"

using std::string;

genericGemm::genericGemm(cxxopts::ParseResult result) {
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
  lda = setLd(ldaS, tA, m, k);
  ldb = setLd(ldbS, tB, k, n);
  // LDC (and LDD) are always max( 1, m), so use that
  ldc = setLd(ldcS, "N", m, 0);
  ldd = setLd(lddS, "N", m, 0);

  // Set matrix dimensions
  std::tie(rows_a, cols_a) = setRowCol(tA, m, k);
  std::tie(rows_b, cols_b) = setRowCol(tB, k, n);
  std::tie(rows_c, cols_c) = setRowCol("N", m, n);
  std::tie(rows_d, cold_d) = setRowCol("N", m, n);

  // Set memory dimensions
  rows_mem_a = lda;
  rows_mem_b = ldb;
  rows_mem_c = ldc;
  rows_mem_d = ldd;
  std::tie(std::ignore, cols_mem_a) = setRowCol(tA, m, k);
  std::tie(std::ignore, cols_mem_b) = setRowCol(tB, k, n);
  std::tie(std::ignore, cols_mem_c) = setRowCol("N", m, n);
  std::tie(std::ignore, cols_mem_d) = setRowCol("N", m, n);

  strided = false;
  batched = false;
  function = result["function"].as<string>();

  iters = result["iters"].as<int>();
  cold_iters = result["cold_iters"].as<int>();
  batch_count = 1;
  if (function.find("Batched") != string::npos || function.find("batched") != string::npos) {
    batched = true;
  }
  batch_count = result["batch_count"].as<int>();
  if (function.find("Strided") != string::npos || function.find("strided") != string::npos) {
    strided = true;
  }
  stride_a = result["stride_a"].as<long long int>();
  stride_b = result["stride_b"].as<long long int>();
  stride_c = result["stride_c"].as<long long int>();
  stride_d = result["stride_d"].as<long long int>();

  flush_batch_count = result["flush_batch_count"].as<int>();
  flush_memory_size = result["flush_memory_size"].as<int>();

  initialization = result["initialization"].as<string>();
  filenameA = result["filenameA"].as<string>();
  filenameB = result["filenameB"].as<string>();
  filenameC = result["filenameC"].as<string>();

  // Set init control information
  if (initialization == "rand_int") {
    controlB = true;
  } else if (initialization == "trig_float") {
    controlA = true;
  }
}

int genericGemm::setLd(std::string ld, std::string OP, int x, int y) {
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

std::pair<int, int> genericGemm::setRowCol(std::string OP, int d1, int d2) {
  if (OP == "N") {
    return std::pair<int, int>(d1, d2);
  } else {
    return std::pair<int, int>(d2, d1);
  }
}

void genericGemm::set_flush_batch_count(uint64_t & a_offset, uint64_t & b_offset, uint64_t & c_offset, uint64_t & d_offset,
                      int a_type_size,  int b_type_size, int c_type_size, int d_type_size, 
                      int a_type_packing,  int b_type_packing, int c_type_packing, int d_type_packing,
                      bool inplace) {
  // test
  uint64_t single_block_size = calculate_offsets(rows_mem_a, cols_mem_a, rows_mem_b, cols_mem_b, rows_mem_c, cols_mem_c, rows_mem_d, cols_mem_d, 
                    a_offset, b_offset, c_offset, d_offset, a_type_size, b_type_size, c_type_size, d_type_size,
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
  } else {
    flush_batch_count = new_flush_batch_count;
  }
  std::cout << "Using flush_batch_count = " << flush_batch_count << std::endl;
}