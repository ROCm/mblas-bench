#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Description of one matrix dump. The raw device bytes are written to
// <prefix>_<matrix>.bin and this description to <prefix>_<matrix>.bin.json.
struct matrix_dump_info {
  char matrix = 'a';          // 'a', 'b', 'c' or 'd'
  std::string type_name;      // e.g. HIP_R_4F_E2M1
  int packing = 1;            // elements per byte (2 for fp4, low nibble first)
  long rows = 0, cols = 0;    // logical dimensions
  long rows_mem = 0, cols_mem = 0;  // dimensions as stored (after transpose)
  long ld = 0;
  int batch = 1;
  long long stride = 0;
  std::string init;           // initialization string used for the host fill
  int device = 0;
  std::string stage;          // "before_gemm" or "after_gemm"
};

// Writes the raw bytes and the JSON sidecar. Returns the .bin path.
// Throws std::runtime_error if a file cannot be written.
std::string write_matrix_dump(const std::string &prefix,
                              const matrix_dump_info &info,
                              const std::vector<uint8_t> &bytes);
