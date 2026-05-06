#pragma once

#include <climits>
#include <cmath>
#include <cstddef>
#include <string>

enum class scale_policy { max_abs, mean_exp };

inline bool is_compound_init(const std::string& s) {
  return s == "max_abs" || s == "mean_exp";
}

inline scale_policy parse_scale_policy(const std::string& s) {
  return s == "mean_exp" ? scale_policy::mean_exp : scale_policy::max_abs;
}

// Returns floor(log2(|x|)) using frexp; INT_MIN sentinel for zero.
inline int floor_log2_abs(float x) {
  if (x == 0.0f) return INT_MIN;
  int e;
  std::frexp(x, &e);
  // frexp returns mantissa in [0.5, 1.0): x = m * 2^e, so 2^(e-1) <= |x| < 2^e
  return e - 1;
}

// Joint data + scale init for UE8M0 block scaling. For each block of
// `block_size` floats along the rows (leading) dim of a column-major
// rows_mem x cols_mem x batch data tensor, derives a power-of-two float scale
// (max_abs or mean_exp of element exponents), divides the block by it, and
// writes the scale into the s_rows x s_cols x batch column-major scale tensor
// at the corresponding (row_block, col, batch) position. Padding entries in
// the scale tensor (rows past rows_mem/block_size, cols past cols_mem) are
// left untouched. All buffers are float; conversion to UE8M0 happens later in
// copy_and_convert.
inline void compound_init_block_ue8m0(float* data, float* scale,
                                      size_t rows_mem, size_t cols_mem,
                                      size_t batch, size_t s_rows,
                                      size_t s_cols, size_t block_size,
                                      scale_policy policy) {
  const size_t row_blocks = rows_mem / block_size;
  const size_t scale_batch_stride = s_rows * s_cols;
  const size_t data_batch_stride = rows_mem * cols_mem;
  const size_t total = batch * cols_mem * row_blocks;
  #pragma omp parallel for
  for (size_t idx = 0; idx < total; idx++) {
    const size_t rb = idx % row_blocks;
    const size_t c  = (idx / row_blocks) % cols_mem;
    const size_t b  = idx / (row_blocks * cols_mem);
    float* block = data + b * data_batch_stride + c * rows_mem + rb * block_size;
    float* s_ptr = scale + b * scale_batch_stride + c * s_rows + rb;

    int exp = INT_MIN;
    if (policy == scale_policy::max_abs) {
      for (size_t i = 0; i < block_size; i++) {
        const int e = floor_log2_abs(block[i]);
        if (e > exp) exp = e;
      }
    } else {
      long sum = 0;
      int count = 0;
      for (size_t i = 0; i < block_size; i++) {
        const int e = floor_log2_abs(block[i]);
        if (e != INT_MIN) { sum += e; count++; }
      }
      if (count > 0) exp = static_cast<int>(std::lround(static_cast<double>(sum) / count));
    }

    if (exp == INT_MIN) {
      *s_ptr = 1.0f;
    } else {
      const float s = std::ldexp(1.0f, exp);
      *s_ptr = s;
      const float inv = 1.0f / s;
      for (size_t i = 0; i < block_size; i++) block[i] *= inv;
    }
  }
}
