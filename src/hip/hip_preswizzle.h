#pragma once

// gfx950 MX scale pre-swizzle, ported from the ROCm hipBLASLt/mxDataGenerator
// implementation (shared/mxdatagenerator/lib/include/mxDataGenerator/PreSwizzle.hpp,
// preSwizzleScalesGFX950). This reorders the UE8M0 block-scale bytes into the
// layout the gfx950 subtile kernels expect for scaleA/scaleB = 1001
// (HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT).
//
// It implements the AITER e8m0_shuffle:
//   scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
//   scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
//   scale = scale.view(sm, sn)

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mblas_preswizzle {

inline size_t roundUp(size_t value, size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

// Total element count of preSwizzleScalesGFX950's padded output.
inline size_t preSwizzleScalesGFX950PaddedSize(size_t numRows, size_t numCols) {
  return roundUp(numRows, 32) * roundUp(numCols, 8);
}

template <typename T>
inline size_t product(std::vector<T> const& x) {
  return std::accumulate(x.begin(), x.end(), size_t(1), std::multiplies<size_t>());
}

inline std::vector<size_t> computeShuffledStrides(std::vector<size_t> const& sizes,
                                                  std::vector<size_t> const& dimOrder) {
  std::vector<size_t> strides(sizes.size(), 0);
  size_t stride = 1;
  for (auto idx : dimOrder) {
    strides.at(idx) = stride;
    stride *= sizes.at(idx);
  }
  return strides;
}

// Reorder data by decomposing into N-D coordinates and remapping through
// distinct source/destination strides.
template <typename T>
inline std::vector<T> shuffleDims(std::vector<T> const& input,
                                  std::vector<size_t> const& sizes,
                                  std::vector<size_t> const& dstStrides,
                                  std::vector<size_t> const& srcStrides) {
  if (sizes.size() != dstStrides.size() || sizes.size() != srcStrides.size())
    throw std::runtime_error("shuffleDims: size/stride dimension mismatch");
  if (sizes.size() < 2)
    throw std::runtime_error("shuffleDims: need at least 2 dimensions");

  size_t totalElements = product(sizes);
  if (input.size() != totalElements) {
    std::ostringstream msg;
    msg << "shuffleDims: input size " << input.size() << " doesn't match expected "
        << totalElements;
    throw std::runtime_error(msg.str());
  }

  std::vector<T> output(input.size());

#pragma omp parallel for
  for (size_t coordNum = 0; coordNum < totalElements; ++coordNum) {
    std::vector<size_t> coord(sizes.size());
    size_t remaining = coordNum;
    for (size_t i = 0; i < sizes.size(); ++i) {
      coord[i] = remaining % sizes[i];
      remaining /= sizes[i];
    }

    size_t srcIdx = 0;
    size_t dstIdx = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
      srcIdx += coord[i] * srcStrides[i];
      dstIdx += coord[i] * dstStrides[i];
    }
    output[dstIdx] = input[srcIdx];
  }

  return output;
}

// Pre-swizzle scale data for gfx950.
//
// input  : row-major scale buffer of shape (numRows, numCols)
//          where numRows = M (free dim) and numCols = K/32 (block dim).
// sizes  : {numRows, numCols}
// return : swizzled buffer, padded to (roundUp(numRows,32), roundUp(numCols,8)).
template <typename T>
inline std::vector<T> preSwizzleScalesGFX950(std::vector<T> const& input,
                                             std::vector<size_t> const& sizes) {
  if (sizes.size() != 2) {
    std::ostringstream msg;
    msg << "preSwizzleScalesGFX950: sizes must have 2 elements, got " << sizes.size();
    throw std::runtime_error(msg.str());
  }

  size_t numRows = sizes[0];  // M dimension (number of scale rows)
  size_t numCols = sizes[1];  // K/32 dimension (number of scale columns)

  size_t totalElements = numRows * numCols;
  if (totalElements != input.size()) {
    std::ostringstream msg;
    msg << "preSwizzleScalesGFX950: input size " << input.size()
        << " doesn't match sizes product " << totalElements;
    throw std::runtime_error(msg.str());
  }

  // Pad rows to multiple of 32 and cols to multiple of 8 if needed.
  size_t paddedRows = roundUp(numRows, 32);
  size_t paddedCols = roundUp(numCols, 8);

  std::vector<T> const* inputPtr = &input;
  std::vector<T> paddedInput;
  if (paddedRows != numRows || paddedCols != numCols) {
    paddedInput.resize(paddedRows * paddedCols, T{});
    for (size_t r = 0; r < numRows; ++r) {
      std::copy(input.begin() + r * numCols,
                input.begin() + r * numCols + numCols,
                paddedInput.begin() + r * paddedCols);
    }
    inputPtr = &paddedInput;
  }

  // 6D view of the 2D row-major input:
  //   (paddedRows/32, 2, 16, paddedCols/8, 2, 4)
  // with row = d0*32 + d1*16 + d2, col = d3*8 + d4*4 + d5,
  // linear = row*paddedCols + col.
  std::vector<size_t> srcSizes = {paddedRows / 32, 2, 16, paddedCols / 8, 2, 4};
  std::vector<size_t> srcStrides = {32 * paddedCols, 16 * paddedCols, paddedCols, 8, 4, 1};

  // Row-major output order derived from the inverse of permute(0,3,5,2,4,1).
  std::vector<size_t> dimOrder = {1, 4, 2, 5, 3, 0};
  auto dstStrides = computeShuffledStrides(srcSizes, dimOrder);

  return shuffleDims(*inputPtr, srcSizes, dstStrides, srcStrides);
}

}  // namespace mblas_preswizzle
