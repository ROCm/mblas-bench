#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <map>
#include <string>
#include <vector>
#include "mblasCuDataType.h"
#include "mblasCuComputeType.h"

bool matchGemmType(mblasDataType precision, std::string function, mblasDataType desiredPrec, std::vector<std::string> acceptable);

std::pair<size_t, size_t> get_scale_tensor_size(int rows, int cols, cublasLtMatmulMatrixScale_t ScaleMode);

static size_t roundoff(size_t  x, size_t granul);