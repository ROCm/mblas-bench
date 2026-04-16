#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <map>
#include <string>
#include <vector>
#include "mblas_cuda_data_type.h"
#include "mblas_cuda_compute_type.h"

bool match_gemm_type(mblas_data_type precision, std::string function, mblas_data_type desiredPrec, std::vector<std::string> acceptable);

#if (ENABLE_CUDA_FP4)
std::pair<size_t, size_t> get_scale_tensor_size(int rows, int cols, cublasLtMatmulMatrixScale_t ScaleMode);
#endif

static size_t roundoff(size_t  x, size_t granul);

uint64_t get_fixed_point_workspace_size_in_bytes(int m, int n, int k, int batchCount, bool isComplex, int mantissaControl, int maxMantissaBitCount);
