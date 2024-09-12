#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <map>
#include <string>
#include <vector>
#include "mblasCuDataType.h"
#include "mblasCuComputeType.h"

bool matchGemmType(mblasDataType precision, std::string function, mblasDataType desiredPrec, std::vector<std::string> acceptable);