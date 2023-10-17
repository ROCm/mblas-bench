#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <map>
#include <string>
#include <vector>

bool isReal(cudaDataType_t type);
bool isFp8(cudaDataType precision);
std::string precToString(cudaDataType precision);
std::string computeToString(cublasComputeType_t compute);
cudaDataType_t precisionStringToDType(std::string stringPrecision);
cudaDataType_t selectScalar(std::string scalarstr, cudaDataType_t precision,
                            cublasComputeType_t compute);
cublasComputeType_t selectCompute(std::string computestr,
                                  cudaDataType_t precision);
cublasOperation_t opStringToOp(std::string opstr);
std::string opToString(cublasOperation_t);
bool matchGemmType(cudaDataType_t precision, std::string function, cudaDataType_t desiredPrec, std::vector<std::string> acceptable);

// data

// clang-format off
const std::map<std::string, cublasOperation_t> opType = {
  {"N", CUBLAS_OP_N},
  {"T", CUBLAS_OP_T},
  {"C", CUBLAS_OP_C},
};

const std::map<std::string, cudaDataType_t> precDType = {
    {"h", CUDA_R_16F},       {"s", CUDA_R_32F},     {"d", CUDA_R_64F},
    {"c", CUDA_C_32F},       {"z", CUDA_C_64F},     {"f16_r", CUDA_R_16F},
    {"f16_c", CUDA_C_16F},   {"f32_r", CUDA_R_32F}, {"f32_c", CUDA_C_32F},
    {"f64_r", CUDA_R_64F},   {"f64_c", CUDA_C_64F}, {"bf16_r", CUDA_R_16BF},
    {"bf16_c", CUDA_C_16BF}, {"i8_r", CUDA_R_8I},   {"i8_c", CUDA_C_8I},
    {"i32_r", CUDA_R_32I},   {"i32_c", CUDA_C_32I},
    {"CUDA_R_16F",  CUDA_R_16F},
    {"CUDA_C_16F",  CUDA_C_16F},
    {"CUDA_R_16BF", CUDA_R_16BF},
    {"CUDA_C_16BF", CUDA_C_16BF},
    {"CUDA_R_32F",  CUDA_R_32F},
    {"CUDA_C_32F",  CUDA_C_32F},
    {"CUDA_R_64F",  CUDA_R_64F},
    {"CUDA_C_64F",  CUDA_C_64F},
    {"CUDA_R_4I",   CUDA_R_4I},
    {"CUDA_C_4I",   CUDA_C_4I},
    {"CUDA_R_4U",   CUDA_R_4U},
    {"CUDA_C_4U",   CUDA_C_4U},
    {"CUDA_R_8I",   CUDA_R_8I},
    {"CUDA_C_8I",   CUDA_C_8I},
    {"CUDA_R_8U",   CUDA_R_8U},
    {"CUDA_C_8U",   CUDA_C_8U},
    {"CUDA_R_16I",  CUDA_R_16I},
    {"CUDA_C_16I",  CUDA_C_16I},
    {"CUDA_R_16U",  CUDA_R_16U},
    {"CUDA_C_16U",  CUDA_C_16U},
    {"CUDA_R_32I",  CUDA_R_32I},
    {"CUDA_C_32I",  CUDA_C_32I},
    {"CUDA_R_32U",  CUDA_R_32U},
    {"CUDA_C_32U",  CUDA_C_32U},
    {"CUDA_R_64I",  CUDA_R_64I},
    {"CUDA_C_64I",  CUDA_C_64I},
    {"CUDA_R_64U",  CUDA_R_64U},
    {"CUDA_C_64U",  CUDA_C_64U},
    {"CUDA_R_8F_E4M3", CUDA_R_8F_E4M3},
    {"CUDA_R_8F_E5M2", CUDA_R_8F_E5M2},
    {"f8_r", CUDA_R_8F_E4M3},
    {"bf8_r", CUDA_R_8F_E5M2},
};



const std::map<std::string, cublasComputeType_t> computeDType = {
    {"CUBLAS_COMPUTE_16F", CUBLAS_COMPUTE_16F},
    {"CUBLAS_COMPUTE_16F_PEDANTIC", CUBLAS_COMPUTE_16F_PEDANTIC},
    {"CUBLAS_COMPUTE_32F", CUBLAS_COMPUTE_32F},
    {"CUBLAS_COMPUTE_32F_PEDANTIC", CUBLAS_COMPUTE_32F_PEDANTIC},
    {"CUBLAS_COMPUTE_32F_FAST_16F", CUBLAS_COMPUTE_32F_FAST_16F},
    {"CUBLAS_COMPUTE_32F_FAST_16BF", CUBLAS_COMPUTE_32F_FAST_16BF},
    {"CUBLAS_COMPUTE_32F_FAST_TF32", CUBLAS_COMPUTE_32F_FAST_TF32},
    {"CUBLAS_COMPUTE_64F", CUBLAS_COMPUTE_64F},
    {"CUBLAS_COMPUTE_64F_PEDANTIC", CUBLAS_COMPUTE_64F_PEDANTIC},
    {"CUBLAS_COMPUTE_32I", CUBLAS_COMPUTE_32I},
    {"CUBLAS_COMPUTE_32I_PEDANTIC", CUBLAS_COMPUTE_32I_PEDANTIC},
    {"f32_r", CUBLAS_COMPUTE_32F},
    {"f64_r", CUBLAS_COMPUTE_64F},
    {"i32_r", CUBLAS_COMPUTE_32I},
};

const std::map<cudaDataType_t, cublasComputeType_t> precToCompute = {
    {CUDA_R_64F, CUBLAS_COMPUTE_64F},
    {CUDA_C_64F, CUBLAS_COMPUTE_64F},
    {CUDA_R_32F, CUBLAS_COMPUTE_32F},
    {CUDA_C_32F, CUBLAS_COMPUTE_32F},
    {CUDA_R_16BF, CUBLAS_COMPUTE_32F},
    {CUDA_C_16BF, CUBLAS_COMPUTE_32F},
    {CUDA_R_16F, CUBLAS_COMPUTE_16F},
    {CUDA_C_16F, CUBLAS_COMPUTE_16F},
    {CUDA_R_32I, CUBLAS_COMPUTE_32I},
};
// clang-format on