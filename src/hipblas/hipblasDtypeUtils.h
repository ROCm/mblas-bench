#include <hip/hipblas.h>
#include <hip/hip_runtime.h>

#include <map>
#include <string>

bool isReal(hipDataType type);
bool isFp8(hipDataType precision);
std::string precToString(hipDataType precision);
std::string computeToString(hipblasComputeType_t compute);
hipDataType precisionStringToDType(std::string stringPrecision);
hipDataType selectScalar(std::string scalarstr, hipDataType precision,
                            hipblasComputeType_t compute);
hipblasComputeType_t selectCompute(std::string computestr,
                                  hipDataType precision);
hipblasOperation_t opStringToOp(std::string opstr);
std::string opToString(hipblasOperation_t);

// data

// clang-format off
const std::map<std::string, hipblasOperation_t> opType = {
  {"N", HIPBLAS_OP_N},
  {"T", HIPBLAS_OP_T},
  {"C", HIPBLAS_OP_C},
};

const std::map<std::string, hipDataType> precDType = {
    {"h", HIP_R_16F},       {"s", HIP_R_32F},     {"d", HIP_R_64F},
    {"c", HIP_C_32F},       {"z", HIP_C_64F},     {"f16_r", HIP_R_16F},
    {"f16_c", HIP_C_16F},   {"f32_r", HIP_R_32F}, {"f32_c", HIP_C_32F},
    {"f64_r", HIP_R_64F},   {"f64_c", HIP_C_64F}, {"bf16_r", HIP_R_16BF},
    {"bf16_c", HIP_C_16BF}, {"i8_r", HIP_R_8I},   {"i8_c", HIP_C_8I},
    {"i32_r", HIP_R_32I},   {"i32_c", HIP_C_32I},
    {"HIP_R_16F",  HIP_R_16F},
    {"HIP_C_16F",  HIP_C_16F},
    {"HIP_R_16BF", HIP_R_16BF},
    {"HIP_C_16BF", HIP_C_16BF},
    {"HIP_R_32F",  HIP_R_32F},
    {"HIP_C_32F",  HIP_C_32F},
    {"HIP_R_64F",  HIP_R_64F},
    {"HIP_C_64F",  HIP_C_64F},
    {"HIP_R_4I",   HIP_R_4I},
    {"HIP_C_4I",   HIP_C_4I},
    {"HIP_R_4U",   HIP_R_4U},
    {"HIP_C_4U",   HIP_C_4U},
    {"HIP_R_8I",   HIP_R_8I},
    {"HIP_C_8I",   HIP_C_8I},
    {"HIP_R_8U",   HIP_R_8U},
    {"HIP_C_8U",   HIP_C_8U},
    {"HIP_R_16I",  HIP_R_16I},
    {"HIP_C_16I",  HIP_C_16I},
    {"HIP_R_16U",  HIP_R_16U},
    {"HIP_C_16U",  HIP_C_16U},
    {"HIP_R_32I",  HIP_R_32I},
    {"HIP_C_32I",  HIP_C_32I},
    {"HIP_R_32U",  HIP_R_32U},
    {"HIP_C_32U",  HIP_C_32U},
    {"HIP_R_64I",  HIP_R_64I},
    {"HIP_C_64I",  HIP_C_64I},
    {"HIP_R_64U",  HIP_R_64U},
    {"HIP_C_64U",  HIP_C_64U},
};



const std::map<std::string, hipblasComputeType_t> computeDType = {
    {"HIPBLAS_COMPUTE_16F", HIPBLAS_COMPUTE_16F},
    {"HIPBLAS_COMPUTE_32F", HIPBLAS_COMPUTE_32F},
    {"HIPBLAS_COMPUTE_64F", HIPBLAS_COMPUTE_64F},
    {"HIPBLAS_COMPUTE_32I", HIPBLAS_COMPUTE_32I},
    {"f16_r", HIPBLAS_COMPUTE_16F},
    {"f32_r", HIPBLAS_COMPUTE_32F},
    {"f64_r", HIPBLAS_COMPUTE_64F},
    {"i32_r", HIPBLAS_COMPUTE_32I},
};

const std::map<hipDataType, hipblasComputeType_t> precToCompute = {
    {HIP_R_64F, HIPBLAS_COMPUTE_64F},
    {HIP_C_64F, HIPBLAS_COMPUTE_64F},
    {HIP_R_32F, HIPBLAS_COMPUTE_32F},
    {HIP_C_32F, HIPBLAS_COMPUTE_32F},
    {HIP_R_16BF, HIPBLAS_COMPUTE_32F},
    {HIP_C_16BF, HIPBLAS_COMPUTE_32F},
    {HIP_R_16F, HIPBLAS_COMPUTE_16F},
    {HIP_C_16F, HIPBLAS_COMPUTE_16F},
    {HIP_R_32I, HIPBLAS_COMPUTE_32I},
};
// clang-format on