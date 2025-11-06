#include <rocblas/rocblas.h>
//#include <hipblaslt/hipblaslt.h>
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>

#include <map>
#include <string>


// data

// clang-format off
const std::map<std::string, rocblas_operation> rocblasOpType = {
  {"N", rocblas_operation_none},
  {"T", rocblas_operation_transpose},
  {"C", rocblas_operation_conjugate_transpose},
};

const std::map<std::string, hipblasOperation_t> hipblasOpType = {
  {"N", HIPBLAS_OP_N},
  {"T", HIPBLAS_OP_T},
  {"C", HIPBLAS_OP_C},
};

const std::map<std::string, rocblas_datatype> precRocblasDType = {
    {"h", rocblas_datatype_f16_r},       {"s", rocblas_datatype_f32_r},     {"d", rocblas_datatype_f64_r},
    {"c", rocblas_datatype_f32_c},       {"z", rocblas_datatype_f64_c},     {"f16_r", rocblas_datatype_f16_r},
    {"f16_c", rocblas_datatype_f16_c},   {"f32_r", rocblas_datatype_f32_r}, {"f32_c", rocblas_datatype_f32_c},
    {"f64_r", rocblas_datatype_f64_r},   {"f64_c", rocblas_datatype_f64_c}, {"bf16_r", rocblas_datatype_bf16_r},
    {"bf16_c", rocblas_datatype_bf16_c}, {"i8_r", rocblas_datatype_i8_r},   {"i8_c", rocblas_datatype_i8_c},
    {"i32_r", rocblas_datatype_i32_r},   {"i32_c", rocblas_datatype_i32_c},
    {"rocblas_datatype_f16_r",  rocblas_datatype_f16_r},
    {"rocblas_datatype_f16_c",  rocblas_datatype_f16_c},
    {"rocblas_datatype_bf16_r", rocblas_datatype_bf16_r},
    {"rocblas_datatype_bf16_c", rocblas_datatype_bf16_c},
    {"rocblas_datatype_f32_r",  rocblas_datatype_f32_r},
    {"rocblas_datatype_f32_c",  rocblas_datatype_f32_c},
    {"rocblas_datatype_f64_r",  rocblas_datatype_f64_r},
    {"rocblas_datatype_f64_c",  rocblas_datatype_f64_c},
    {"rocblas_datatype_i8_r",   rocblas_datatype_i8_r},
    {"rocblas_datatype_i8_c",   rocblas_datatype_i8_c},
    {"rocblas_datatype_u8_r",   rocblas_datatype_u8_r},
    {"rocblas_datatype_u8_c",   rocblas_datatype_u8_c},
    {"rocblas_datatype_i32_r",  rocblas_datatype_i32_r},
    {"rocblas_datatype_i32_c",  rocblas_datatype_i32_c},
    {"rocblas_datatype_u32_r",  rocblas_datatype_u32_r},
    {"rocblas_datatype_u32_c",  rocblas_datatype_u32_c},
};

const std::map<std::string, hipDataType> precHipblasDType = {
    {"h", HIP_R_16F},       {"s", HIP_R_32F},       {"d", HIP_R_64F},
    {"c", HIP_C_32F},       {"z", HIP_C_64F},       {"f16_r", HIP_R_16F},
    {"f16_c", HIP_C_16F},   {"f32_r", HIP_R_32F},   {"f32_c", HIP_C_32F},
    {"f64_r", HIP_R_64F},   {"f64_c", HIP_C_64F},   {"bf16_r", HIP_R_16BF},
    {"bf16_c", HIP_C_16BF},  {"i8_r", HIP_R_8I},     {"i8_c", HIP_C_8I},
    {"i32_r", HIP_R_32I},   {"i32_c", HIP_C_32I},
    {"HIPBLAS_R_16F",   HIP_R_16F},
    {"HIPBLAS_C_16F",   HIP_C_16F},
    {"HIPBLAS_R_16B",   HIP_R_16BF},
    {"HIPBLAS_C_16B",   HIP_C_16BF},
    {"HIPBLAS_R_32F",   HIP_R_32F},
    {"HIPBLAS_C_32F",   HIP_C_32F},
    {"HIPBLAS_R_64F",   HIP_R_64F},
    {"HIPBLAS_C_64F",   HIP_C_64F},
    {"HIPBLAS_R_8I",    HIP_R_8I},
    {"HIPBLAS_C_8I",    HIP_C_8I},
    {"HIPBLAS_R_8U",    HIP_R_8U},
    {"HIPBLAS_C_8U",    HIP_C_8U},
    {"HIPBLAS_R_32I",   HIP_R_32I},
    {"HIPBLAS_C_32I",   HIP_C_32I},
    {"HIPBLAS_R_32U",   HIP_R_32U},
    {"HIPBLAS_C_32U",   HIP_C_32U},
};

const std::map<std::string, rocblas_datatype> computeRocblasDType = {
    {"rocblas_datatype_f16_r", rocblas_datatype_f16_r},
    {"rocblas_datatype_f32_r", rocblas_datatype_f32_r},
    {"rocblas_datatype_f64_r", rocblas_datatype_f64_r},
    {"rocblas_datatype_i32_r", rocblas_datatype_i32_r},
    {"f16_r", rocblas_datatype_f16_r},
    {"f32_r", rocblas_datatype_f32_r},
    {"f64_r", rocblas_datatype_f64_r},
    {"i32_r", rocblas_datatype_i32_r},
};

const std::map<std::string, hipblasComputeType_t> computeHipblasDType = {
    {"HIPBLASLT_COMPUTE_F32", HIPBLAS_COMPUTE_32F},
    {"f32_r", HIPBLAS_COMPUTE_32F},
};

const std::map<rocblas_datatype, rocblas_datatype> precToRocblasCompute = {
    {rocblas_datatype_f64_r, rocblas_datatype_f64_r},
    {rocblas_datatype_f64_c, rocblas_datatype_f64_r},
    {rocblas_datatype_f32_r, rocblas_datatype_f32_r},
    {rocblas_datatype_f32_c, rocblas_datatype_f32_r},
    {rocblas_datatype_bf16_r, rocblas_datatype_bf16_r},
    {rocblas_datatype_bf16_c, rocblas_datatype_bf16_r},
    {rocblas_datatype_f16_r, rocblas_datatype_f16_r},
    {rocblas_datatype_f16_c, rocblas_datatype_f16_r},
    {rocblas_datatype_i32_r, rocblas_datatype_i32_r},
};

const std::map<hipDataType, hipblasComputeType_t> precToHipblasCompute = {
    {HIP_R_32F, HIPBLAS_COMPUTE_32F},
    {HIP_C_32F, HIPBLAS_COMPUTE_32F},
};
// clang-format on