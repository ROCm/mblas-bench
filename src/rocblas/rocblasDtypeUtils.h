#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>

#include <map>
#include <string>

bool isReal(rocblas_datatype type);
bool isFp8(rocblas_datatype precision);
std::string precToString(rocblas_datatype precision);
std::string computeToString(rocblas_datatype compute);
rocblas_datatype precisionStringToDType(std::string stringPrecision);
rocblas_datatype selectScalar(std::string scalarstr, rocblas_datatype precision,
                              rocblas_datatype compute);
rocblas_datatype selectCompute(std::string computestr,
                               rocblas_datatype precision);
rocblas_operation opStringToOp(std::string opstr);
std::string opToString(rocblas_operation);

// data

// clang-format off
const std::map<std::string, rocblas_operation> opType = {
  {"N", rocblas_operation_none},
  {"T", rocblas_operation_transpose},
  {"C", rocblas_operation_conjugate_transpose},
};

const std::map<std::string, rocblas_datatype> precDType = {
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



const std::map<std::string, rocblas_datatype> computeDType = {
    {"rocblas_datatype_f16_r", rocblas_datatype_f16_r},
    {"rocblas_datatype_f32_r", rocblas_datatype_f32_r},
    {"rocblas_datatype_f64_r", rocblas_datatype_f64_r},
    {"rocblas_datatype_i32_r", rocblas_datatype_i32_r},
    {"f16_r", rocblas_datatype_f16_r},
    {"f32_r", rocblas_datatype_f32_r},
    {"f64_r", rocblas_datatype_f64_r},
    {"i32_r", rocblas_datatype_i32_r},
};

const std::map<rocblas_datatype, rocblas_datatype> precToCompute = {
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
// clang-format on