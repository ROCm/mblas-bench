#include "rocblasDtypeUtils.h"

#include <hip/hip_runtime.h>

#include <iostream>
#include <map>
#include <string>
using namespace std;

// bool isReal(rocblas_datatype type) {
//   // You could also do this based on the string version with _R_ or _C_, but
//   // those are hardcoded anyway
//   switch (type) {
//     case rocblas_datatype_f16_r:
//     case rocblas_datatype_bf16_r:
//     case rocblas_datatype_f32_r:
//     case rocblas_datatype_f64_r:
//     case rocblas_datatype_i8_r:
//     case rocblas_datatype_u8_r:
//     case rocblas_datatype_i32_r:
//     case rocblas_datatype_u32_r:
//       return true;
//       break;
// 
//     // Complex numbers
//     case rocblas_datatype_f16_c:
//     case rocblas_datatype_bf16_c:
//     case rocblas_datatype_f32_c:
//     case rocblas_datatype_f64_c:
//     case rocblas_datatype_i8_c:
//     case rocblas_datatype_u8_c:
//     case rocblas_datatype_i32_c:
//     case rocblas_datatype_u32_c:
//       return false;
//       break;
//     // Assume real I guess
//     default:
//       return true;
//   }
// }

// bool isReal(hipDataType type) {
//   // You could also do this based on the string version with _R_ or _C_, but
//   // those are hardcoded anyway
//   switch (type) {
//     case HIP_R_16F:
//     case HIP_R_16BF:
//     case HIP_R_32F:
//     case HIP_R_64F:
//     case HIP_R_8I:
//     case HIP_R_8U:
//     case HIP_R_32I:
//     case HIP_R_32U:
//       return true;
//       break;
// 
//     // Complex numbers
//     case HIP_C_16F:
//     case HIP_C_16BF:
//     case HIP_C_32F:
//     case HIP_C_64F:
//     case HIP_C_8I:
//     case HIP_C_8U:
//     case HIP_C_32I:
//     case HIP_C_32U:
//       return false;
//       break;
//     // Assume real I guess
//     default:
//       return true;
//   }
// }
// 
// bool isFp8(rocblas_datatype precision) {
//   // if (precision == CUDA_R_8F_E4M3 || precision == CUDA_R_8F_E5M2) {
//   //   return true;
//   // }
//   return false;
// }
// 
// bool isFp8(hipDataType precision) {
//   // if (precision == CUDA_R_8F_E4M3 || precision == CUDA_R_8F_E5M2) {
//   //   return true;
//   // }
//   return false;
// }
// 
// // std::string precToString(rocblas_datatype precision) {
// //   for (auto ele : precRocblasDType) {
// //     if (ele.second == precision && ele.first.find("rocblas_datatype") != string::npos) {
// //       return ele.first;
// //     }
// //   }
// //   return "";
// // }
// 
// std::string precToString(hipDataType precision) {
//   for (auto ele : precHipblasDType) {
//     if (ele.second == precision && ele.first.find("HIP") != string::npos) {
//       return ele.first;
//     }
//   }
//   return "";
// }
// 
// // std::string computeToString(rocblas_datatype compute) {
// //   for (auto ele : computeRocblasDType) {
// //     if (ele.second == compute && ele.first.find("rocblas_datatype") != string::npos) {
// //       return ele.first;
// //     }
// //   }
// //   return "";
// // }
// 
// std::string computeToString(hipblasComputeType_t compute) {
//   for (auto ele : computeHipblasDType) {
//     if (ele.second == compute && ele.first.find("HIPBLASLT_COMPUTE") != string::npos) {
//       return ele.first;
//     }
//   }
//   return "";
// }
// 
// // rocblas_datatype precisionStringToRocblasDType(std::string stringPrecision) {
// //   try {
// //     return precRocblasDType.at(stringPrecision);
// //   } catch (std::out_of_range &e) {
// //     std::cerr << "Failed to parse precision: " << stringPrecision << std::endl;
// //     throw e;
// //     return rocblas_datatype_f32_r;
// //   }
// // }
// 
// hipDataType precisionStringToHipDType(std::string stringPrecision) {
//   try {
//     return precHipblasDType.at(stringPrecision);
//   } catch (std::out_of_range &e) {
//     std::cerr << "Failed to parse precision: " << stringPrecision << std::endl;
//     throw e;
//     return HIPBLAS_R_32F;
//   }
// }
// 
// hipDataType selectComputeR(std::string computestr,
//                                hipDataType precision) {
//   if (computestr == "") {
//     // If the user doesnt specify, just guess based on precision
//     hipDataType compute;
//     try {
//       compute = precToRocblasCompute.at(precision);
//     } catch (std::out_of_range &e) {
//       compute = HIP_R_32F;
//     }
//     return compute;
//   }
//   try {
//     return computeRocblasDType.at(computestr);
//   } catch (std::out_of_range &e) {
//     std::cerr << "Failed to parse precision: " << computestr << std::endl;
//     throw e;
//     return HIP_R_32F;
//   }
// }
// 
// hipblasLtComputeType_t selectCompute(std::string computestr,
//                                   hipDataType precision) {
//   if (computestr == "") {
//     // If the user doesnt specify, just guess based on precision
//     hipblasLtComputeType_t compute;
//     try {
//       compute = precToHipblasCompute.at(precision);
//     } catch (std::out_of_range &e) {
//       compute = HIPBLASLT_COMPUTE_F32;
//     }
//     return compute;
//   }
//   try {
//     return computeHipblasDType.at(computestr);
//   } catch (std::out_of_range &e) {
//     std::cerr << "Failed to parse precision: " << computestr << std::endl;
//     throw e;
//     return HIPBLASLT_COMPUTE_F32;
//   }
// }
// 
// hipDataType selectScalar(std::string scalarstr, hipDataType precision,
//                               hipDataType compute) {
//   if (scalarstr == "") {
//     // Scalar type not specified, setting based on compute type
//     for (auto ele : precToRocblasCompute) {
//       if (ele.second == compute && isReal(precision) == isReal(ele.first)) {
//         return ele.first;
//       }
//     }
//     // something terrible has happened
//     return precision;
//   }
//   return precisionStringToRocblasDType(scalarstr);
// }
// 
// hipDataType selectScalar(std::string scalarstr, hipDataType precision,
//                               hipblasLtComputeType_t compute) {
//   if (scalarstr == "") {
//     // Scalar type not specified, setting based on compute type
//     for (auto ele : precToHipblasCompute) {
//       if (ele.second == compute && isReal(precision) == isReal(ele.first)) {
//         return ele.first;
//       }
//     }
//     // something terrible has happened
//     return precision;
//   }
//   return precisionStringToHipDType(scalarstr);
// }
// 
// // rocblas_operation opStringToRocblasOp(std::string opstr) {
// //   if (opstr.empty()) {
// //     return rocblas_operation_none;
// //   }
// //   try {
// //     return rocblasOpType.at(opstr);
// //   } catch (std::out_of_range &e) {
// //     std::cerr << "Failed to parse precision: " << opstr << std::endl;
// //     throw e;
// //   }
// // }
// 
// hipblasOperation_t opStringToHipblasOp(std::string opstr) {
//   if (opstr.empty()) {
//     return HIPBLAS_OP_N;
//   }
//   try {
//     return hipblasOpType.at(opstr);
//   } catch (std::out_of_range &e) {
//     std::cerr << "Failed to parse precision: " << opstr << std::endl;
//     throw e;
//   }
// }
// 
// // std::string opToString(rocblas_operation op) {
// //   for (auto ele : rocblasOpType) {
// //     if (ele.second == op) {
// //       return ele.first;
// //     }
// //   }
// //   return "N";
// // }
// 
// std::string opToString(hipblasOperation_t op) {
//   for (auto ele : hipblasOpType) {
//     if (ele.second == op) {
//       return ele.first;
//     }
//   }
//   return "N";
// }