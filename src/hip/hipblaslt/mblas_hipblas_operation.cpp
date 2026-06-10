#include "mblas_hipblas_operation.h"
#include <iostream>

hipblasOperation_t mblas_hipblas_operation::convert_to_hip(mblas_hipblas_operation data)  { return convert_to_hip(&data); }

hipblasOperation_t mblas_hipblas_operation::convert_to_hip(const mblas_hipblas_operation *data) {
  try {
    return prec_mappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to Hip Operation " << data->to_string() << std::endl;
    throw e;
  }
}

hipblasOperation_t mblas_hipblas_operation::convert_to_hip() {
  return mblas_hipblas_operation::convert_to_hip(this);
}

// void mblas_hipblas_operation::operator = (const hipblasOperation_t cudt) {
//   for (auto ele : prec_mappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblas_hipblas_operation & mblas_hipblas_operation::operator = (const mblas_hipblas_operation& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}

// mblas_hipblas_operation::operator hipblasOperation_t() const {
//   return convert_to_hip(this);
// }

const std::map<mblas_operation, hipblasOperation_t> mblas_hipblas_operation::prec_mappings = {
    {MBLAS_OP_N,    HIPBLAS_OP_N},
    {MBLAS_OP_T,    HIPBLAS_OP_T},
    {MBLAS_OP_C,    HIPBLAS_OP_C},
};

//hipblasOperation_t opStringToOp(std::string opstr) {
//  if (opstr.empty()) {
//    return HIPBLAS_OP_N;
//  }
//  try {
//    return opType.at(opstr);
//  } catch (std::out_of_range &e) {
//    std::cerr << "Failed to parse precision: " << opstr << std::endl;
//    throw e;
//  }
//}
