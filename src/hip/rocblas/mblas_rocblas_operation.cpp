#include "mblas_rocblas_operation.h"
#include <iostream>

rocblas_operation mblas_rocblas_operation::convert_to_rocm(mblas_rocblas_operation data)  { return convert_to_rocm(&data); }

rocblas_operation mblas_rocblas_operation::convert_to_rocm(const mblas_rocblas_operation *data) {
  try {
    return prec_mappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to rocBLAS operation" << data->to_string() << std::endl;
    throw e;
  }
}

rocblas_operation mblas_rocblas_operation::convert_to_rocm() {
  return mblas_rocblas_operation::convert_to_rocm(this);
}

// void mblas_rocblas_operation::operator = (const rocblas_operation cudt) {
//   for (auto ele : prec_mappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblas_rocblas_operation & mblas_rocblas_operation::operator = (const mblas_rocblas_operation& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}

// mblas_rocblas_operation::operator rocblas_operation() const {
//   return convert_to_rocm(this);
// }

const std::map<mblas_operation, rocblas_operation> mblas_rocblas_operation::prec_mappings = {
    {MBLAS_OP_N,    rocblas_operation_none},
    {MBLAS_OP_T,    rocblas_operation_transpose},
    {MBLAS_OP_C,    rocblas_operation_conjugate_transpose},
};

//rocblas_operation opStringToOp(std::string opstr) {
//  if (opstr.empty()) {
//    return CUBLAS_OP_N;
//  }
//  try {
//    return opType.at(opstr);
//  } catch (std::out_of_range &e) {
//    std::cerr << "Failed to parse precision: " << opstr << std::endl;
//    throw e;
//  }
//}
