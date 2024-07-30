#include "mblasRocOperation.h"
#include <iostream>

rocblas_operation mblasRocOperation::convertToRocm(mblasRocOperation data)  { return convertToRocm(&data); }

rocblas_operation mblasRocOperation::convertToRocm(const mblasRocOperation *data) {
  try {
    return precMappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to rocBLAS operation" << data->toString() << std::endl;
    throw e;
  }
}

rocblas_operation mblasRocOperation::convertToRocm() {
  return mblasRocOperation::convertToRocm(this);
}

// void mblasRocOperation::operator = (const rocblas_operation cudt) {
//   for (auto ele : precMappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblasRocOperation & mblasRocOperation::operator = (const mblasRocOperation& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}

// mblasRocOperation::operator rocblas_operation() const {
//   return convertToRocm(this);
// }

const std::map<mblasOperation, rocblas_operation> mblasRocOperation::precMappings = {
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
