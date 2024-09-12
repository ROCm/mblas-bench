#include "mblasCuOperation.h"
#include <iostream>

cublasOperation_t mblasCuOperation::convertToCuda(mblasCuOperation data)  { return convertToCuda(&data); }

cublasOperation_t mblasCuOperation::convertToCuda(const mblasCuOperation *data) {
  try {
    return precMappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to CUDA Datatype " << data->toString() << std::endl;
    throw e;
  }
}

cublasOperation_t mblasCuOperation::convertToCuda() {
  return mblasCuOperation::convertToCuda(this);
}

// void mblasCuOperation::operator = (const cublasOperation_t cudt) {
//   for (auto ele : precMappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblasCuOperation & mblasCuOperation::operator = (const mblasCuOperation& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}

// mblasCuOperation::operator cublasOperation_t() const {
//   return convertToCuda(this);
// }

const std::map<mblasOperation, cublasOperation_t> mblasCuOperation::precMappings = {
    {MBLAS_OP_N,    CUBLAS_OP_N},
    {MBLAS_OP_T,    CUBLAS_OP_T},
    {MBLAS_OP_C,    CUBLAS_OP_C},
    {MBLAS_OP_CONJG,    CUBLAS_OP_CONJG},
};

//cublasOperation_t opStringToOp(std::string opstr) {
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
