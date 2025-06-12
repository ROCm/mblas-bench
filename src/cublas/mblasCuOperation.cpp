#include "mblasCuOperation.h"
#include <iostream>

cublasOperation_t mblasCuOperation::convert_to_cuda(mblasCuOperation data)  { return convert_to_cuda(&data); }

cublasOperation_t mblasCuOperation::convert_to_cuda(const mblasCuOperation *data) {
  try {
    return prec_mappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to CUDA Datatype " << data->to_string() << std::endl;
    throw e;
  }
}

cublasOperation_t mblasCuOperation::convert_to_cuda() {
  return mblasCuOperation::convert_to_cuda(this);
}

// void mblasCuOperation::operator = (const cublasOperation_t cudt) {
//   for (auto ele : prec_mappings) {
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
//   return convert_to_cuda(this);
// }

const std::map<mblasOperation, cublasOperation_t> mblasCuOperation::prec_mappings = {
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
