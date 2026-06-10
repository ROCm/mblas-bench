#include "mblas_cuda_operation.h"
#include <iostream>

cublasOperation_t mblas_cuda_operation::convert_to_cuda(mblas_cuda_operation data)  { return convert_to_cuda(&data); }

cublasOperation_t mblas_cuda_operation::convert_to_cuda(const mblas_cuda_operation *data) {
  try {
    return prec_mappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to CUDA Datatype " << data->to_string() << std::endl;
    throw e;
  }
}

cublasOperation_t mblas_cuda_operation::convert_to_cuda() {
  return mblas_cuda_operation::convert_to_cuda(this);
}

// void mblas_cuda_operation::operator = (const cublasOperation_t cudt) {
//   for (auto ele : prec_mappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblas_cuda_operation & mblas_cuda_operation::operator = (const mblas_cuda_operation& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}

// mblas_cuda_operation::operator cublasOperation_t() const {
//   return convert_to_cuda(this);
// }

const std::map<mblas_operation, cublasOperation_t> mblas_cuda_operation::prec_mappings = {
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
