#include "mblasHipOperation.h"
#include <iostream>

hipblasOperation_t mblasHipOperation::convertToHip(mblasHipOperation data)  { return convertToHip(&data); }

hipblasOperation_t mblasHipOperation::convertToHip(const mblasHipOperation *data) {
  try {
    return precMappings.at(*data);
  } catch (std::out_of_range &e) {
    std::cout << "Failed to convert to Hip Operation " << data->toString() << std::endl;
    throw e;
  }
}

hipblasOperation_t mblasHipOperation::convertToHip() {
  return mblasHipOperation::convertToHip(this);
}

// void mblasHipOperation::operator = (const hipblasOperation_t cudt) {
//   for (auto ele : precMappings) {
//     if (ele.second == cudt) {
//       set(ele.first);
//     }
//   }
// }

mblasHipOperation & mblasHipOperation::operator = (const mblasHipOperation& mdt) {
  if (this == &mdt)
    return *this;
  // Use parent class default = operator
  set(mdt);
  return *this;
}

// mblasHipOperation::operator hipblasOperation_t() const {
//   return convertToHip(this);
// }

const std::map<mblasOperation, hipblasOperation_t> mblasHipOperation::precMappings = {
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
