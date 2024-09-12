#include "cublasDtypeUtils.h"

#include <cuda_runtime.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>
using namespace std;

bool matchGemmType(mblasDataType precision, std::string function, mblasDataType desiredPrec, std::vector<string> acceptable) {
  if (precision != desiredPrec) {
    return false;
  }
  for (auto afunc : acceptable) {
    if (function == afunc) {
      return true;
    }
  }
  return false;
}