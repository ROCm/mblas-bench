#pragma once

#include <cassert>
#include <iostream>
#include <cstdint>

template <typename T>
T ceil_division(T x, T y) {
  return (x + y - 1) / y;
}
