#pragma once 
#include <genericGemm.h>
#include <cxxopts.hpp>
#include <iostream>

class genericGemmFactory {
 public:
  virtual ~genericGemmFactory(){};
  virtual void create_gemm(cxxopts::ParseResult) = 0;
  /**
   * Also note that, despite its name, the Creator's primary responsibility is
   * not creating products. Usually, it contains some core business logic that
   * relies on Product objects, returned by the factory method. Subclasses can
   * indirectly change that business logic by overriding the factory method and
   * returning a different type of product from it.
   */
 protected:
  genericGemm * gemm;

 public:
  std::string prepare_array() { return gemm->prepare_array(); }
  void test() { gemm->test(); } 
  std::string get_result_string() { return gemm->get_result_string(); }
  void free_mem() { gemm->free_mem(); }
};