#pragma once 
#include <generic_gemm.h>
#include <cxxopts.hpp>
#include <iostream>

class generic_gemm_factory {
 public:
  virtual ~generic_gemm_factory() { delete gemm; }
  virtual void create_gemm(cxxopts::ParseResult) = 0;
  /**
   * Also note that, despite its name, the Creator's primary responsibility is
   * not creating products. Usually, it contains some core business logic that
   * relies on Product objects, returned by the factory method. Subclasses can
   * indirectly change that business logic by overriding the factory method and
   * returning a different type of product from it.
   */
 protected:
  generic_gemm * gemm = nullptr;

 public:
  std::string prepare_array() { return gemm->prepare_array(); }
  void run_solutions() { gemm->run_solutions(); }
  void query_only() { gemm->query_only(); }
  void free_mem() { gemm->free_mem(); }
};
