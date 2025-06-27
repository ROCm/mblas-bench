#include "genericGemm.h"

#include <string>
#include <utility>

#include "third_party/cxxopts.hpp"
#include "genericSetup.h"

using std::string;

genericGemm::genericGemm(cxxopts::ParseResult result) {
  // Parse basic information
  m = result["m"].as<int>();
  n = result["n"].as<int>();
  k = result["k"].as<int>();

  string ldaS = result["lda"].as<string>();
  string ldbS = result["ldb"].as<string>();
  string ldcS = result["ldc"].as<string>();
  string lddS = result["ldd"].as<string>();

  // We may need these for LDX, but parse them in child implementation
  std::string tA = result["transposeA"].as<std::string>();
  std::string tB = result["transposeB"].as<std::string>();

  // Select a default LD based on OP.  See documentation here:
  // https://netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html
  lda = setLd(ldaS, tA, m, k);
  ldb = setLd(ldbS, tB, k, n);
  // LDC (and LDD) are always max( 1, m), so use that
  ldc = setLd(ldcS, "N", m, 0);
  ldd = setLd(lddS, "N", m, 0);

  // Set matrix dimensions
  std::tie(rowsA, colsA) = setRowCol(tA, m, k);
  std::tie(rowsB, colsB) = setRowCol(tB, k, n);
  std::tie(rowsC, colsC) = setRowCol("N", m, n);
  std::tie(rowsD, colsD) = setRowCol("N", m, n);

  // Set memory dimensions
  rowsMemA = lda;
  rowsMemB = ldb;
  rowsMemC = ldc;
  rowsMemD = ldd;
  std::tie(std::ignore, colsMemA) = setRowCol(tA, m, k);
  std::tie(std::ignore, colsMemB) = setRowCol(tB, k, n);
  std::tie(std::ignore, colsMemC) = setRowCol("N", m, n);
  std::tie(std::ignore, colsMemD) = setRowCol("N", m, n);

  strided = false;
  batched = false;
  function = result["function"].as<string>();

  iters = result["iters"].as<int>();
  cold_iters = result["cold_iters"].as<int>();
  batchct = 1;
  if (function.find("Batched") != string::npos || function.find("batched") != string::npos) {
    batched = true;
  }
  batchct = result["batch_count"].as<int>();
  if (function.find("Strided") != string::npos || function.find("strided") != string::npos) {
    strided = true;
  }
  stride_a = result["stride_a"].as<long long int>();
  stride_b = result["stride_b"].as<long long int>();
  stride_c = result["stride_c"].as<long long int>();

  initialization = result["initialization"].as<string>();
  filenameA = result["filenameA"].as<string>();
  filenameB = result["filenameB"].as<string>();
  filenameC = result["filenameC"].as<string>();

  // Set init control information
  if (initialization == "rand_int") {
    controlB = true;
  } else if (initialization == "trig_float") {
    controlA = true;
  }
}

int genericGemm::setLd(std::string ld, std::string OP, int x, int y) {
  // Use user specified value
  if (ld != "") {
    return stoi(ld);
  }
  if (OP == "N") {
    return x;
  } else {
    return y;
  }
}

std::pair<int, int> genericGemm::setRowCol(std::string OP, int d1, int d2) {
  if (OP == "N") {
    return std::pair<int, int>(d1, d2);
  } else {
    return std::pair<int, int>(d2, d1);
  }
}
