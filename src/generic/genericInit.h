#pragma once

#include <complex>
#include <iostream>
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Rand int gen
template <typename T>
inline T randIntGen(std::uniform_int_distribution<int> &idist,
                    std::mt19937 &gen, T &dummy) {
  return T(idist(gen));
}

template <typename T>
inline std::complex<T> randIntGen(std::uniform_int_distribution<int> &idist,
                             std::mt19937 &gen, std::complex<T> &dummy) {
  return {T(idist(gen)), T(idist(gen))};
}

template <typename T>
inline T randIntGenN(std::uniform_int_distribution<int> &idist,
                     std::mt19937 &gen, T &dummy) {
  return -T(idist(gen));
}

template <typename T>
inline std::complex<T> randIntGenN(std::uniform_int_distribution<int> &idist,
                              std::mt19937 &gen, std::complex<T> &dummy) {
  return {-T(idist(gen)), -T(idist(gen))};
}

template <typename T>
inline T normalFloatGen(std::normal_distribution<double> &ndist,
                        std::mt19937 &gen, T &dummy) {
  return T(ndist(gen));
}

template <typename T>
void fillRandHostBlasgemm(void *ptr, int rows_A, int cols_A, int ld, int batch,
                          long long int stride) {
  int a = 1;
  T *A = (T *)ptr;
  for (size_t i = 0; i < rows_A * cols_A * batch; i++) {
    A[i] = (T)rand() / (T)(RAND_MAX / a);
  }
}

template <typename T>
void fillRandHostConstant(void *ptr, int rows_A, int cols_A, int ld, int batch,
                          long long int stride, float constant) {
  int a = 1;
  T *A = (T *)ptr;
  for (size_t i = 0; i < rows_A * cols_A * batch; i++) {
    A[i] = (T)(constant);
  }
}

template <typename T>
void fillRandHostRandIntAS(void *ptr, int rows_A, int cols_A, int ld, int batch,
                           long long int stride, bool alternating) {
  std::random_device r;
  std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> uniform_dist(1, 10);
  T *A = (T *)ptr;
  T dummy;
  for (size_t i_batch = 0; i_batch < batch; i_batch++) {
    for (size_t j = 0; j < cols_A; ++j) {
      size_t offset = j * ld + i_batch * stride;
      for (size_t i = 0; i < rows_A; ++i) {
        if ((!alternating) || (j % 2 ^ i % 2)) {
          A[i + offset] = randIntGen(uniform_dist, gen, dummy);
        } else {
          A[i + offset] = randIntGenN(uniform_dist, gen, dummy);
        }
      }
    }
  }
}

template <typename T>
void fillRandHostTrigFloat(void *ptr, int rows_A, int cols_A, int ld, int batch,
                           long long int stride, bool isSin) {
  T *A = (T *)ptr;
  for (size_t i_batch = 0; i_batch < batch; i_batch++) {
    for (size_t j = 0; j < cols_A; ++j) {
      size_t offset = j * ld + i_batch * stride;
      for (size_t i = 0; i < rows_A; ++i) {
        if (isSin) {
          A[i + offset] = T(sin(i + offset));
        } else {
          A[i + offset] = T(cos(i + offset));
        }
      }
    }
  }
}

template <typename T>
void fillRandHostFromCSV(void *ptr, int rows_A, int cols_A, int ld, int batch,
                         long long int stride, std::string filename) {
  std::ifstream file(filename);
  std::vector<std::vector<T>> result;

  for (std::string line; std::getline(file, line, '\n'); ) {
    result.push_back(std::vector<T>());
    std::istringstream ss(line);
    for (std::string field; std::getline(ss, field, ','); ) {
      result.back().push_back((T)std::stod(field));
    }
    if (result.back().empty()) {
      std::cout << "warning: empty row in csv file" << std::endl;
    }
  }
  if (result.empty()) {
    std::cout << "warning: csv file is empty" << std::endl;
  }

  T *A = (T *)ptr;
  size_t n_rows = result.size();
  for (size_t i_batch = 0; i_batch < batch; i_batch++) {
    for (size_t j = 0; j < cols_A; ++j) {
      size_t offset = j * ld + i_batch * stride;
      for (size_t i = 0; i < rows_A; ++i) {
        size_t n_cols = result[i % n_rows].size();
        A[i + offset] = result[i % n_rows][j % n_cols];
      }
    }
  }
}

template <typename T>
void fillRandHostNormalFloat(void *ptr, int rows_A, int cols_A, int ld, int batch,
                             long long int stride) {
  std::random_device r;
  std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
  std::mt19937 gen(seed);
  std::normal_distribution<double> normal_dist(5.0, 2.0);
  T *A = (T *)ptr;
  T dummy;
  for (size_t i_batch = 0; i_batch < batch; i_batch++) {
    for (size_t j = 0; j < cols_A; ++j) {
      size_t offset = j * ld + i_batch * stride;
      for (size_t i = 0; i < rows_A; ++i) {
        A[i + offset] = normalFloatGen(normal_dist, gen, dummy);
      }
    }
  }
}

template <typename T>
struct initHost {
  void operator()(std::string initialization, void *ptr, int rows_A, int cols_A,
                  int ld, int batch, long long int stride, bool control = false,
                  float constant = 0.f, std::string filename = "");
};

template <typename T>
void initHost<T>::operator()(std::string initialization, void *ptr, int rows_A,
                             int cols_A, int ld, int batch,
                             long long int stride, bool control,
                             float constant, std::string filename) {
  if (!filename.empty()) {
    fillRandHostFromCSV<T>(ptr, rows_A, cols_A, ld, batch, stride, filename);
  } else if (initialization == "rand_int") {
    fillRandHostRandIntAS<T>(ptr, rows_A, cols_A, ld, batch, stride, control);
  } else if (initialization == "trig_float") {
    fillRandHostTrigFloat<T>(ptr, rows_A, cols_A, ld, batch, stride, control);
  } else if (initialization == "normal_float") {
    fillRandHostNormalFloat<T>(ptr, rows_A, cols_A, ld, batch, stride);
  } else if (initialization == "hpl") {
  } else if (initialization == "blasgemm") {
    fillRandHostBlasgemm<T>(ptr, rows_A, cols_A, ld, batch, stride);
  } else if (initialization == "constant") {
    fillRandHostConstant<T>(ptr, rows_A, cols_A, ld, batch, stride, constant);
  }
}

