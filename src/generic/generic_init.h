#pragma once

#include <complex>
#include <iostream>
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <omp.h>

// Rand int gen
template <typename T>
inline T rand_int_gen(std::uniform_int_distribution<int> &idist,
                    std::mt19937 &gen, T &dummy) {
  return T(idist(gen));
}

template <typename T>
inline std::complex<T> rand_int_gen(std::uniform_int_distribution<int> &idist,
                             std::mt19937 &gen, std::complex<T> &dummy) {
  //return {T(idist(gen)), T(idist(gen))};
  return std::complex<T>(idist(gen), idist(gen));
}

template <typename T>
inline T rand_int_gen_negative(std::uniform_int_distribution<int> &idist,
                     std::mt19937 &gen, T &dummy) {
  return -T(idist(gen));
}

template <typename T>
inline std::complex<T> rand_int_gen_negative(std::uniform_int_distribution<int> &idist,
                              std::mt19937 &gen, std::complex<T> &dummy) {
  //return {-T(idist(gen)), -T(idist(gen))};
  return std::complex<T>(-idist(gen), -idist(gen));
}

template <typename T>
inline T normal_float_gen(std::normal_distribution<double> &ndist,
                        std::mt19937 &gen, T &dummy) {
  return T(ndist(gen));
}

template <typename T>
void fill_rand_host_blasgemm(void *ptr, long rows_A, long cols_A, long ld, int batch,
                          long long int stride) {
  int a = 1;
  T *A = (T *)ptr;
  for (size_t i = 0; i < rows_A * cols_A * batch; i++) {
    A[i] = (T)rand() / (T)(RAND_MAX / a);
  }
}

template <typename T>
void fill_rand_host_constant(void *ptr, long rows_A, long cols_A, long ld, int batch,
                          long long int stride, float constant) {
  T *A = (T *)ptr;
  for (size_t i_batch = 0; i_batch < batch; i_batch++) {
    for (size_t j = 0; j < cols_A; ++j) {
      for (size_t i = 0; i < rows_A; ++i) {
        A[i + j * ld + i_batch * stride] = (T)(constant);
      }
    }
  }
}

template <typename T>
void fill_rand_host_rand_int_alternating(void *ptr, long rows_A, long cols_A, long ld, int batch,
                           long long int stride, bool alternating, int random_dev_seed) {
  T *A = (T *)ptr;
  #pragma omp parallel shared(A) 
  {
    std::seed_seq seed{random_dev_seed, omp_get_thread_num()};
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> uniform_dist(1, 10);
    T dummy;
    #pragma omp for collapse(3) 
    for (size_t i_batch = 0; i_batch < batch; i_batch++) {
      for (size_t j = 0; j < cols_A; ++j) {
        for (size_t i = 0; i < rows_A; ++i) {
          if ((!alternating) || (j % 2 ^ i % 2)) {
            A[i + j * ld + i_batch * stride] = rand_int_gen(uniform_dist, gen, dummy);
          } else {
            A[i + j * ld + i_batch * stride] = rand_int_gen_negative(uniform_dist, gen, dummy);
          }
        }
      }
    }
  }
}

template <typename T>
void fill_rand_host_normal_float(void *ptr, long rows_A, long cols_A, long ld, int batch,
                             long long int stride, float mean = 0.0f, float std_dev = 1.0f) {
  T *A = (T *)ptr;
  std::random_device r;
  int random_dev_seed = r();
  #pragma omp parallel shared(A) 
  {
    std::seed_seq seed{random_dev_seed, omp_get_thread_num()};
    std::mt19937 gen(seed);
    std::normal_distribution<T> normal_dist(mean, std_dev);
    #pragma omp for collapse(3) 
    for (size_t i_batch = 0; i_batch < batch; i_batch++) {
      for (size_t j = 0; j < cols_A; ++j) {
        for (size_t i = 0; i < rows_A; ++i) {
          A[i + j * ld + i_batch * stride] = normal_dist(gen);
        }
      }
    }
  }
}

template <typename T>
void fill_rand_host_uniform(void *ptr, long rows_A, long cols_A, long ld, int batch,
                           long long int stride, float min_val = 0.0f, float max_val = 1.0f) {
  T *A = (T *)ptr;
  std::random_device r;
  int random_dev_seed = r();
  #pragma omp parallel shared(A) 
  {
    std::seed_seq seed{random_dev_seed, omp_get_thread_num()};
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> uniform_dist(min_val, max_val);
    #pragma omp for collapse(3) 
    for (size_t i_batch = 0; i_batch < batch; i_batch++) {
      for (size_t j = 0; j < cols_A; ++j) {
        for (size_t i = 0; i < rows_A; ++i) {
          A[i + j * ld + i_batch * stride] = uniform_dist(gen);
        }
      }
    }
  }
}

template <typename T>
void fill_rand_host_trig_float(void *ptr, long rows_A, long cols_A, long ld, int batch,
                           long long int stride, bool isSin, float scaling) {
  T *A = (T *)ptr;
  #pragma omp parallel for shared(A) collapse(3)
  for (size_t i_batch = 0; i_batch < batch; i_batch++) {
    for (size_t j = 0; j < cols_A; ++j) {
      // size_t offset = j * ld + i_batch * stride;
      for (size_t i = 0; i < rows_A; ++i) {
        if (isSin) {
          A[i + j * ld + i_batch * stride] = T(scaling * sin(i + j * ld + i_batch * stride));
        } else {
          A[i + j * ld + i_batch * stride] = T(scaling * cos(i + j * ld + i_batch * stride));
        }
      }
    }
  }
}

template <typename T>
void fill_rand_host_csv(void *ptr, long rows_A, long cols_A, long ld, int batch,
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
struct initHost {
  void operator()(std::string initialization, void *ptr, long rows_A, long cols_A,
                  long ld, int batch, long long int stride, bool control = false,
                  float constant = 0.f, std::string filename = "");
};

// Generic helper function to detect and parse parameterized initialization patterns
template<typename... Args>
bool parse_parameterized_init(const std::string& initialization, 
                           const std::vector<std::string>& patterns,
                           Args&... default_and_output_params);

template <typename T>
void initHost<T>::operator()(std::string initialization, void *ptr, long rows_A,
                             long cols_A, long ld, int batch,
                             long long int stride, bool control,
                             float constant, std::string filename) {
  // Norm defaults
  float mean = 0.0;
  float std_dev = 1.0;
  // Uniform defaults
  float min_val = 0.0;
  float max_val = 1.0;
  
  if (initialization == "csv" && !filename.empty()) {
    fill_rand_host_csv<T>(ptr, rows_A, cols_A, ld, batch, stride, filename);
  } else if (initialization == "rand_int") {
    std::random_device r;
    fill_rand_host_rand_int_alternating<T>(ptr, rows_A, cols_A, ld, batch, stride, control, r());
  } else if (initialization == "trig_float") {
    fill_rand_host_trig_float<T>(ptr, rows_A, cols_A, ld, batch, stride, control, constant);
  } else if (parse_parameterized_init(initialization, 
            {"normal_float", "norm_float", "norm_dist"}, mean, std_dev)) {
    // Can be "normal_float", "norm_float", or "norm_dist"
    if constexpr (std::is_floating_point_v<T>) {
      //std::cout << "mean: " << mean << " std_dev: " << std_dev << std::endl;
      fill_rand_host_normal_float<T>(ptr, rows_A, cols_A, ld, batch, stride, mean, std_dev);
    } else {
      std::string error_string = "Error: normal distribution not supported for non-floating-point types";
      throw std::invalid_argument(error_string);
    }
  } else if (parse_parameterized_init(initialization, 
            {"uniform_dist", "uniform"}, min_val, max_val)) {
    // Can be "uniform_dist" or "uniform"
    if constexpr (std::is_floating_point_v<T>) {
      //std::cout << "min_val: " << min_val << " max_val: " << max_val << std::endl;
      fill_rand_host_uniform<T>(ptr, rows_A, cols_A, ld, batch, stride, min_val, max_val);
    } else {
      std::string error_string = "Error: uniform distribution not supported for non-floating-point types";
      throw std::invalid_argument(error_string);
    }
  //} else if (initialization == "hpl") {
  } else if (initialization == "blasgemm") {
    fill_rand_host_blasgemm<T>(ptr, rows_A, cols_A, ld, batch, stride);
  } else if (initialization == "constant") {
    fill_rand_host_constant<T>(ptr, rows_A, cols_A, ld, batch, stride, constant);
  } else {
    std::string error_string = "Error: \"" + initialization + "\" not supported";
    throw std::invalid_argument(error_string);
  }
}

// Helper function to parse a single parameter from string to the target type
template<typename T>
bool parse_parameter(const std::string& param_str, T& value) {
  try {
    if constexpr (std::is_same_v<T, float>) {
      value = std::stof(param_str);
    } else if constexpr (std::is_same_v<T, double>) {
      value = std::stod(param_str);
    } else if constexpr (std::is_same_v<T, int>) {
      value = std::stoi(param_str);
    } else if constexpr (std::is_same_v<T, long>) {
      value = std::stol(param_str);
    } else if constexpr (std::is_same_v<T, long long>) {
      value = std::stoll(param_str);
    } else {
      // For other types, try to use assignment from double
      value = static_cast<T>(std::stod(param_str));
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// Generic helper function to detect and parse parameterized initialization patterns
template<typename... Args>
bool parse_parameterized_init(const std::string& initialization, 
                             const std::vector<std::string>& patterns,
                             Args&... default_and_output_params) {
  for (const auto& pattern : patterns) {
    if (initialization.find(pattern) == 0) {
      // Pattern found, now check for parameters
      if (initialization.length() > pattern.length() && initialization[pattern.length()] == '_') {
        // Parse additional parameters like "pattern_param1_param2_param3"
        std::string params = initialization.substr(pattern.length() + 1);
        std::istringstream ss(params);
        std::vector<std::string> param_strs;
        std::string param;
        
        // Split parameters by underscore
        while (std::getline(ss, param, '_')) {
          if (!param.empty()) {
            param_strs.push_back(param);
          }
        }
        
        // Parse parameters using fold expression (C++17)
        size_t index = 0;
        ((index < param_strs.size() ? parse_parameter(param_strs[index++], default_and_output_params) : false), ...);
      }
      // Pattern matched (with or without parameters)
      return true;
    }
  }
  return false;
}
