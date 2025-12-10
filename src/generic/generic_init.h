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
void fill_rand_host_blasgemm(void **ptr_array, long rows_A, long cols_A, long ld, int batch,
                          long long int stride, int flush_batch_count) {
  int a = 1;
  for (int flush_idx = 0; flush_idx < flush_batch_count; flush_idx++) {
    T *A = (T *)ptr_array[flush_idx];
    for (size_t i = 0; i < rows_A * cols_A * batch; i++) {
      A[i] = (T)rand() / (T)(RAND_MAX / a);
    }
  }
}

template <typename T>
void fill_rand_host_constant(void **ptr_array, long rows_A, long cols_A, long ld, int batch,
                            long long int stride, int flush_batch_count, float constant) {
  #pragma omp parallel for collapse(4)
  for (int flush_idx = 0; flush_idx < flush_batch_count; flush_idx++) {
    for (size_t i_batch = 0; i_batch < batch; i_batch++) {
      for (size_t j = 0; j < cols_A; ++j) {
        for (size_t i = 0; i < rows_A; ++i) {
          T *A = (T *)ptr_array[flush_idx];
          A[i + j * ld + i_batch * stride] = (T)(constant);
        }
      }
    }
  }
}

template <typename T>
void fill_rand_host_rand_int_alternating(void **ptr_array, long rows_A, long cols_A, long ld, int batch,
                           long long int stride, int flush_batch_count, bool alternating, int random_dev_seed) {
  #pragma omp parallel
  {
    std::seed_seq seed{random_dev_seed, omp_get_thread_num()};
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> uniform_dist(1, 10);
    T dummy;
    #pragma omp for collapse(4) 
    for (int flush_idx = 0; flush_idx < flush_batch_count; flush_idx++) {
      for (size_t i_batch = 0; i_batch < batch; i_batch++) {
        for (size_t j = 0; j < cols_A; ++j) {
          for (size_t i = 0; i < rows_A; ++i) {
            T *A = (T *)ptr_array[flush_idx];
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
}

template <typename T>
void fill_rand_host_normal_float(void **ptr_array, long rows_A, long cols_A, long ld, int batch,
                             long long int stride, int flush_batch_count, float mean = 0.0f, float std_dev = 1.0f) {
  std::random_device r;
  int random_dev_seed = r();
  #pragma omp parallel
  {
    std::seed_seq seed{random_dev_seed, omp_get_thread_num()};
    std::mt19937 gen(seed);
    std::normal_distribution<T> normal_dist(mean, std_dev);
    #pragma omp for collapse(4) 
    for (int flush_idx = 0; flush_idx < flush_batch_count; flush_idx++) {
      for (size_t i_batch = 0; i_batch < batch; i_batch++) {
        for (size_t j = 0; j < cols_A; ++j) {
          for (size_t i = 0; i < rows_A; ++i) {
            T *A = (T *)ptr_array[flush_idx];
            A[i + j * ld + i_batch * stride] = normal_dist(gen);
          }
        }
      }
    }
  }
}

template <typename T>
void fill_rand_host_uniform(void **ptr_array, long rows_A, long cols_A, long ld, int batch,
                           long long int stride, int flush_batch_count, float min_val = 0.0f, float max_val = 1.0f) {
  std::random_device r;
  int random_dev_seed = r();
  #pragma omp parallel
  {
    std::seed_seq seed{random_dev_seed, omp_get_thread_num()};
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> uniform_dist(min_val, max_val);
    #pragma omp for collapse(4) 
    for (int flush_idx = 0; flush_idx < flush_batch_count; flush_idx++) {
      for (size_t i_batch = 0; i_batch < batch; i_batch++) {
        for (size_t j = 0; j < cols_A; ++j) {
          for (size_t i = 0; i < rows_A; ++i) {
            T *A = (T *)ptr_array[flush_idx];
            A[i + j * ld + i_batch * stride] = uniform_dist(gen);
          }
        }
      }
    }
  }
}

template <typename T>
void fill_rand_host_pow2_binomial(void **ptr_array, long rows_A, long cols_A, long ld, int batch,
                                   long long int stride, int flush_batch_count, int n = 10) {
  std::random_device r;
  int random_dev_seed = r();
  #pragma omp parallel
  {
    std::seed_seq seed{random_dev_seed, omp_get_thread_num()};
    std::mt19937 gen(seed);
    std::binomial_distribution<int> binomial_dist(2 * n + 1, 0.5);
    #pragma omp for collapse(4) 
    for (int flush_idx = 0; flush_idx < flush_batch_count; flush_idx++) {
      for (size_t i_batch = 0; i_batch < batch; i_batch++) {
        for (size_t j = 0; j < cols_A; ++j) {
          for (size_t i = 0; i < rows_A; ++i) {
            T *A = (T *)ptr_array[flush_idx];
            int binomial_value = binomial_dist(gen);
            int offset_value = binomial_value - (n + 1);
            A[i + j * ld + i_batch * stride] = T(std::ldexp(T(1), offset_value));
          }
        }
      }
    }
  }
}

template <typename T>
void fill_rand_host_trig_float(void **ptr_array, long rows_A, long cols_A, long ld, int batch,
                           long long int stride, int flush_batch_count, bool isSin, float scaling) {
  const long long int matrix_size = rows_A * cols_A * batch;
  #pragma omp parallel
  {
    #pragma omp for collapse(4)
    for (int flush_idx = 0; flush_idx < flush_batch_count; flush_idx++) {
      for (size_t i_batch = 0; i_batch < batch; i_batch++) {
        for (size_t j = 0; j < cols_A; ++j) {
          // size_t offset = j * ld + i_batch * stride;
          for (size_t i = 0; i < rows_A; ++i) {
            T *A = (T *)ptr_array[flush_idx];
            // Add offset based on flush_idx to ensure different matrices for each rotating tensor
            long long int flush_offset = flush_idx * matrix_size;
            if (isSin) {
              A[i + j * ld + i_batch * stride] = T(scaling * sin(flush_offset + i + j * ld + i_batch * stride));
            } else {
              A[i + j * ld + i_batch * stride] = T(scaling * cos(flush_offset + i + j * ld + i_batch * stride));
            }
          }
        }
      }
    }
  }
}

template <typename T>
void fill_rand_host_csv(void **ptr_array, long rows_A, long cols_A, long ld, int batch,
                         long long int stride, int flush_batch_count, std::string filename) {
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

  size_t n_rows = result.size();
  for (int flush_idx = 0; flush_idx < flush_batch_count; flush_idx++) {
    T *A = (T *)ptr_array[flush_idx];
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
}


template <typename T>
struct initHost {
  void operator()(std::string initialization, void **ptr_array, long rows_A, long cols_A,
                  long ld, int batch, long long int stride, int flush_batch_count,
                  bool control = false, float constant = 0.f, std::string filename = "");
};

// Generic helper function to detect and parse parameterized initialization patterns
template<typename... Args>
bool parse_parameterized_init(const std::string& initialization, 
                           const std::vector<std::string>& patterns,
                           Args&... default_and_output_params);

template <typename T>
void initHost<T>::operator()(std::string initialization, void **ptr_array, long rows_A,
                             long cols_A, long ld, int batch,
                             long long int stride, int flush_batch_count, bool control,
                             float constant, std::string filename) {
  // Norm defaults
  float mean = 0.0;
  float std_dev = 1.0;
  // Uniform defaults
  float min_val = 0.0;
  float max_val = 1.0;
  
  if (!filename.empty()) {
    fill_rand_host_csv<T>(ptr_array, rows_A, cols_A, ld, batch, stride, flush_batch_count, filename);
  } else if (initialization == "rand_int") {
    std::random_device r;
    fill_rand_host_rand_int_alternating<T>(ptr_array, rows_A, cols_A, ld, batch, stride, flush_batch_count, control, r());
  } else if (initialization == "trig_float") {
    fill_rand_host_trig_float<T>(ptr_array, rows_A, cols_A, ld, batch, stride, flush_batch_count, control, constant);
  } else if (parse_parameterized_init(initialization, 
            {"normal_float", "norm_float", "norm_dist"}, mean, std_dev)) {
    // Can be "normal_float", "norm_float", or "norm_dist"
    if constexpr (std::is_floating_point_v<T>) {
      //std::cout << "mean: " << mean << " std_dev: " << std_dev << std::endl;
      fill_rand_host_normal_float<T>(ptr_array, rows_A, cols_A, ld, batch, stride, flush_batch_count, mean, std_dev);
    } else {
      std::string error_string = "Error: normal distribution not supported for non-floating-point types";
      throw std::invalid_argument(error_string);
    }
  } else if (parse_parameterized_init(initialization, 
            {"uniform_dist", "uniform"}, min_val, max_val)) {
    // Can be "uniform_dist" or "uniform"
    if constexpr (std::is_floating_point_v<T>) {
      //std::cout << "min_val: " << min_val << " max_val: " << max_val << std::endl;
      fill_rand_host_uniform<T>(ptr_array, rows_A, cols_A, ld, batch, stride, flush_batch_count, min_val, max_val);
    } else {
      std::string error_string = "Error: uniform distribution not supported for non-floating-point types";
      throw std::invalid_argument(error_string);
    }
  } else if (parse_parameterized_init(initialization, 
            {"pow2_binomial"}, (int&)batch)) {
    // Parse n parameter, default is 10
    int n = 10;
    parse_parameterized_init(initialization, {"pow2_binomial"}, n);
    if constexpr (std::is_floating_point_v<T>) {
      fill_rand_host_pow2_binomial<T>(ptr_array, rows_A, cols_A, ld, batch, stride, flush_batch_count, n);
    } else {
      std::string error_string = "Error: pow2_binomial distribution not supported for non-floating-point types";
      throw std::invalid_argument(error_string);
    }
  } else if (initialization == "hpl") {
    // HPL initialization not yet implemented for rotating tensors
  } else if (initialization == "blasgemm") {
    fill_rand_host_blasgemm<T>(ptr_array, rows_A, cols_A, ld, batch, stride, flush_batch_count);
  } else if (initialization == "constant") {
    fill_rand_host_constant<T>(ptr_array, rows_A, cols_A, ld, batch, stride, flush_batch_count, constant);
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
