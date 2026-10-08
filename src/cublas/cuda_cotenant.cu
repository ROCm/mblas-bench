#include "cuda_cotenant.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

namespace {

constexpr int kBlockSize = 256;
constexpr int kSleepNanoseconds = 1000;
constexpr int kSleepsBetweenStopChecks = 256;
constexpr int kMaxRequestedOccupancy = 64;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cotenant: ") + operation + ": " +
                             cudaGetErrorString(status));
  }
}

std::string driver_error(CUresult status, const char* operation) {
  const char* name = nullptr;
  const char* description = nullptr;
  (void)cuGetErrorName(status, &name);
  (void)cuGetErrorString(status, &description);

  std::ostringstream message;
  message << "cotenant: " << operation << ": "
          << (name ? name : "unknown CUDA driver error");
  if (description)
    message << " (" << description << ')';
  return message.str();
}

void check_driver(CUresult status, const char* operation) {
  if (status != CUDA_SUCCESS)
    throw std::runtime_error(driver_error(status, operation));
}

void log_message(const std::string& message) {
  static std::mutex log_mutex;
  const std::lock_guard<std::mutex> lock(log_mutex);
  std::cerr << message << std::endl;
}

__global__ void cotenant_kernel(unsigned int* ready, unsigned int* stop) {
  extern __shared__ volatile unsigned char shared_memory[];
  if (threadIdx.x == 0) {
    shared_memory[0] = 1;
    atomicAdd(ready, 1U);
  }
  __syncthreads();

  for (;;) {
#pragma unroll 1
    for (int i = 0; i < kSleepsBetweenStopChecks; ++i)
      __nanosleep(kSleepNanoseconds);

    unsigned int done = 0;
    if (threadIdx.x % warpSize == 0)
      done = atomicAdd(stop, 0U);
    if (__shfl_sync(0xffffffffU, done, 0) != 0)
      return;
  }
}

int query_occupancy(std::size_t dynamic_shared_bytes) {
  int blocks_per_sm = 0;
  check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &blocks_per_sm, cotenant_kernel, kBlockSize, dynamic_shared_bytes),
             "querying cotenant occupancy");
  return blocks_per_sm;
}

struct reservation {
  std::size_t bytes;
  int blocks_per_sm;
};

reservation choose_reservation(const cudaDeviceProp& properties,
                               int requested_max_occupancy) {
  cudaFuncAttributes attributes{};
  check_cuda(cudaFuncGetAttributes(&attributes, cotenant_kernel),
             "querying cotenant kernel attributes");

  const std::size_t per_block_limit =
      std::max(properties.sharedMemPerBlock, properties.sharedMemPerBlockOptin);
  if (per_block_limit <= attributes.sharedSizeBytes) {
    throw std::runtime_error(
        "cotenant: no dynamic shared memory is available for the cotenant kernel");
  }
  const std::size_t max_dynamic_bytes = per_block_limit - attributes.sharedSizeBytes;
  if (max_dynamic_bytes > static_cast<std::size_t>(INT_MAX))
    throw std::runtime_error("cotenant: dynamic shared-memory limit exceeds INT_MAX");

  check_cuda(cudaFuncSetAttribute(cotenant_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(max_dynamic_bytes)),
             "opting into cotenant dynamic shared memory");

  const std::size_t shared_per_sm = properties.sharedMemPerMultiprocessor;
  if (shared_per_sm == 0)
    throw std::runtime_error("cotenant: device reports zero shared memory per SM");

  std::size_t candidate =
      (shared_per_sm + static_cast<std::size_t>(requested_max_occupancy) - 1) /
      static_cast<std::size_t>(requested_max_occupancy);
  candidate = std::min(candidate, max_dynamic_bytes);

  int candidate_occupancy = query_occupancy(candidate);
  if (candidate_occupancy == 0) {
    // Find the largest launchable reservation at or below the desired size.
    std::size_t low = 1;
    std::size_t high = candidate;
    std::size_t best = 0;
    int best_occupancy = 0;
    while (low <= high) {
      const std::size_t mid = low + (high - low) / 2;
      const int occupancy = query_occupancy(mid);
      if (occupancy > 0) {
        best = mid;
        best_occupancy = occupancy;
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }
    candidate = best;
    candidate_occupancy = best_occupancy;
  }

  if (candidate_occupancy > requested_max_occupancy) {
    // CUDA shared-memory allocation is architecture-specific. Find the smallest
    // larger reservation whose reported occupancy enforces the requested cap.
    std::size_t low = candidate + 1;
    std::size_t high = max_dynamic_bytes;
    std::size_t best = 0;
    int best_occupancy = 0;
    while (low <= high) {
      const std::size_t mid = low + (high - low) / 2;
      const int occupancy = query_occupancy(mid);
      if (occupancy == 0 || occupancy <= requested_max_occupancy) {
        if (occupancy > 0) {
          best = mid;
          best_occupancy = occupancy;
        }
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    }
    candidate = best;
    candidate_occupancy = best_occupancy;
  }

  if (candidate == 0 || candidate_occupancy < 1) {
    throw std::runtime_error(
        "cotenant: no launchable shared-memory reservation satisfies the requested occupancy");
  }
  if (candidate_occupancy > requested_max_occupancy) {
    throw std::runtime_error(
        "cotenant: the device cannot enforce the requested occupancy with shared memory");
  }

  return {candidate, candidate_occupancy};
}

void log_configuration_once(int device, const cudaDeviceProp& properties,
                            const cuda_cotenant_config& config,
                            const reservation& selected) {
  using key_type = std::tuple<int, int, int, std::size_t, int>;
  static std::mutex seen_mutex;
  static std::set<key_type> seen;

  const key_type key{device, config.workgroups, config.max_occupancy,
                     selected.bytes, selected.blocks_per_sm};
  {
    const std::lock_guard<std::mutex> lock(seen_mutex);
    if (!seen.insert(key).second)
      return;
  }

  std::ostringstream message;
  message << "cotenant: device=" << device << " (" << properties.name
          << ") grid=" << config.workgroups << " block=" << kBlockSize
          << " max_occupancy=" << config.max_occupancy
          << " max_blocks_per_sm=" << selected.blocks_per_sm
          << " shared_reserved=" << selected.bytes << '/'
          << properties.sharedMemPerMultiprocessor;
  log_message(message.str());
}

}  // namespace

class scoped_cuda_cotenant::impl {
 public:
  impl(cuda_cotenant_config config, cudaStream_t gemm_stream)
      : config_(config), gemm_stream_(gemm_stream) {
    try {
      start();
    } catch (...) {
      stop(true);
      throw;
    }
  }

  ~impl() {
    stop(false);
  }

 private:
  void start() {
    if (config_.workgroups < 1)
      throw std::invalid_argument("cotenant: workgroups must be greater than zero");
    if (config_.max_occupancy < 1 ||
        config_.max_occupancy > kMaxRequestedOccupancy) {
      throw std::invalid_argument("cotenant: max occupancy must be in [1, 64]");
    }
    if (config_.ready_timeout.count() < 0)
      throw std::invalid_argument("cotenant: readiness timeout must be nonnegative");
    if (gemm_stream_ == nullptr)
      throw std::invalid_argument("cotenant: GEMM stream must not be null");

    check_cuda(cudaGetDevice(&device_), "querying current device");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device_),
               "querying device properties");
    if (config_.workgroups >= properties.multiProcessorCount) {
      throw std::invalid_argument(
          "cotenant: workgroup count must be less than the device SM count (" +
          std::to_string(properties.multiProcessorCount) + ')');
    }

    const reservation selected =
        choose_reservation(properties, config_.max_occupancy);
    shared_bytes_ = selected.bytes;

    check_cuda(cudaStreamSynchronize(gemm_stream_),
               "synchronizing GEMM setup before cotenant launch");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ready_), sizeof(*ready_)),
               "allocating readiness counter");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&stop_), sizeof(*stop_)),
               "allocating stop flag");
    check_cuda(cudaMemset(ready_, 0, sizeof(*ready_)),
               "initializing readiness counter");
    check_cuda(cudaMemset(stop_, 0, sizeof(*stop_)), "initializing stop flag");
    check_cuda(cudaStreamCreateWithFlags(&cotenant_stream_, cudaStreamNonBlocking),
               "creating cotenant stream");

    cotenant_kernel<<<config_.workgroups, kBlockSize, shared_bytes_,
                      cotenant_stream_>>>(ready_, stop_);
    check_cuda(cudaGetLastError(), "launching cotenant kernel");
    launched_ = true;

    check_driver(cuStreamCreate(&control_stream_, CU_STREAM_NON_BLOCKING),
                 "creating readiness stream");
    check_driver(cuStreamWaitValue32(control_stream_,
                                     reinterpret_cast<CUdeviceptr>(ready_),
                                     static_cast<cuuint32_t>(config_.workgroups),
                                     CU_STREAM_WAIT_VALUE_GEQ),
                 "waiting for cotenant residency");
    wait_ready();
    log_configuration_once(device_, properties, config_, selected);
  }

  void wait_ready() {
    const auto deadline = std::chrono::steady_clock::now() + config_.ready_timeout;
    for (;;) {
      const CUresult status = cuStreamQuery(control_stream_);
      if (status == CUDA_SUCCESS)
        return;
      if (status != CUDA_ERROR_NOT_READY)
        throw std::runtime_error(driver_error(status, "querying cotenant readiness"));
      if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error(
            "cotenant: timed out waiting for workgroup residency");
      }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  }

  void stop(bool abort_wait) noexcept {
    if (stopped_)
      return;
    stopped_ = true;

    if (launched_) {
      cudaError_t status = cudaMemsetAsync(stop_, 1, sizeof(*stop_), gemm_stream_);
      if (status == cudaSuccess && abort_wait) {
        const unsigned int ready_target =
            static_cast<unsigned int>(config_.workgroups);
        status = cudaMemcpyAsync(ready_, &ready_target, sizeof(ready_target),
                                 cudaMemcpyHostToDevice, gemm_stream_);
      }
      if (status == cudaSuccess)
        status = cudaStreamSynchronize(gemm_stream_);
      if (status != cudaSuccess) {
        log_message(std::string("cotenant: stop signal failed: ") +
                    cudaGetErrorString(status));
        std::terminate();
      }
    }

    if (cotenant_stream_ != nullptr) {
      const cudaError_t status = cudaStreamSynchronize(cotenant_stream_);
      if (status != cudaSuccess) {
        log_message(std::string("cotenant: cotenant stream failed: ") +
                    cudaGetErrorString(status));
      }
    }
    if (control_stream_ != nullptr) {
      const CUresult status = cuStreamSynchronize(control_stream_);
      if (status != CUDA_SUCCESS)
        log_message(driver_error(status, "synchronizing readiness stream"));
      (void)cuStreamDestroy(control_stream_);
      control_stream_ = nullptr;
    }
    if (cotenant_stream_ != nullptr) {
      (void)cudaStreamDestroy(cotenant_stream_);
      cotenant_stream_ = nullptr;
    }
    if (stop_ != nullptr) {
      (void)cudaFree(stop_);
      stop_ = nullptr;
    }
    if (ready_ != nullptr) {
      (void)cudaFree(ready_);
      ready_ = nullptr;
    }
  }

  cuda_cotenant_config config_;
  cudaStream_t gemm_stream_ = nullptr;
  cudaStream_t cotenant_stream_ = nullptr;
  CUstream control_stream_ = nullptr;
  unsigned int* ready_ = nullptr;
  unsigned int* stop_ = nullptr;
  std::size_t shared_bytes_ = 0;
  int device_ = 0;
  bool launched_ = false;
  bool stopped_ = false;
};

scoped_cuda_cotenant::scoped_cuda_cotenant(cuda_cotenant_config config,
                                           cudaStream_t gemm_stream) {
  if (config.workgroups != 0)
    impl_ = std::make_unique<impl>(config, gemm_stream);
}

scoped_cuda_cotenant::~scoped_cuda_cotenant() noexcept = default;
