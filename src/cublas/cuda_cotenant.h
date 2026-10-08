#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <memory>

struct cuda_cotenant_config {
  int workgroups = 0;
  int max_occupancy = 1;
  std::chrono::milliseconds ready_timeout{30000};
};

// Owns a persistent shared-memory cotenant kernel for one CUDA device.
// Construction waits for every workgroup to become resident. Destruction
// orders shutdown after all work already queued on gemm_stream.
class scoped_cuda_cotenant {
 public:
  scoped_cuda_cotenant(cuda_cotenant_config config, cudaStream_t gemm_stream);
  ~scoped_cuda_cotenant() noexcept;

  scoped_cuda_cotenant(const scoped_cuda_cotenant&) = delete;
  scoped_cuda_cotenant& operator=(const scoped_cuda_cotenant&) = delete;

 private:
  class impl;
  std::unique_ptr<impl> impl_;
};
