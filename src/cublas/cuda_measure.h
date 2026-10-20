#pragma once
#include <cuda_runtime.h>

#include <cstdint>

#include "adaptive_timing.h"
#include "cuda_error.h"

// Time `launch(i)` on `stream` with CUDA events. `launch` performs exactly one
// enqueue for global index `i` and handles any rotating-buffer indexing.
template <typename Launch>
void cuda_measure(Launch &&launch, cudaStream_t stream, const timing_config &cfg,
                  timing_result &out) {
  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start));
  check_cuda(cudaEventCreate(&stop));
  run_measurement(
      [&](int64_t first, int32_t count) {
        check_cuda(cudaEventRecord(start, stream));
        for (int32_t i = 0; i < count; i++) launch(first + i);
        check_cuda(cudaEventRecord(stop, stream));
        check_cuda(cudaEventSynchronize(stop));
        float elapsed_ms;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop));
        return static_cast<double>(elapsed_ms) * 1000.0;
      },
      cfg, out);
  check_cuda(cudaGetLastError());
  check_cuda(cudaEventDestroy(start));
  check_cuda(cudaEventDestroy(stop));
}
