#pragma once
#include <hip/hip_runtime.h>

#include <cstdint>

#include "adaptive_timing.h"
#include "hip_error.h"

// Time `launch(i)` on `stream` with HIP events. `launch` performs exactly one
// enqueue for global index `i` and handles any rotating-buffer indexing.
template <typename Launch>
void hip_measure(Launch &&launch, hipStream_t stream, const timing_config &cfg,
                 timing_result &out) {
  hipEvent_t start, stop;
  check_hip(hipEventCreate(&start));
  check_hip(hipEventCreate(&stop));
  run_measurement(
      [&](int64_t first, int32_t count) {
        check_hip(hipEventRecord(start, stream));
        for (int32_t i = 0; i < count; i++) launch(first + i);
        check_hip(hipEventRecord(stop, stream));
        check_hip(hipEventSynchronize(stop));
        float elapsed_ms;
        check_hip(hipEventElapsedTime(&elapsed_ms, start, stop));
        return static_cast<double>(elapsed_ms) * 1000.0;
      },
      cfg, out);
  check_hip(hipGetLastError());
  check_hip(hipEventDestroy(start));
  check_hip(hipEventDestroy(stop));
}
