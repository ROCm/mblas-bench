#pragma once
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "adaptive_timing.h"
#include "hip_error.h"

// Ring of chained boundary events: sample k spans events[k] to events[k + 1], so
// back-to-back samples share an event and nothing is enqueued between them.
template <typename Launch>
class hip_sample_timer {
 public:
  hip_sample_timer(Launch &launch, hipStream_t stream, int32_t depth)
      : launch_(launch), stream_(stream), events_(depth + 1) {
    for (auto &e : events_) check_hip(hipEventCreate(&e));
  }
  ~hip_sample_timer() {
    for (auto e : events_) hipEventDestroy(e);
  }
  hip_sample_timer(const hip_sample_timer &) = delete;
  hip_sample_timer &operator=(const hip_sample_timer &) = delete;

  void enqueue(int64_t first, int32_t count) {
    for (int32_t i = 0; i < count; i++) launch_(first + i);
  }
  void restart() {
    tail_ = head_;
    record();
  }
  void submit(int64_t first, int32_t count) {
    enqueue(first, count);
    record();
  }
  double wait() {
    hipEvent_t start = slot(tail_), stop = slot(tail_ + 1);
    check_hip(hipEventSynchronize(stop));
    float elapsed_ms;
    check_hip(hipEventElapsedTime(&elapsed_ms, start, stop));
    ++tail_;
    return static_cast<double>(elapsed_ms) * 1000.0;
  }

 private:
  void record() {
    // Re-recording the oldest unread start event would corrupt its sample.
    if (head_ - tail_ >= static_cast<int64_t>(events_.size()))
      throw std::logic_error("hip_sample_timer: more samples pending than pipeline depth");
    check_hip(hipEventRecord(slot(head_++), stream_));
  }
  hipEvent_t slot(int64_t i) const { return events_[i % events_.size()]; }

  Launch &launch_;
  hipStream_t stream_;
  std::vector<hipEvent_t> events_;
  int64_t head_ = 0;  // next event to record
  int64_t tail_ = 0;  // start event of the oldest unread sample
};

// Time `launch(i)` on `stream` with HIP events. `launch` performs exactly one
// enqueue for global index `i` and handles any rotating-buffer indexing.
template <typename Launch>
void hip_measure(Launch &&launch, hipStream_t stream, const timing_config &cfg,
                 timing_result &out) {
  hip_sample_timer<std::remove_reference_t<Launch>> timer(launch, stream,
                                                          timing_pipeline_depth(cfg));
  run_measurement(timer, cfg, out);
  check_hip(hipGetLastError());
}
