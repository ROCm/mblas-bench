#pragma once
// Adaptive, distribution-based timing. Ported from hipBLASLt-bench
// (ROCm/rocm-libraries#8337, clients/common/include/benchmark_{stats,timing}.hpp).
//
// A *sample* is one timed span over `iters_per_sample` back-to-back enqueues,
// yielding one per-iteration time. iters_per_sample is sized from a warmup so each
// sample spans ~sample_time. Samples are collected for at least the floor (min_iters and
// measure_time); past the floor the run stops when either (a) the mean's relative
// standard error drops below noise_threshold (converged), or (b) the robust spread
// (rel_iqr) has plateaued (stable) -- a fallback for heavy-tailed/unlocked-clock
// kernels that never hit the precision target. Otherwise it runs to the
// max_measure_time / max_iters ceiling (noisy). noise_threshold <= 0 disables both
// checks. The median is the headline statistic.
//
// Measure-phase samples are pipelined: pipeline_depth samples stay queued on the
// stream, and consecutive samples share a boundary event, so the GPU runs them back
// to back while the host processes results. Every timed span starts behind untimed
// queued work (cold iters, or a primer sample) rather than on an idle GPU.
//
// This header has no vendor dependencies: the caller supplies a timer with
//   void enqueue(int64_t first, int32_t count)  // enqueue [first, first + count), untimed
//   void restart()                              // record a new chain-start event
//   void submit(int64_t first, int32_t count)   // enqueue, then record the next boundary
//   double wait()                               // block on the oldest pending sample; us
// holding at least timing_pipeline_depth(cfg) pending samples.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// All time budgets are in milliseconds.
struct timing_config {
  bool adaptive = false;  // false => fixed-count path (cold_iters + iters)
  int32_t iters = 10;
  int32_t cold_iters = 2;
  float warmup_time = 0.0f;       // warm up until this wall-time is reached
  float sample_time = 0.0f;       // target span per sample; 0 => one enqueue per sample
  float measure_time = 0.0f;      // minimum total measure time (floor)
  float max_measure_time = 0.0f;  // measure ceiling; 0 => unbounded
  int32_t min_iters = 0;          // floor on total timed iterations
  int32_t max_iters = 0;          // ceiling on total timed iterations; 0 => unbounded
  float noise_threshold = 0.0f;   // rel. std error convergence target; 0 => disabled
  // Noise-plateau fallback: past the floor, end the run if convergence cannot be
  // reached but the robust spread (rel_iqr) has settled. 0 threshold disables it.
  float stability_threshold = 0.0f;  // max rel. spread of recent rel_iqr readings
  int32_t stability_window = 0;      // rel_iqr readings tested for the plateau (>= 2)
  int32_t stability_interval = 0;    // record a rel_iqr reading every N samples (>= 1)
  int32_t pipeline_depth = 1;        // measure samples kept in flight (>= 1)
};

// Defaults for the --adaptive preset; used as the CLI defaults so they appear in --help.
namespace adaptive_defaults {
constexpr float warmup_time = 50.0f;
constexpr float sample_time = 1.0f;
constexpr float measure_time = 500.0f;
constexpr float max_measure_time = 2000.0f;
constexpr int32_t min_iters = 10;
constexpr int32_t max_iters = 0;
constexpr float noise_threshold = 0.01f;
// window * interval = 512-sample look-back, so the fallback can only fire ~512
// samples past the measure_time floor: kernels that converge first report
// "converged", only those that cannot report "stable".
constexpr float stability_threshold = 0.05f;
constexpr int32_t stability_window = 32;
constexpr int32_t stability_interval = 16;
constexpr int32_t pipeline_depth = 2;
}  // namespace adaptive_defaults

// Pending samples the timer must hold. Fixed-count mode never pipelines.
inline int32_t timing_pipeline_depth(const timing_config &cfg) {
  return cfg.adaptive ? std::max<int32_t>(1, cfg.pipeline_depth) : 1;
}

// Per-iteration statistics, in microseconds.
struct timing_result {
  double median_us = 0.0;
  double min_us = 0.0;
  double mean_us = 0.0;
  double cv = 0.0;       // stddev / mean across samples (n-1)
  double rel_iqr = 0.0;  // interquartile range / median
  int32_t iters_per_sample = 0;  // enqueues per sample
  int32_t samples = 0;           // number of samples collected
  int64_t hot_iters = 0;         // total timed enqueues (iters_per_sample * samples)
  bool adaptive = false;
  bool noise_active = false;  // convergence checking was enabled (noise_threshold > 0)
  bool converged = false;     // precision target met
  bool stable = false;        // rel_iqr plateau reached without convergence
};

inline const char *timing_status(const timing_result &r) {
  if (!r.noise_active) return "-";
  if (r.converged) return "converged";
  if (r.stable) return "stable";
  return "noisy";
}

// Returns an empty string on success, or a description of the first problem.
inline std::string validate_adaptive_config(const timing_config &cfg) {
  if (cfg.warmup_time < 0.0f)
    return "warmup_time (" + std::to_string(cfg.warmup_time) + " ms) must be >= 0";
  if (cfg.sample_time < 0.0f)
    return "sample_time (" + std::to_string(cfg.sample_time) + " ms) must be >= 0";
  if (cfg.measure_time < 0.0f)
    return "measure_time (" + std::to_string(cfg.measure_time) + " ms) must be >= 0";
  if (cfg.max_measure_time < 0.0f)
    return "max_measure_time (" + std::to_string(cfg.max_measure_time) + " ms) must be >= 0";
  if (cfg.min_iters < 0)
    return "min_iters (" + std::to_string(cfg.min_iters) + ") must be >= 0";
  if (cfg.max_iters < 0)
    return "max_iters (" + std::to_string(cfg.max_iters) + ") must be >= 0";
  if (cfg.noise_threshold < 0.0f)
    return "noise_threshold (" + std::to_string(cfg.noise_threshold) + ") must be >= 0";
  if (cfg.stability_threshold < 0.0f)
    return "stability_threshold (" + std::to_string(cfg.stability_threshold) + ") must be >= 0";
  if (cfg.max_iters > 0 && cfg.min_iters > cfg.max_iters)
    return "min_iters (" + std::to_string(cfg.min_iters) + ") > max_iters (" +
           std::to_string(cfg.max_iters) + ")";
  if (cfg.max_measure_time > 0.0f && cfg.measure_time > cfg.max_measure_time)
    return "measure_time (" + std::to_string(cfg.measure_time) + " ms) > max_measure_time (" +
           std::to_string(cfg.max_measure_time) + " ms)";
  if (cfg.max_measure_time <= 0.0f && cfg.max_iters <= 0)
    return "adaptive timing requires a ceiling: set max_measure_time or max_iters > 0";
  if (cfg.stability_threshold > 0.0f && (cfg.stability_window < 2 || cfg.stability_interval < 1))
    return "stability_threshold > 0 requires stability_window >= 2 and stability_interval >= 1";
  if (cfg.pipeline_depth < 1)
    return "pipeline_depth (" + std::to_string(cfg.pipeline_depth) + ") must be >= 1";
  return {};
}

// One-line summary of a resolved adaptive configuration. No trailing newline.
inline std::string format_adaptive_timing_summary(const timing_config &cfg) {
  std::ostringstream oss;
  oss << "Adaptive timing: warmup " << cfg.warmup_time << "ms. Sample ";
  if (cfg.sample_time == 0.0f)
    oss << "single-enqueue";
  else
    oss << cfg.sample_time << "ms";
  oss << ". Measure " << cfg.measure_time << "ms (max ";
  if (cfg.max_measure_time == 0.0f)
    oss << "unbounded";
  else
    oss << cfg.max_measure_time << "ms";
  oss << "). Iters " << cfg.min_iters << " (max ";
  if (cfg.max_iters == 0)
    oss << "unbounded";
  else
    oss << cfg.max_iters;
  oss << "). Noise threshold ";
  if (cfg.noise_threshold == 0.0f)
    oss << "disabled";
  else
    oss << cfg.noise_threshold * 100 << "%";
  if (cfg.stability_threshold == 0.0f)
    oss << " (stability fallback disabled)";
  else
    oss << " (stability " << cfg.stability_threshold * 100 << "%, window "
        << cfg.stability_window << ", interval " << cfg.stability_interval << ")";
  oss << ". Pipeline depth " << cfg.pipeline_depth << ".";
  return oss.str();
}

// Combine per-device results into one row: worst status, worst dispersion, mean
// latencies, and the smallest sample counts.
inline timing_result aggregate(const std::vector<timing_result> &results) {
  timing_result out;
  if (results.empty()) return out;
  out = results.front();
  double median = 0.0, mean = 0.0, min = 0.0;
  for (const auto &r : results) {
    median += r.median_us;
    mean += r.mean_us;
    min += r.min_us;
    out.cv = std::max(out.cv, r.cv);
    out.rel_iqr = std::max(out.rel_iqr, r.rel_iqr);
    out.iters_per_sample = std::min(out.iters_per_sample, r.iters_per_sample);
    out.samples = std::min(out.samples, r.samples);
    out.hot_iters = std::min(out.hot_iters, r.hot_iters);
    out.converged = out.converged && r.converged;
  }
  out.stable = !out.converged &&
               std::all_of(results.begin(), results.end(),
                           [](const timing_result &r) { return r.converged || r.stable; });
  out.median_us = median / results.size();
  out.mean_us = mean / results.size();
  out.min_us = min / results.size();
  return out;
}

namespace timing_detail {

inline double mean_of(const std::vector<double> &v) {
  return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

inline double sum_sq_dev(const std::vector<double> &v, double mean) {
  return std::accumulate(v.begin(), v.end(), 0.0,
                         [mean](double acc, double x) { return acc + (x - mean) * (x - mean); });
}

// Incremental mean/variance via Welford's recurrence: O(1) per sample.
struct running_stats {
  int64_t count = 0;
  double mean = 0.0;
  double m2 = 0.0;

  void add(double x) {
    ++count;
    const double delta = x - mean;
    mean += delta / static_cast<double>(count);
    m2 += delta * (x - mean);
  }

  // (sample stddev / mean) / sqrt(n)
  double relative_std_error() const {
    if (count < 2 || mean <= 0.0) return std::numeric_limits<double>::infinity();
    const double var = m2 / static_cast<double>(count - 1);
    return (std::sqrt(var) / mean) / std::sqrt(static_cast<double>(count));
  }
};

// A stddev from fewer samples is unreliable.
constexpr int min_samples_for_convergence = 10;

// Linear-interpolated quantile of a sorted, non-empty vector (q in [0, 1]).
inline double quantile(const std::vector<double> &sorted, double q) {
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - static_cast<double>(lo));
}

// IQR / median over all samples so far. Sorts `samples` in place; nothing relies
// on its order. `samples` must be non-empty.
inline double cumulative_rel_iqr(std::vector<double> &samples) {
  std::sort(samples.begin(), samples.end());
  const double median = quantile(samples, 0.5);
  const double iqr = quantile(samples, 0.75) - quantile(samples, 0.25);
  return median > 0.0 ? iqr / median : 0.0;
}

// Whether the last `window` rel_iqr readings have a relative stddev below `threshold`.
inline bool noise_plateaued(const std::vector<double> &iqr_history, int window, double threshold) {
  if (window < 2 || static_cast<int>(iqr_history.size()) < window) return false;
  const std::vector<double> recent(iqr_history.end() - window, iqr_history.end());
  const double mean = mean_of(recent);
  if (mean <= 0.0) return false;
  const double rel_spread = std::sqrt(sum_sq_dev(recent, mean) / (recent.size() - 1)) / mean;
  return rel_spread < threshold;
}

// Keep room for at least two samples within max_iters; unbounded when unset.
inline int64_t max_iters_per_sample(const timing_config &cfg) {
  return cfg.max_iters > 0 ? std::max<int64_t>(1, cfg.max_iters / 2)
                           : std::numeric_limits<int32_t>::max();
}

// Enqueues per sample so that one sample spans ~sample_time. In [1, cap].
inline int32_t iters_per_sample(const timing_config &cfg, double per_iter_us, int64_t cap) {
  if (cfg.sample_time <= 0.0f || per_iter_us <= 0.0) return 1;
  const double want = std::ceil(static_cast<double>(cfg.sample_time) * 1000.0 / per_iter_us);
  return static_cast<int32_t>(
      std::clamp(static_cast<int64_t>(want), static_cast<int64_t>(1), cap));
}

inline void fill_distribution(const std::vector<double> &samples, timing_result &out) {
  std::vector<double> sorted(samples);
  std::sort(sorted.begin(), sorted.end());
  const size_t n = sorted.size();
  const double mean = mean_of(samples);
  const double var = n > 1 ? sum_sq_dev(samples, mean) / (n - 1) : 0.0;
  const double median = quantile(sorted, 0.5);
  const double iqr = quantile(sorted, 0.75) - quantile(sorted, 0.25);

  out.mean_us = mean;
  out.median_us = median;
  out.min_us = sorted.front();
  out.cv = mean > 0.0 ? std::sqrt(var) / mean : 0.0;
  out.rel_iqr = median > 0.0 ? iqr / median : 0.0;
  out.samples = static_cast<int32_t>(n);
}

struct measure_bounds {
  int64_t min_iters;  // floor: minimum total timed iterations
  double min_us;      // floor: minimum total measured time
  double max_us;      // ceiling: maximum total measured time (0 = none)
};

// Whether the measure loop should stop after the latest sample. Floors take
// precedence: the ceiling can only fire after the floor is satisfied.
inline bool reached_target(const timing_config &cfg, std::vector<double> &samples,
                           const running_stats &stats, std::vector<double> &iqr_history,
                           int64_t total_iters, double total_us, const measure_bounds &bounds,
                           bool &converged, bool &stable) {
  const bool floor_met = total_iters >= bounds.min_iters && total_us >= bounds.min_us;
  const int n = static_cast<int>(samples.size());
  if (floor_met && cfg.noise_threshold > 0.0f && n >= min_samples_for_convergence) {
    converged = stats.relative_std_error() < cfg.noise_threshold;
    const bool fallback_on = cfg.stability_threshold > 0.0f && cfg.stability_interval >= 1 &&
                             cfg.stability_window >= 2;
    if (!converged && fallback_on && n % cfg.stability_interval == 0) {
      iqr_history.push_back(cumulative_rel_iqr(samples));
      stable = noise_plateaued(iqr_history, cfg.stability_window, cfg.stability_threshold);
    }
  }
  const bool ceiling = (bounds.max_us > 0.0 && total_us >= bounds.max_us) ||
                       (cfg.max_iters > 0 && total_iters >= cfg.max_iters);
  return floor_met && (converged || stable || ceiling);
}

}  // namespace timing_detail

// Fixed-count mode queues cfg.cold_iters untimed, then times one span of cfg.iters.
// Adaptive mode self-sizes iters_per_sample and collects samples until the mean
// converges, the robust spread plateaus, or a ceiling is hit.
template <typename Timer>
void run_measurement(Timer &timer, const timing_config &cfg, timing_result &out) {
  out = timing_result{};

  // Traditional fixed iteration mode
  if (!cfg.adaptive) {
    if (cfg.cold_iters > 0) {
      // Warmup iters 
      timer.enqueue(0, cfg.cold_iters);
    }
    const int32_t iters = cfg.iters > 0 ? cfg.iters : 1;
    timer.restart();
    // Benchmark iterations
    timer.submit(0, iters);
    const double per_iter = timer.wait() / iters;
    out.mean_us = per_iter;
    out.median_us = per_iter;
    out.min_us = per_iter;
    out.iters_per_sample = iters;
    out.samples = 1;
    out.hot_iters = iters;
    return;
  }

  // Adaptive timing mode from here on out
  if (const auto err = validate_adaptive_config(cfg); !err.empty()) {
    throw std::invalid_argument("Invalid adaptive timing config: " + err);
  }

  // Warmup: one cold probe seeds iters_per_sample, then warm up in self-sized chunks
  // until warmup_time, refining the per-iteration estimate. Each chunk's time is
  // needed before sizing the next, so warmup is synchronous.
  const auto time_iters = [&timer](int64_t first, int32_t count) {
    timer.restart();
    timer.submit(first, count);
    return timer.wait();
  };
  int64_t rotating_index = 0;
  const int64_t cap = timing_detail::max_iters_per_sample(cfg);
  double per_iter_est = time_iters(rotating_index++, 1);
  {
    const double warm_min_us = static_cast<double>(cfg.warmup_time) * 1000.0;
    double warm_us = 0.0;
    int32_t chunk = timing_detail::iters_per_sample(cfg, per_iter_est, cap);
    while (warm_us < warm_min_us) {
      const double chunk_us = time_iters(rotating_index, chunk);
      rotating_index += chunk;
      warm_us += chunk_us;
      // A chunk below timer resolution can't advance warm_us, so grow the chunk
      // instead; give up if even the max chunk still reads as zero.
      if (chunk_us > 0.0) {
        per_iter_est = chunk_us / chunk;
        chunk = timing_detail::iters_per_sample(cfg, per_iter_est, cap);
      } else if (chunk < cap) {
        chunk = static_cast<int32_t>(std::min<int64_t>(cap, int64_t{chunk} * 2));
      } else {
        break;
      }
    }
  }

  const int32_t iters_per_sample = timing_detail::iters_per_sample(cfg, per_iter_est, cap);
  const timing_detail::measure_bounds bounds{std::max<int64_t>(1, cfg.min_iters),
                                             static_cast<double>(cfg.measure_time) * 1000.0,
                                             static_cast<double>(cfg.max_measure_time) * 1000.0};

  std::vector<double> samples;
  std::vector<double> iqr_history;
  timing_detail::running_stats stats;
  double total_us = 0.0;
  int64_t total_iters = 0;
  bool converged = false;
  bool stable = false;

  // The untimed primer keeps the GPU busy while the first timed samples are queued.
  const int32_t depth = timing_pipeline_depth(cfg);
  timer.enqueue(rotating_index, iters_per_sample);
  rotating_index = 0;  // restart rotation for the timed phase
  timer.restart();
  for (int32_t d = 0; d < depth; ++d) {
    timer.submit(rotating_index, iters_per_sample);
    rotating_index += iters_per_sample;
  }
  while (true) {
    const double sample_us = timer.wait();
    const double per_iter = sample_us / iters_per_sample;
    samples.push_back(per_iter);
    stats.add(per_iter);
    total_us += sample_us;
    total_iters += iters_per_sample;
    if (timing_detail::reached_target(cfg, samples, stats, iqr_history, total_iters, total_us,
                                      bounds, converged, stable)) {
      break;
    }
    timer.submit(rotating_index, iters_per_sample);
    rotating_index += iters_per_sample;
  }
  // Samples still in flight overshoot the stop decision; drain and discard them.
  for (int32_t d = 1; d < depth; ++d) {
    timer.wait();
  }

  timing_detail::fill_distribution(samples, out);
  out.iters_per_sample = iters_per_sample;
  out.hot_iters = total_iters;
  out.adaptive = true;
  out.noise_active = cfg.noise_threshold > 0.0f;
  out.converged = converged;
  out.stable = stable;
}
