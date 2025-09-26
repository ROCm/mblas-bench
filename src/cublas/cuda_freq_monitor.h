#pragma once

#include "cuda_error.h"

#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstdint>

#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdlib>


namespace cuda_frequency {

class monitor
{

private:
    // Constants for frequency conversion
    static constexpr double MHz_TO_Hz = 1000000.0;
    static constexpr double Hz_TO_MHz = 1.0 / MHz_TO_Hz;
    static constexpr int SAMPLING_INTERVAL_MS = 50;

    // Thread management
    std::thread monitoring_thread;
    std::atomic<bool> should_stop{false};
    std::atomic<bool> should_exit{false};
    std::mutex data_mutex;
    std::condition_variable cv;
    std::packaged_task<void()> monitoring_task;
    std::future<void> monitoring_future;

    // Device information
    nvmlDevice_t nvml_device;
    int cuda_device_id;

    // Frequency data storage
    std::vector<uint64_t> gpu_frequencies;  // in Hz
    std::vector<uint64_t> mem_frequencies;  // in Hz
    uint64_t gpu_freq_sum = 0;
    uint64_t mem_freq_sum = 0;

    bool monitoring_active = false;

public:
    monitor(const monitor&) = delete;

    monitor() {
        init_nvml();
        init_thread();
    }

    ~monitor() {
        should_exit = true;
        cv.notify_all();
        if (monitoring_thread.joinable()) {
            monitoring_thread.join();
        }
    }

    void set_device_id(int device_id) {
        /* Device ID here is physical device ID, not CUDA_VISIBLE_DEVICES index */ 
        cuda_device_id = device_id;
        check_nvml(nvmlDeviceGetHandleByIndex(cuda_device_id, &nvml_device));
    }

    bool enabled() const {
        const char* freq_env = std::getenv("CUBLAS_BENCH_FREQ");
        return freq_env != nullptr;
    }

    void start() {
        if (!enabled()) return;
        
        clear();
        run();
    }

    void stop() {
        if (!enabled()) return;
        
        if (monitoring_active) {
            should_stop = true;
            wait();
        }
    }

    float get_avg_sysclk_mhz() const {
        if (gpu_frequencies.empty()) return 0.0;
        return (static_cast<float>(gpu_freq_sum) / gpu_frequencies.size()) * Hz_TO_MHz;
    }

    float get_med_sysclk_mhz() const {
        if (gpu_frequencies.empty()) return 0.0;
        
        auto freq_copy = gpu_frequencies;
        std::sort(freq_copy.begin(), freq_copy.end());
        
        size_t n = freq_copy.size();
        double median_hz;
        if (n % 2 == 0) {
            median_hz = (freq_copy[n/2 - 1] + freq_copy[n/2]) / 2.0;
        } else {
            median_hz = freq_copy[n/2];
        }
        return static_cast<float>(median_hz * Hz_TO_MHz);
    }

    float get_avg_memclk_mhz() const {
        if (mem_frequencies.empty()) return 0.0;
        return (static_cast<float>(mem_freq_sum) / mem_frequencies.size()) * Hz_TO_MHz;
    }

    float get_med_memclk_mhz() const {
        if (mem_frequencies.empty()) return 0.0;
        
        auto freq_copy = mem_frequencies;
        std::sort(freq_copy.begin(), freq_copy.end());
        
        size_t n = freq_copy.size();
        double median_hz;
        if (n % 2 == 0) {
            median_hz = (freq_copy[n/2 - 1] + freq_copy[n/2]) / 2.0;
        } else {
            median_hz = freq_copy[n/2];
        }
        return static_cast<float>(median_hz * Hz_TO_MHz);
    }

private:

    void init_nvml() {
        static bool nvml_initialized = false;
        if (!nvml_initialized) {
            check_nvml(nvmlInit());
            nvml_initialized = true;
        }
    }

    void init_thread() {
        monitoring_thread = std::thread([this]() {
            std::unique_lock<std::mutex> lock(data_mutex);
            while (!should_exit) {
                while (!monitoring_task.valid() && !should_exit) {
                    cv.wait(lock);
                }
                
                if (should_exit) break;
                
                monitoring_task();
                monitoring_task = std::packaged_task<void()>();
            }
        });
    }

    void run() {
        if (monitoring_active) return;
        
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            monitoring_task = std::packaged_task<void()>([this]() {
                collect();
            });
            monitoring_future = monitoring_task.get_future();
            should_stop = false;
            monitoring_active = true;
        }
        cv.notify_all();
    }

    void collect() {
        while (!should_stop && !should_exit) {
            unsigned int gpu_clock, mem_clock;
            
            // Sample GPU core frequency
            nvmlReturn_t gpu_result = nvmlDeviceGetClockInfo(nvml_device, NVML_CLOCK_GRAPHICS, &gpu_clock);
            if (gpu_result == NVML_SUCCESS) {
                uint64_t gpu_hz = static_cast<uint64_t>(gpu_clock) * static_cast<uint64_t>(MHz_TO_Hz);
                gpu_frequencies.push_back(gpu_hz);
                gpu_freq_sum += gpu_hz;
            }

            // Sample memory frequency
            nvmlReturn_t mem_result = nvmlDeviceGetClockInfo(nvml_device, NVML_CLOCK_MEM, &mem_clock);
            if (mem_result == NVML_SUCCESS) {
                uint64_t mem_hz = static_cast<uint64_t>(mem_clock) * static_cast<uint64_t>(MHz_TO_Hz);
                mem_frequencies.push_back(mem_hz);
                mem_freq_sum += mem_hz;
            }

            // Wait before next sample
            std::this_thread::sleep_for(std::chrono::milliseconds(SAMPLING_INTERVAL_MS));
        }
    }

    void clear() {
        gpu_frequencies.clear();
        mem_frequencies.clear();
        gpu_freq_sum = 0;
        mem_freq_sum = 0;
    }

    void wait() {
        if (!monitoring_future.valid()) return;
        
        monitoring_future.wait();
        monitoring_future = std::future<void>();
        monitoring_active = false;
    }
};

} // namespace cuda_frequency
