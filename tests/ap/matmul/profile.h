#pragma once

#include "matmul.h"
#include <cuda_profiler_api.h>
#include <functional>

namespace ap {

class GpuTimer {
public:
  explicit GpuTimer(bool profile) : profile_(profile) {
    CHECK_CUDA(cudaEventCreate(&start_));
    CHECK_CUDA(cudaEventCreate(&stop_));
  }

  ~GpuTimer() {
    CHECK_CUDA(cudaEventDestroy(start_));
    CHECK_CUDA(cudaEventDestroy(stop_));
  }

  void Start(cudaStream_t stream) {
    CHECK_CUDA(cudaEventRecord(start_, stream));
    if (profile_) {
      CHECK_CUDA(cudaProfilerStart());
    }
  }

  void Stop(cudaStream_t stream) {
    CHECK_CUDA(cudaEventRecord(stop_, stream));
    if (profile_) {
      CHECK_CUDA(cudaProfilerStop());
    }
  }

  float ElapsedTime() {
    float milliseconds = 0;
    CHECK_CUDA(cudaEventSynchronize(stop_));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds;
  }

private:
  bool profile_{false};
  cudaEvent_t start_{nullptr};
  cudaEvent_t stop_{nullptr};
};

template <typename FuncType, typename... Args>
int ProfileBestConfig(const std::vector<FuncType> &funcs, cudaStream_t stream,
                      Args &&...args) {
  std::cout
      << "=================================================================="
      << std::endl;

  constexpr int kWarmupIters = 1;
  constexpr int kRepeatIters = 100;

  GpuTimer gpu_timer(false);
  float min_time_ms = 100000.f;
  int min_time_idx = -1;

  for (int idx = 0; idx < funcs.size(); ++idx) {
    auto func = funcs[idx];
    for (int i = 0; i < kWarmupIters; i++) {
      func(std::forward<Args>(args)...);
    }
    if (stream) {
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    gpu_timer.Start(stream);
    for (int i = 0; i < kRepeatIters; i++) {
      func(std::forward<Args>(args)...);
    }
    gpu_timer.Stop(stream);

    float elapsed_time_ms = gpu_timer.ElapsedTime();
    std::cout << "-- [ProfileBestConfig] No " << idx
              << ", elapsed_time: " << elapsed_time_ms << " ms" << std::endl;
    if (elapsed_time_ms < min_time_ms) {
      min_time_ms = elapsed_time_ms;
      min_time_idx = idx;
    }
  }

  std::cout << "-- [ProfileBestConfig] best config idx: " << min_time_idx
            << std::endl;
  std::cout
      << "=================================================================="
      << std::endl;
  return min_time_idx;
}

} // namespace ap
