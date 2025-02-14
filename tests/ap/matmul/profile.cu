#include "matmul.h"
#include "profile.h"
#include <cuda.h>
#include <functional>

namespace ap {

int ProfileBestConfig(
    const std::vector<std::function<void(const GemmEpilogueParams &)>>
        &gemm_functions,
    const GemmEpilogueParams &params) {
  std::cout
      << "=================================================================="
      << std::endl;
  std::cout << "-- [ProfileBestConfig] Tunning for problem: {"
            << params.batch_count << ", " << params.m << ", " << params.n
            << ", " << params.k << "}" << std::endl;

  constexpr int kWarmupIters = 10;
  constexpr int kRepeatIters = 1000;

  GpuTimer gpu_timer(false);
  float min_time_ms = 100000.f;
  int min_time_idx = -1;

  for (int idx = 0; idx < gemm_functions.size(); ++idx) {
    auto func = gemm_functions[idx];
    for (int i = 0; i < kWarmupIters; i++) {
      func(params);
    }
    if (params.stream) {
      CHECK_CUDA(cudaStreamSynchronize(params.stream));
    }

    gpu_timer.Start(params.stream);
    for (int i = 0; i < kRepeatIters; i++) {
      func(params);
    }
    gpu_timer.Stop(params.stream);

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
