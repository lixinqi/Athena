#include "cutlass_matmul.cuh"
#include "default_config_id.h"
#include "epilogue_op.h"
#include "profile.h"
#include <vector>

namespace ap {

template <int TuningConfigId>
static void RunMatmulKernel(const GemmEpilogueParams &params) {
#if AP_USE_FLOAT16
  using ElementT = half;
  using ElementComputeT = float;
#else
  using ElementT = float;
  using ElementComputeT = float;
#endif

  if (params.transpose_b) {
    CutlassMatmul<ElementT, ElementComputeT, false, true, TuningConfigId>(
        params);
  } else {
    CutlassMatmul<ElementT, ElementComputeT, false, false, TuningConfigId>(
        params);
  }
}

void MatmulKernel(cudaStream_t *stream, const void *input, const void *weight,
                  void *output, const std::vector<int64_t> &input_shape,
                  const std::vector<int64_t> &weight_shape, bool transpose_b) {
  GemmEpilogueParams params(*stream, input, weight, nullptr, output,
                            input_shape, weight_shape, std::vector<int64_t>{},
                            false, transpose_b);

#if AP_ENABLE_AUTOTUNE
#if AP_USE_FLOAT16
  AP_AUTOTUNE_half(RunMatmulKernel, *stream, params);
#else
  AP_AUTOTUNE_float(RunMatmulKernel, *stream, params);
#endif
#else
  RunMatmulKernel<DefaultConfig::kConfigId>(params);
#endif
}

} // namespace ap
