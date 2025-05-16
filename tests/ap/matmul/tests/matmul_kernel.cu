#include "autotune.h"
#include "cutlass_matmul.cuh"
#include "default_config_id.h"
#include "epilogue_op.h"
#include <vector>

namespace ap {

struct MatmulRunner {
  template <int TuningConfigId>
  static void Apply(const GemmEpilogueParams &params) {
    using ElementT = KernelUtils::Type;
    using ElementComputeT = float;

    CutlassMatmul<ElementT, ElementComputeT, false, false, TuningConfigId>(
        params);
  }
};

void MatmulKernel(cudaStream_t *stream, const void *input, const void *weight,
                  void *output, const std::vector<int64_t> &input_shape,
                  const std::vector<int64_t> &weight_shape, bool transpose_b) {
  GemmEpilogueParams params(*stream, input, weight, nullptr, output,
                            input_shape, weight_shape, std::vector<int64_t>{},
                            false, transpose_b);

  static int selected_config_id = -1;
  selected_config_id = RunWithAutotune<KernelUtils::Type, MatmulRunner>(
      *stream, selected_config_id, params);
}

} // namespace ap
