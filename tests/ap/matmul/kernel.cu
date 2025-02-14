#include "cutlass_matmul.cuh"
#include "epilogue_op.h"
#include "profile.h"
#include <vector>

namespace ap {

void MatmulKernel(cudaStream_t *stream, const void *input, const void *weight,
                  void *output, const std::vector<int64_t> &input_shape,
                  const std::vector<int64_t> &weight_shape, bool transpose_b) {
  GemmEpilogueParams params(*stream, input, weight, nullptr, output,
                            input_shape, weight_shape);

#if AP_USE_FLOAT16
  using ElementT = half;
  using ElementComputeT = float;
#else
  using ElementT = float;
  using ElementComputeT = float;
#endif

  if (transpose_b) {
    CutlassMatmul<ElementT, ElementComputeT, false, true>(params);
  } else {
    CutlassMatmul<ElementT, ElementComputeT, false, false>(params);
  }
}

void MatmulAddBroadcastKernel(cudaStream_t *stream, const void *input,
                              const void *weight, const void *bias,
                              void *broadcast, void *broadcast_out,
                              void *output,
                              const std::vector<int64_t> &input_shape,
                              const std::vector<int64_t> &weight_shape,
                              bool need_broadcast) {
  GemmBroadcastEpilogueParams params(*stream, input, weight, bias, broadcast,
                                     broadcast_out, output, input_shape,
                                     weight_shape, need_broadcast);

  CutlassMatmulAddBroadcast<half, float>(params);
}

} // namespace ap
