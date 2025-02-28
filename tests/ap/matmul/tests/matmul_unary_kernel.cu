#include "cutlass_matmul.cuh"
#include "default_config_id.h"
#include "epilogue_op.h"
#include "profile.h"
#include <vector>

namespace ap {

template <typename T>
// using UnaryEpilogueFunctor = ScaleFunctor<T>;
using UnaryEpilogueFunctor = IdentityFunctor<T>;

template <int TuningConfigId>
static void RunMatmulAddUnaryKernel(const GemmEpilogueParams &params) {
#if AP_USE_FLOAT16
  using ElementT = half;
  using ElementComputeT = float;
#else
  using ElementT = float;
  using ElementComputeT = float;
#endif

  // typename UnaryEpilogueFunctor<ElementComputeT>::Arguments unary_args{0.1};
  typename UnaryEpilogueFunctor<ElementComputeT>::Arguments unary_args;

  if (params.transpose_b) {
    CutlassMatmulAddUnary<ElementT, ElementComputeT, UnaryEpilogueFunctor,
                          false, true, TuningConfigId>(params, unary_args);
  } else {
    CutlassMatmulAddUnary<ElementT, ElementComputeT, UnaryEpilogueFunctor,
                          false, false, TuningConfigId>(params, unary_args);
  }
}

void MatmulAddUnaryKernel(cudaStream_t *stream, const void *input,
                          const void *weight, const void *bias, void *output,
                          const std::vector<int64_t> &input_shape,
                          const std::vector<int64_t> &weight_shape,
                          const std::vector<int64_t> &bias_shape,
                          bool transpose_b) {
  GemmEpilogueParams params(*stream, input, weight, bias, output, input_shape,
                            weight_shape, bias_shape, false, transpose_b);

#if AP_ENABLE_AUTOTUNE
#if AP_USE_FLOAT16
  AP_AUTOTUNE_half(RunMatmulAddUnaryKernel);
#else
  AP_AUTOTUNE_float(RunMatmulAddUnaryKernel);
#endif
#else
  RunMatmulAddUnaryKernel<DefaultConfig::kConfigId>(params);
#endif
}

} // namespace ap
