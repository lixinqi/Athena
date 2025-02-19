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

#if AP_ENABLE_AUTO_TUNING
  static int selected_config_id = -1;

  std::vector<std::function<void(const GemmEpilogueParams &)>>
      matmul_functions = {
          RunMatmulAddUnaryKernel<0>,  RunMatmulAddUnaryKernel<1>,
          RunMatmulAddUnaryKernel<2>,  RunMatmulAddUnaryKernel<3>,
          RunMatmulAddUnaryKernel<4>,  RunMatmulAddUnaryKernel<5>,
          RunMatmulAddUnaryKernel<6>,  RunMatmulAddUnaryKernel<7>,
          RunMatmulAddUnaryKernel<8>,  RunMatmulAddUnaryKernel<9>,
          RunMatmulAddUnaryKernel<10>, RunMatmulAddUnaryKernel<11>,
          RunMatmulAddUnaryKernel<12>, RunMatmulAddUnaryKernel<13>,
          RunMatmulAddUnaryKernel<14>, RunMatmulAddUnaryKernel<15>,
          RunMatmulAddUnaryKernel<16>, RunMatmulAddUnaryKernel<17>,
          RunMatmulAddUnaryKernel<18>, RunMatmulAddUnaryKernel<19>,
          RunMatmulAddUnaryKernel<20>, RunMatmulAddUnaryKernel<21>,
          RunMatmulAddUnaryKernel<22>, RunMatmulAddUnaryKernel<23>,
          RunMatmulAddUnaryKernel<24>, RunMatmulAddUnaryKernel<25>};
  if (selected_config_id == -1) {
    selected_config_id = ProfileBestConfig(matmul_functions, params);
  }
  matmul_functions[selected_config_id](params);
#else
  RunMatmulAddUnaryKernel<DefaultConfig::kConfigId>(params);
#endif
}

} // namespace ap
