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

#if AP_ENABLE_AUTO_TUNING
  static int selected_config_id = -1;

  std::vector<std::function<void(const GemmEpilogueParams &)>>
      matmul_functions = {
          RunMatmulKernel<0>,  RunMatmulKernel<1>,  RunMatmulKernel<2>,
          RunMatmulKernel<3>,  RunMatmulKernel<4>,  RunMatmulKernel<5>,
          RunMatmulKernel<6>,  RunMatmulKernel<7>,  RunMatmulKernel<8>,
          RunMatmulKernel<9>,  RunMatmulKernel<10>, RunMatmulKernel<11>,
          RunMatmulKernel<12>, RunMatmulKernel<13>, RunMatmulKernel<14>,
          RunMatmulKernel<15>, RunMatmulKernel<16>, RunMatmulKernel<17>,
          RunMatmulKernel<18>, RunMatmulKernel<19>, RunMatmulKernel<20>,
          RunMatmulKernel<21>, RunMatmulKernel<22>, RunMatmulKernel<23>,
          RunMatmulKernel<24>, RunMatmulKernel<25>};
  if (selected_config_id == -1) {
    selected_config_id = ProfileBestConfig(matmul_functions, params);
  }
  matmul_functions[selected_config_id](params);
#else
  RunMatmulKernel<DefaultConfig::kConfigId>(params);
#endif
}

} // namespace ap
