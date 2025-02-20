#include "cutlass_matmul.cuh"
#include "default_config_id.h"
#include "epilogue_op.h"
#include "profile.h"
#include <vector>

namespace ap {

template <int TuningConfigId>
static void RunMatmulAddBinaryKernel(const GemmEpilogueParams &params) {
#if AP_USE_FLOAT16
  using ElementT = half;
  using ElementComputeT = float;
#else
  using ElementT = float;
  using ElementComputeT = float;
#endif

  typename VariadicEpilogueFunctor<ElementComputeT>::Arguments variadic_args;
  if (params.epilogue_in_ptrs.size() > 0U) {
    std::vector<int64_t> epilogue_in0_shape = params.epilogue_in_shapes[0];
    size_t begin = 3 - epilogue_in0_shape.size();
    int64_t stride = 1;
    for (int i = epilogue_in0_shape.size() - 1; i >= 0; --i) {
      variadic_args.in0_shape[begin + i] = epilogue_in0_shape[i];
      variadic_args.in0_strides[begin + i] = stride;
      // std::cout << "stride[" << begin + i << "]=" << stride << std::endl;
      stride *= epilogue_in0_shape[i];
    }
    for (size_t i = 0; i < begin; ++i) {
      variadic_args.in0_shape[i] = 1;
      variadic_args.in0_strides[i] = stride;
      // std::cout << "stride[" << i << "]=" << stride << std::endl;
    }
    variadic_args.in0_ptr =
        reinterpret_cast<const ElementT *>(params.epilogue_in_ptrs[0]);
  }

  CutlassMatmulAddVariadic<ElementT, ElementComputeT, VariadicEpilogueFunctor,
                           TuningConfigId>(params, variadic_args);
}

void MatmulAddBinaryKernel(
    cudaStream_t *stream, const void *input, const void *weight,
    const void *bias, void *output,
    const std::vector<const void *> &epilogue_ins,
    const std::vector<int64_t> &input_shape,
    const std::vector<int64_t> &weight_shape,
    const std::vector<int64_t> &bias_shape,
    const std::vector<std::vector<int64_t>> &epilogue_shapes) {
  GemmEpilogueParams params(*stream, input, weight, bias, output, input_shape,
                            weight_shape, bias_shape);
  params.SetEpilogues(epilogue_ins, epilogue_shapes);

#if AP_ENABLE_AUTO_TUNING
  static int selected_config_id = -1;

  static std::vector<std::function<void(const GemmEpilogueParams &)>>
      matmul_functions = {
          RunMatmulAddBinaryKernel<0>,  RunMatmulAddBinaryKernel<1>,
          RunMatmulAddBinaryKernel<2>,  RunMatmulAddBinaryKernel<3>,
          RunMatmulAddBinaryKernel<4>,  RunMatmulAddBinaryKernel<5>,
          RunMatmulAddBinaryKernel<6>,  RunMatmulAddBinaryKernel<7>,
          RunMatmulAddBinaryKernel<8>,  RunMatmulAddBinaryKernel<9>,
          RunMatmulAddBinaryKernel<10>, RunMatmulAddBinaryKernel<11>,
          RunMatmulAddBinaryKernel<12>, RunMatmulAddBinaryKernel<13>,
          RunMatmulAddBinaryKernel<14>, RunMatmulAddBinaryKernel<15>,
          RunMatmulAddBinaryKernel<16>, RunMatmulAddBinaryKernel<17>,
          RunMatmulAddBinaryKernel<18>, RunMatmulAddBinaryKernel<19>,
          RunMatmulAddBinaryKernel<20>, RunMatmulAddBinaryKernel<21>,
          RunMatmulAddBinaryKernel<22>, RunMatmulAddBinaryKernel<23>,
          RunMatmulAddBinaryKernel<24>, RunMatmulAddBinaryKernel<25>};
  if (selected_config_id == -1) {
    selected_config_id = ProfileBestConfig(matmul_functions, params);
  }
  matmul_functions[selected_config_id](params);
#else
  RunMatmulAddBinaryKernel<DefaultConfig::kConfigId>(params);
#endif
}

} // namespace ap
