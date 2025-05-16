#include "autotune.h"
#include "cutlass_matmul.cuh"
#include "default_config_id.h"
#include "epilogue_op.h"
#include <vector>

namespace ap {

struct MatmulAddBinaryRunner {
  template <int TuningConfigId, SwizzleType ST>
  static void Apply(const GemmEpilogueParams &params) {
    using ElementT = KernelUtils::Type;
    using ElementComputeT = float;

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

    constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementT>::value;
    constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementT>::value;
    CutlassMatmulAddVariadic<ElementT, ElementComputeT, VariadicEpilogueFunctor,
                             AlignA, AlignB, TuningConfigId, ST>(params,
                                                                 variadic_args);
  }
};

void MatmulAddBinaryKernel(
    cudaStream_t *stream, const void *input, const void *weight,
    const void *bias, void *output,
    const std::vector<const void *> &epilogue_ins,
    const std::vector<void *> &epilogue_outs,
    const std::vector<int64_t> &input_shape,
    const std::vector<int64_t> &weight_shape,
    const std::vector<int64_t> &bias_shape,
    const std::vector<std::vector<int64_t>> &epilogue_in_shapes,
    const std::vector<std::vector<int64_t>> &epilogue_out_shapes) {
  GemmEpilogueParams params(*stream, input, weight, bias, output, input_shape,
                            weight_shape, bias_shape);
  params.SetEpilogueAndShapes(epilogue_ins, epilogue_in_shapes, epilogue_outs,
                              epilogue_out_shapes);

  static int selected_config_id = -1;
  selected_config_id =
      RunWithAutotune<KernelUtils::Type, MatmulAddBinaryRunner, true>(
          *stream, selected_config_id, params);
}

} // namespace ap
