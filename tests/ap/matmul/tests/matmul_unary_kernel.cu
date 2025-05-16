#include "autotune.h"
#include "cutlass_matmul.cuh"
#include "default_config_id.h"
#include "epilogue_op.h"
#include <vector>

namespace ap {

template <typename T>
// using UnaryEpilogueFunctor = ScaleFunctor<T>;
using UnaryEpilogueFunctor = IdentityFunctor<T>;

struct MatmulAddUnaryRunner {
  template <int TuningConfigId, SwizzleType ST>
  static void Apply(const GemmEpilogueParams &params) {
    using ElementT = KernelUtils::Type;
    using ElementComputeT = float;

    // typename UnaryEpilogueFunctor<ElementComputeT>::Arguments
    // unary_args{0.1};
    typename UnaryEpilogueFunctor<ElementComputeT>::Arguments unary_args;

    constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementT>::value;
    constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementT>::value;
    CutlassMatmulAddUnary<ElementT, ElementComputeT, UnaryEpilogueFunctor,
                          AlignA, AlignB, TuningConfigId, ST>(params,
                                                              unary_args);
  }
};

void MatmulAddUnaryKernel(cudaStream_t *stream, const void *input,
                          const void *weight, const void *bias, void *output,
                          const std::vector<int64_t> &input_shape,
                          const std::vector<int64_t> &weight_shape,
                          const std::vector<int64_t> &bias_shape,
                          bool transpose_b) {
  GemmEpilogueParams params(*stream, input, weight, bias, output, input_shape,
                            weight_shape, bias_shape, false, transpose_b);

  static int selected_config_id = -1;
  selected_config_id =
      RunWithAutotune<KernelUtils::Type, MatmulAddUnaryRunner, true>(
          *stream, selected_config_id, params);
}

} // namespace ap
