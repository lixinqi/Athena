#include "cutlass_matmul.cuh"
#include "epilogue_op.h"
#include <vector>

namespace ap {

void MatmulKernel(cudaStream_t *stream, const void *input, const void *weight,
                  void *output, const std::vector<int64_t> &input_shape,
                  const std::vector<int64_t> &weight_shape, bool transpose_b) {
  ap::GemmEpilogueParams params(*stream, input, weight, nullptr, output,
                                input_shape, weight_shape);

#if USE_FLOAT16
  using ElementT = half;
  using ElementComputeT = float;
#else
  using ElementT = float;
  using ElementComputeT = float;
#endif

  if (transpose_b) {
    ap::CutlassMatmul<ElementT, ElementComputeT, false, true>(params);
  } else {
    ap::CutlassMatmul<ElementT, ElementComputeT, false, false>(params);
  }
}

template <typename T>
// using UnaryEpilogueFunctor = ScaleFunctor<T>;
using UnaryEpilogueFunctor = IdentityFunctor<T>;

void MatmulAddUnaryKernel(cudaStream_t *stream, const void *input,
                          const void *weight, const void *bias, void *output,
                          const std::vector<int64_t> &input_shape,
                          const std::vector<int64_t> &weight_shape,
                          bool transpose_b) {
  ap::GemmEpilogueParams params(*stream, input, weight, bias, output,
                                input_shape, weight_shape, false, transpose_b);

#if USE_FLOAT16
  using ElementT = half;
  using ElementComputeT = float;
#else
  using ElementT = float;
  using ElementComputeT = float;
#endif

  // typename UnaryEpilogueFunctor<ElementComputeT>::Arguments unary_args{0.1};
  typename UnaryEpilogueFunctor<ElementComputeT>::Arguments unary_args;

  if (transpose_b) {
    ap::CutlassMatmulAddUnary<ElementT, ElementComputeT, UnaryEpilogueFunctor,
                              false, true>(params, unary_args);
  } else {
    ap::CutlassMatmulAddUnary<ElementT, ElementComputeT, UnaryEpilogueFunctor,
                              false, false>(params, unary_args);
  }
}

void MatmulAddBroadcastKernel(cudaStream_t *stream, const void *input,
                              const void *weight, const void *bias,
                              void *broadcast, void *broadcast_out,
                              void *output,
                              const std::vector<int64_t> &input_shape,
                              const std::vector<int64_t> &weight_shape,
                              bool need_broadcast) {
  ap::GemmBroadcastEpilogueParams params(
      *stream, input, weight, bias, broadcast, broadcast_out, output,
      input_shape, weight_shape, need_broadcast);

  ap::CutlassMatmulAddBroadcast<half, float>(params);
}

void MatmulAddBinaryKernel(cudaStream_t *stream, const void *input,
                           const void *weight, const void *bias,
                           const void *another, void *output,
                           const std::vector<int64_t> &input_shape,
                           const std::vector<int64_t> &weight_shape) {
  ap::GemmEpilogueParams params(*stream, input, weight, bias, output,
                                input_shape, weight_shape);

#if USE_FLOAT16
  using ElementT = half;
  using ElementComputeT = float;
#else
  using ElementT = float;
  using ElementComputeT = float;
#endif

  typename ap::VariadicEpilogueFunctor<ElementComputeT>::Arguments
      variadic_args;
  variadic_args.in0_shape[0] = params.batch_count;
  variadic_args.in0_shape[1] = params.m;
  variadic_args.in0_shape[2] = params.n;
  variadic_args.in0_ptr = reinterpret_cast<const ElementT *>(another);

  ap::CutlassMatmulAddVariadic<ElementT, ElementComputeT,
                               ap::VariadicEpilogueFunctor>(params,
                                                            variadic_args);
}

} // namespace ap
