#include "cutlass_matmul.cuh"
#include "epilogue_op.h"
#include <vector>

namespace ap {

template <typename T> struct UnaryEpilogueFunctor {
  using Arguments = typename ScaleFunctor<T>::Arguments;

  __forceinline__ __host__ __device__ T operator()(T x, Arguments args) const {
    return ScaleFunctor<T>()(x, args);
  }
};

template <typename T> struct VariadicEpilogueFunctor {
  struct Arguments {
    int n;
    const T *another{nullptr};
  };

  __forceinline__ __host__ __device__ T
  operator()(T x, const Arguments &args, const MatrixCoord &coord) const {
    T y = args.another[coord.row * args.n + coord.column];
    return x + y;
  }
};

void MatmulAddKernel(cudaStream_t *stream, const void *input,
                     const void *weight, const void *bias, void *output,
                     std::vector<int64_t> &input_shape,
                     std::vector<int64_t> &weight_shape, bool transpose_b) {
  ap::GemmEpilogueParams params(*stream, input, weight, bias, output,
                                input_shape, weight_shape);

#if USE_FLOAT16
  using ElementT = cutlass::half_t;
  using ElementComputeT = float;
#else
  using ElementT = float;
  using ElementComputeT = float;
#endif

  if (transpose_b) {
    ap::CutlassMatmulAdd<ElementT, ElementComputeT, false, true>(params);
  } else {
    ap::CutlassMatmulAdd<ElementT, ElementComputeT, false, false>(params);
  }
}

void MatmulAddUnaryKernel(cudaStream_t *stream, const void *input,
                          const void *weight, const void *bias, void *output,
                          const std::vector<int64_t> &input_shape,
                          const std::vector<int64_t> &weight_shape,
                          bool transpose_b) {
  ap::GemmEpilogueParams params(*stream, input, weight, bias, output,
                                input_shape, weight_shape, false, transpose_b);

#if USE_FLOAT16
  using ElementT = cutlass::half_t;
  using ElementComputeT = float;
#else
  using ElementT = float;
  using ElementComputeT = float;
#endif

  typename ap::UnaryEpilogueFunctor<ElementComputeT>::Arguments unary_args{0.1};
  if (transpose_b) {
    ap::CutlassMatmulAddUnary<ElementT, ElementComputeT,
                              ap::UnaryEpilogueFunctor, false, true>(
        params, unary_args);
  } else {
    ap::CutlassMatmulAddUnary<ElementT, ElementComputeT,
                              ap::UnaryEpilogueFunctor, false, false>(
        params, unary_args);
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

  ap::CutlassMatmulAddBroadcast<cutlass::half_t, float>(params);
}

void MatmulAddBinaryKernel(cudaStream_t *stream, const void *input,
                           const void *weight, const void *bias,
                           const void *another, void *output,
                           const std::vector<int64_t> &input_shape,
                           const std::vector<int64_t> &weight_shape) {
  ap::GemmEpilogueParams params(*stream, input, weight, bias, output,
                                input_shape, weight_shape);

#if USE_FLOAT16
  using ElementT = cutlass::half_t;
  using ElementComputeT = float;
#else
  using ElementT = float;
  using ElementComputeT = float;
#endif

  typename ap::VariadicEpilogueFunctor<ElementComputeT>::Arguments
      variadic_args;
  variadic_args.n = params.n;
  variadic_args.another = reinterpret_cast<const ElementComputeT *>(another);

  ap::CutlassMatmulAddVariadic<ElementT, ElementComputeT,
                               ap::VariadicEpilogueFunctor>(params,
                                                            variadic_args);
}

} // namespace ap
