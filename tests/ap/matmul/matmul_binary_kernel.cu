#include "cutlass_matmul.cuh"
#include "epilogue_op.h"
#include "profile.h"
#include <vector>

namespace ap {

void MatmulAddBinaryKernel(cudaStream_t *stream, const void *input,
                           const void *weight, const void *bias,
                           const void *another, void *output,
                           const std::vector<int64_t> &input_shape,
                           const std::vector<int64_t> &weight_shape) {
  GemmEpilogueParams params(*stream, input, weight, bias, output, input_shape,
                            weight_shape);

#if AP_USE_FLOAT16
  using ElementT = half;
  using ElementComputeT = float;
#else
  using ElementT = float;
  using ElementComputeT = float;
#endif

  typename VariadicEpilogueFunctor<ElementComputeT>::Arguments variadic_args;
  variadic_args.in0_shape[0] = params.batch_count;
  variadic_args.in0_shape[1] = params.m;
  variadic_args.in0_shape[2] = params.n;
  variadic_args.in0_ptr = reinterpret_cast<const ElementT *>(another);

  CutlassMatmulAddVariadic<ElementT, ElementComputeT, VariadicEpilogueFunctor>(
      params, variadic_args);
}

} // namespace ap
