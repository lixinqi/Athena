#include <iostream>

#include "kernel.h"
#include "test_util.h"

template <typename T>
void TestMatmulAddBinary(cudaStream_t stream, int batch_count, int m, int n,
                         int k, bool add_bias) {
  bool transpose_b = false;

  std::vector<int64_t> input_shape{batch_count, m, k};
  std::vector<int64_t> weight_shape{k, n};
  std::vector<int64_t> bias_shape;
  std::vector<int64_t> output_shape{batch_count, m, n};

  T *input = AllocateAndInit<T>(stream, input_shape, false, 1.);
  T *weight = AllocateAndInit<T>(stream, weight_shape, false, 1.);

  T *bias = nullptr;
  if (add_bias) {
    // bias_shape = {n};
    bias_shape = {batch_count, m, n};
    std::vector<float> bias_ref;
    bias_ref.resize(Product(bias_shape));
    for (size_t i = 0; i < bias_ref.size(); ++i) {
      bias_ref[i] = static_cast<float>(1000 * (i % 11));
    }
    bias = AllocateAndInit<T>(stream, bias_shape, false, 0., bias_ref);
  }

  std::vector<float> another_ref;
  another_ref.resize(Product(input_shape));
  for (size_t i = 0; i < batch_count * m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      another_ref[i * n + j] = static_cast<float>(10000 * (i % 5));
    }
  }
  T *another = AllocateAndInit<T>(stream, output_shape, false, 0., another_ref);

  T *output = AllocateAndInit<T>(stream, output_shape, false, 0.);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(
      cudaMemsetAsync(output, 0, sizeof(T) * Product(output_shape), stream));

  KERNEL_PROFILE(ap::MatmulAddBinaryKernel(&stream, input, weight, bias,
                                           another, output, input_shape,
                                           weight_shape, bias_shape));

  Print<T>(stream, reinterpret_cast<T *>(output), batch_count, m, n);

  cudaFree(input);
  cudaFree(weight);
  if (add_bias) {
    cudaFree(bias);
  }
  cudaFree(output);
}

int main(int argc, const char *argv[]) {
  ProblemSizeArgs args = ParseArgs(argc, argv);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  bool add_bias = true;

#if AP_USE_FLOAT16
  TestMatmulAddBinary<half>(stream, args.batch_count, args.m, args.n, args.k,
                            add_bias);
#else
  TestMatmulAddBinary<float>(stream, args.batch_count, args.m, args.n, args.k,
                             add_bias);
#endif

  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
