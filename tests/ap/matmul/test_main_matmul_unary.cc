#include <iostream>

#include "test_util.h"
#include "kernel.h"

template <typename T>
void TestMatmulAddUnary(cudaStream_t stream, int batch_count, int m, int n,
                        int k, bool add_bias) {
  bool transpose_b = false;

  std::vector<int64_t> input_shape{batch_count, m, k};
  std::vector<int64_t> weight_shape{k, n};

  T *input = AllocateAndInit<T>(stream, batch_count * m * k, false, 1.);
  T *weight = AllocateAndInit<T>(stream, k * n, false, 1.);

  T *bias = nullptr;
  if (add_bias) {
    std::vector<float> bias_ref;
    bias_ref.resize(n);
    for (size_t i = 0; i < bias_ref.size(); ++i) {
      bias_ref[i] = static_cast<float>(1000 * (i % 11));
    }
    bias = AllocateAndInit<T>(stream, n, false, 0., bias_ref);
  }

  T *output = AllocateAndInit<T>(stream, batch_count * m * n, false, 0.);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(
      cudaMemsetAsync(output, 0, sizeof(T) * batch_count * m * n, stream));

  KERNEL_PROFILE(ap::MatmulAddUnaryKernel(&stream, input, weight, bias, output,
                                          input_shape, weight_shape,
                                          transpose_b));

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
  TestMatmulAddUnary<half>(stream, args.batch_count, args.m, args.n, args.k,
                           add_bias);
#else
  TestMatmulAddUnary<float>(stream, args.batch_count, args.m, args.n, args.k,
                            add_bias);
#endif

  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
