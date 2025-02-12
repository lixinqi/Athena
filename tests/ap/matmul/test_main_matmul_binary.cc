#include <iostream>

#include "profile.h"
#include "test_util.h"

#if USE_AP_GENERATED_KERNEL
#include "matmul_binary_kernel.h"
#else
#include "kernel.h"
#endif

template <typename T>
void TestMatmulAddBinary(cudaStream_t stream, int batch_count, int m, int n,
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

  std::vector<float> another_ref;
  another_ref.resize(batch_count * m * n);
  for (size_t i = 0; i < batch_count * m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      another_ref[i * n + j] = static_cast<float>(10000 * (i % 5));
    }
  }
  T *another =
      AllocateAndInit<T>(stream, batch_count * m * n, false, 0., another_ref);

  T *output = AllocateAndInit<T>(stream, batch_count * m * n, false, 0.);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(
      cudaMemsetAsync(output, 0, sizeof(T) * batch_count * m * n, stream));

#if USE_AP_GENERATED_KERNEL
  KERNEL_PROFILE(MatmulAddBinaryKernel(&stream, input, weight, output,
                                       batch_count, m, n, k));
#else
  KERNEL_PROFILE(ap::MatmulAddBinaryKernel(&stream, input, weight, bias,
                                           another, output, input_shape,
                                           weight_shape));
#endif

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

  bool add_bias = false;

#if USE_FLOAT16
  TestMatmulAddBinary<half>(stream, args.batch_count, args.m, args.n, args.k,
                            add_bias);
#else
  TestMatmulAddBinary<float>(stream, args.batch_count, args.m, args.n, args.k,
                             add_bias);
#endif

  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
