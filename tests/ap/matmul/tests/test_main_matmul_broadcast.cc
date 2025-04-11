#include <iostream>

#include "kernel.h"
#include "test_util.h"

template <typename T> void TestMatmulAddBroadcast(cudaStream_t stream) {
  int batch_count = 1;
  int m = 256;
  int n = 512;
  int k = 256;
  bool need_broadcast = false;

  std::cout << "we are running for problem: [" << m << ", " << n << ", " << k
            << "]" << std::endl;

  std::vector<int64_t> input_shape{batch_count, m, k};
  std::vector<int64_t> weight_shape{k, n};
  std::vector<int64_t> bias_shape{n};
  std::vector<int64_t> output_shape{batch_count, m, n};

  T *input = AllocateAndInit<T>(stream, input_shape, false, 1.);
  T *weight = AllocateAndInit<T>(stream, weight_shape, false, 1.);

  std::vector<float> bias_ref;
  bias_ref.resize(Product(bias_shape));
  for (size_t i = 0; i < bias_ref.size(); ++i) {
    bias_ref[i] = static_cast<float>(1000 * (i % 10));
  }
  T *bias = AllocateAndInit<T>(stream, bias_shape, false, 0., bias_ref);

  std::vector<int64_t> broadcast_shape = need_broadcast ? {m} : {m, n};
  std::vector<float> broadcast_ref;
  broadcast_ref.resize(Product(broadcast_shape));
  if (need_broadcast) {
    for (size_t i = 0; i < broadcast_ref.size(); ++i) {
      broadcast_ref[i] = static_cast<float>(10000 * (i % 5));
    }
  } else {
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        broadcast_ref[i * n + j] = static_cast<float>(10000 * (i % 5));
      }
    }
  }
  T *broadcast =
      AllocateAndInit<T>(stream, broadcast_shape, false, 0., broadcast_ref);

  T *output = AllocateAndInit<T>(stream, output_shape, false, 0.);
  T *broadcast_out = AllocateAndInit<T>(stream, broadcast_shape, false, 0.);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(
      cudaMemsetAsync(output, 0, sizeof(T) * Product(output_shape), stream));
  CHECK_CUDA(cudaMemsetAsync(broadcast_out, 0,
                             sizeof(T) * Product(broadcast_shape), stream));
  ap::MatmulAddBroadcastKernel(&stream, input, weight, bias, broadcast,
                               broadcast_out, output, input_shape, weight_shape,
                               bias_shape, need_broadcast);

  Print<T>(stream, reinterpret_cast<T *>(output), batch_count, m, n);

  cudaFree(input);
  cudaFree(weight);
  cudaFree(bias);
  cudaFree(output);
  cudaFree(broadcast);
  cudaFree(broadcast_out);
}

int main(int argc, const char *arg[]) {
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  TestMatmulAddBroadcast<half>(stream);

  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
