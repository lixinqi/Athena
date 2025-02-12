#include <iostream>

#include "kernel.h"
#include "profile.h"
#include "test_util.h"

template <typename T>
void TestMatmul(cudaStream_t stream, int batch_count, int m, int n, int k) {
  bool transpose_b = false;

  std::vector<int64_t> input_shape{batch_count, m, k};
  std::vector<int64_t> weight_shape{k, n};

  T *input = AllocateAndInit<T>(stream, batch_count * m * k, false, 1.);
  T *weight = AllocateAndInit<T>(stream, k * n, false, 1.);
  T *output = AllocateAndInit<T>(stream, batch_count * m * n, false, 0.);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(
      cudaMemsetAsync(output, 0, sizeof(T) * batch_count * m * n, stream));
  KERNEL_PROFILE(ap::MatmulKernel(&stream, input, weight, output, input_shape,
                                  weight_shape, transpose_b));
  // KERNEL_PROFILE(NativeMatmulAddKernel(&stream, input, weight, bias, output,
  //                                      batch_count, m, n, k, transpose_b));

  Print<T>(stream, reinterpret_cast<T *>(output), batch_count, m, n);

  cudaFree(input);
  cudaFree(weight);
  cudaFree(output);
}

int main(int argc, const char *argv[]) {
  ProblemSizeArgs args = ParseArgs(argc, argv);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

#if USE_FLOAT16
  TestMatmul<half>(stream, args.batch_count, args.m, args.n, args.k);
#else
  TestMatmul<float>(stream, args.batch_count, args.m, args.n, args.k);
#endif

  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
