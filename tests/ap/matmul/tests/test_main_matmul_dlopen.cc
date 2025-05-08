#include <dlfcn.h>
#include <iostream>
#include <string>

#include "test_util.h"

typedef void (*MatmulFunc)(void *, const half *, const half *, half *, int64_t,
                           int64_t, int64_t, int64_t, const half *);

struct DlHandle {
  explicit DlHandle(const std::string &so_path) {
    path = so_path;
    handle = dlopen(so_path.c_str(), RTLD_LAZY);
    if (!handle) {
      std::cerr << "Cannot open library: " << dlerror() << std::endl;
    }
  }

  template <typename... Args> void Call(Args &&...args) {
    MatmulFunc func = (MatmulFunc)dlsym(handle, "MatmulVariadicKernel");
    const char *error = dlerror();
    if (error) {
      std::cerr << "Cannot load symbol 'MatmulVariadicKernel': " << error
                << std::endl;
    } else {
      func(std::forward<Args>(args)...);
    }
  }

  ~DlHandle() { dlclose(handle); }

  std::string path;
  void *handle;
};

template <typename T>
void TestMain(const std::string &so_path, cudaStream_t stream, int batch_count,
              int m, int n, int k, bool add_bias) {
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

  T *output = AllocateAndInit<T>(stream, output_shape, false, 0.);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(
      cudaMemsetAsync(output, 0, sizeof(T) * Product(output_shape), stream));

  int64_t input0_dim0 = batch_count;
  int64_t input0_dim1 = m;
  int64_t input0_dim2 = k;
  int64_t input1_dim1 = n;
  T *in_ptr_0 = bias;

  DlHandle handle(so_path);
  KERNEL_PROFILE(handle.Call(&stream, input, weight, output, input0_dim0,
                             input0_dim1, input0_dim2, input1_dim1, in_ptr_0));

  Print<T>(stream, reinterpret_cast<T *>(output), batch_count, m, n);

  cudaFree(input);
  cudaFree(weight);
  if (add_bias) {
    cudaFree(bias);
  }
  cudaFree(output);
}

int main(int argc, const char *argv[]) {
  std::cout << "TestName: test_main_matmul_dlopen" << std::endl;
  ProblemSizeArgs args = ParseArgs(argc, argv);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  std::string so_path = "libmatmul_variadic_kernel.so";
  bool add_bias = false;

#if AP_USE_FLOAT16
  TestMain<half>(so_path, stream, args.batch_count, args.m, args.n, args.k,
                 add_bias);
#else
  TestMain<float>(so_path, stream, args.batch_count, args.m, args.n, args.k,
                  add_bias);
#endif

  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
