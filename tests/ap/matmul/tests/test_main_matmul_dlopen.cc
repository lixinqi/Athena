#include <dlfcn.h>
#include <iostream>
#include <string>

#include "test_util.h"

typedef void (*MatmulFunc)(void *, void *, void *, void *, int64_t, int64_t,
                           int64_t, int64_t, void *);
typedef void (*ApiWrapperFunc)(void *, void *, void **);

struct DlHandle {
  explicit DlHandle(const std::string &main_so_path,
                    const std::string &api_wrapper_so_path) {
    std::cout << "main_so_path: " << main_so_path << std::endl;
    std::cout << "api_wrapper_so_path: " << api_wrapper_so_path << std::endl;
    main_handle = dlopen(main_so_path.c_str(), RTLD_LAZY);
    if (!main_handle) {
      std::cerr << "Cannot open library: " << dlerror() << std::endl;
    }
    api_wrapper_handle = dlopen(api_wrapper_so_path.c_str(), RTLD_LAZY);
    if (!api_wrapper_handle) {
      std::cerr << "Cannot open library: " << dlerror() << std::endl;
    }
  }

  template <typename... Args> void Call(Args &&...args) {
    MatmulFunc func = (MatmulFunc)dlsym(main_handle, "MatmulVariadicKernel");
    const char *func_error = dlerror();
    if (func_error) {
      std::cerr << "Cannot load symbol 'MatmulVariadicKernel': " << func_error
                << std::endl;
    } else {
      func(std::forward<Args>(args)...);
    }
  }

  void CallWithApiWrapper(void **args) {
    void *main_func = dlsym(main_handle, "MatmulVariadicKernel");
    const char *main_error = dlerror();
    if (main_error) {
      std::cerr << "Cannot load symbol 'MatmulVariadicKernel': " << main_error
                << std::endl;
    }

    ApiWrapperFunc api_wrapper_func =
        (ApiWrapperFunc)dlsym(api_wrapper_handle, "MatmulVariadicKernel");
    const char *api_wrapper_error = dlerror();
    if (api_wrapper_error) {
      std::cerr << "Cannot load symbol 'MatmulVariadicKernel': "
                << api_wrapper_error << std::endl;
    }

    void *ret;
    api_wrapper_func(ret, main_func, args);
  }

  ~DlHandle() {
    dlclose(main_handle);
    dlclose(api_wrapper_handle);
  }

  void *main_handle;
  void *api_wrapper_handle;
};

template <typename T>
void TestMain(const std::string &main_so_path,
              const std::string &api_wrapper_so_path, cudaStream_t stream,
              int batch_count, int m, int n, int k, bool add_bias) {
  bool random = true;

  std::vector<int64_t> input_shape{batch_count, m, k};
  std::vector<int64_t> weight_shape{k, n};
  std::vector<int64_t> bias_shape;
  std::vector<int64_t> output_shape{batch_count, m, n};

  T *input = AllocateAndInit<T>(stream, input_shape, random, 10.);
  T *weight = AllocateAndInit<T>(stream, weight_shape, random, 10.);

  T *bias = nullptr;
  if (add_bias) {
    // bias_shape = {n};
    bias_shape = {batch_count, m, n};
    std::vector<float> bias_ref;
    bias_ref.resize(Product(bias_shape));
    for (size_t i = 0; i < bias_ref.size(); ++i) {
      bias_ref[i] = static_cast<float>(1000 * (i % 11));
    }
    bias = AllocateAndInit<T>(stream, bias_shape, random, 0., bias_ref);
  }

  T *output0 = AllocateAndInit<T>(stream, output_shape, false, 0.);
  T *output1 = AllocateAndInit<T>(stream, output_shape, false, 0.);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(
      cudaMemsetAsync(output0, 0, sizeof(T) * Product(output_shape), stream));
  CHECK_CUDA(
      cudaMemsetAsync(output1, 0, sizeof(T) * Product(output_shape), stream));

  void *stream_ptr = &stream;
  int64_t input0_dim0 = batch_count;
  int64_t input0_dim1 = m;
  int64_t input0_dim2 = k;
  int64_t input1_dim1 = n;
  T *in_ptr_0 = bias;

  DlHandle handle(main_so_path, api_wrapper_so_path);
  // KERNEL_PROFILE(handle.Call(stream_ptr, input, weight, output, input0_dim0,
  //                            input0_dim1, input0_dim2, input1_dim1,
  //                            in_ptr_0));

  ap::GpuTimer gpu_timer(true);
  for (int i = 0; i < 1010; ++i) {
    if (i == 10) {
      CHECK_CUDA(cudaStreamSynchronize(stream));
      gpu_timer.Start(stream);
    }
    std::vector<void *> args;
    args.push_back(&stream_ptr);
    args.push_back(&input);
    args.push_back(&weight);
    if (i % 2 == 0) {
      args.push_back(&output0);
    } else {
      args.push_back(&output1);
    }
    args.push_back(&input0_dim0);
    args.push_back(&input0_dim1);
    args.push_back(&input0_dim2);
    args.push_back(&input1_dim1);
    args.push_back(&in_ptr_0);
    // KERNEL_PROFILE(handle.CallWithApiWrapper(args.data()));
    handle.CallWithApiWrapper(args.data());
  }
  gpu_timer.Stop(stream);

  Print<T>(stream, reinterpret_cast<T *>(output0), batch_count, m, n);

  cudaFree(input);
  cudaFree(weight);
  if (add_bias) {
    cudaFree(bias);
  }
  cudaFree(output0);
  cudaFree(output1);
}

int main(int argc, const char *argv[]) {
  std::cout << "TestName: test_main_matmul_dlopen" << std::endl;
  ProblemSizeArgs args = ParseArgs(argc, argv);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  std::string main_so_path = "libmatmul_variadic_kernel.so";
  std::string api_wrapper_so_path = "api_wrapper.so";
  bool add_bias = false;

#if AP_USE_FLOAT16
  TestMain<half>(main_so_path, api_wrapper_so_path, stream, args.batch_count,
                 args.m, args.n, args.k, add_bias);
#else
  TestMain<float>(main_so_path, api_wrapper_so_path, stream, args.batch_count,
                  args.m, args.n, args.k, add_bias);
#endif

  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
