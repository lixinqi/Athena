// auto generated

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

namespace ap {

void MatmulKernel(cudaStream_t *stream, const void *input, const void *weight,
                  void *output, const std::vector<int64_t> &input_shape,
                  const std::vector<int64_t> &weight_shape, bool transpose_b);

void MatmulAddUnaryKernel(cudaStream_t *stream, const void *input,
                          const void *weight, const void *bias, void *output,
                          const std::vector<int64_t> &input_shape,
                          const std::vector<int64_t> &weight_shape,
                          const std::vector<int64_t> &bias_shape,
                          bool transpose_b);

void MatmulAddBroadcastKernel(cudaStream_t *stream, const void *input,
                              const void *weight, const void *bias,
                              void *broadcast, void *broadcast_out,
                              void *output,
                              const std::vector<int64_t> &input_shape,
                              const std::vector<int64_t> &weight_shape,
                              const std::vector<int64_t> &bias_shape,
                              bool need_broadcast);

void MatmulAddBinaryKernel(
    cudaStream_t *stream, const void *input, const void *weight,
    const void *bias, void *output,
    const std::vector<const void *> &epilogue_ins,
    const std::vector<int64_t> &input_shape,
    const std::vector<int64_t> &weight_shape,
    const std::vector<int64_t> &bias_shape,
    const std::vector<std::vector<int64_t>> &epilogue_shapes);

} // namespace ap

void NativeMatmulAddKernel(cudaStream_t *stream, const void *input,
                           const void *weight, const void *bias, void *output,
                           int batch_count, int m, int n, int k,
                           bool transpose_b);
