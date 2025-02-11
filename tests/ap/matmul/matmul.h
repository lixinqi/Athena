#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
#include <map>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"

#include "cutlass_patch/batched_matrix_coord.h"

#define CHECK_CUTLASS(status)                                                  \
  {                                                                            \
    cutlass::Status error = status;                                            \
    if (error != cutlass::Status::kSuccess) {                                  \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error)      \
                << " at: " << __LINE__ << std::endl;                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t err = func;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "[" << __FILE__ << ":" << __LINE__ << ", " << __FUNCTION__  \
                << "] "                                                        \
                << "CUDA error(" << err << "), " << cudaGetErrorString(err)    \
                << " when call " << #func << std::endl;                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define ASSERT_CHECK(__cond)                                                   \
  do {                                                                         \
    const bool __cond_var = (__cond);                                          \
    if (!__cond_var) {                                                         \
      ::std::string __err_msg = ::std::string("`") + #__cond +                 \
                                "` check failed at " + __FILE__ + ":" +        \
                                ::std::to_string(__LINE__);                    \
      throw std::runtime_error(__err_msg);                                     \
    }                                                                          \
  } while (0)

namespace ap {

using MatrixCoord = cutlass::BatchedMatrixCoord;

struct GemmEpilogueParams {
  int batch_count;
  int m;
  int n;
  int k;

  // Shape related aruguments
  struct ShapeArguments {
    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_D;
    int64_t lda;
    int64_t ldb;
    int64_t ldc_bias;
    int64_t ldd;
  };

  ShapeArguments shape_args;

  const void *input;
  const void *weight;
  const void *bias;
  void *output;

  bool is_C_bias{true};
  cudaStream_t stream;

  GemmEpilogueParams() {}
  GemmEpilogueParams(cudaStream_t stream, const void *input, const void *weight,
                     const void *bias, void *output,
                     const std::vector<int64_t> &input_shape,
                     const std::vector<int64_t> &weight_shape,
                     bool transpose_a = false, bool transpose_b = false)
      : stream(stream), input(input), weight(weight), bias(bias),
        output(output) {
    ASSERT_CHECK(input_shape.size() >= 2U);
    ASSERT_CHECK(weight_shape.size() >= 2U);

    batch_count = 1;
    for (size_t i = 0; i < input_shape.size() - 2; ++i) {
      batch_count *= input_shape[i];
    }

    if (transpose_a) {
      m = input_shape[input_shape.size() - 1];
      k = input_shape[input_shape.size() - 2];
    } else {
      m = input_shape[input_shape.size() - 2];
      k = input_shape[input_shape.size() - 1];
    }
    if (transpose_b) {
      n = weight_shape[weight_shape.size() - 2];
    } else {
      n = weight_shape[weight_shape.size() - 1];
    }

#if AP_ENABLE_DEBUG
    std::cout << "-- [GemmEpilogueParams] m: " << m << ", n: " << n
              << ", k: " << k << std::endl;
    std::cout << "-- [GemmEpilogueParams] input: " << input << std::endl;
    std::cout << "-- [GemmEpilogueParams] weight: " << weight << std::endl;
    std::cout << "-- [GemmEpilogueParams] bias: " << bias << std::endl;
    std::cout << "-- [GemmEpilogueParams] output: " << output << std::endl;
    std::cout << "-- [GemmEpilogueParams] stream: " << stream << std::endl;
#endif

    shape_args.batch_stride_A = m * k;
    shape_args.batch_stride_B = (weight_shape.size() == 2) ? 0 : n * k;
    shape_args.batch_stride_D = m * n;

    /// Only available in RRR format
    shape_args.batch_stride_C = is_C_bias ? 0 : m * n;

    shape_args.lda = transpose_a ? m : k;
    shape_args.ldb = transpose_b ? k : n;
    shape_args.ldc_bias = is_C_bias ? 0 : n;
    shape_args.ldd = n;
  }
};

struct GemmBroadcastEpilogueParams : GemmEpilogueParams {
  bool need_broadcast;
  void *broadcast;
  void *broadcast_out;

  GemmBroadcastEpilogueParams(cudaStream_t stream, const void *input,
                              const void *weight, const void *bias,
                              void *broadcast, void *broadcast_out,
                              void *output,
                              const std::vector<int64_t> &input_shape,
                              const std::vector<int64_t> &weight_shape,
                              bool need_broadcast, bool transpose_a = false,
                              bool transpose_b = false)
      : GemmEpilogueParams(stream, input, weight, bias, output, input_shape,
                           weight_shape, transpose_a, transpose_b),
        need_broadcast(need_broadcast), broadcast(broadcast),
        broadcast_out(broadcast_out) {}
};

// Convert to cutlass data type
template <typename T> struct CutlassDataType { using Type = T; };

template <> struct CutlassDataType<half> { using Type = cutlass::half_t; };

// Convert to cutlass layout
template <bool Transposed> struct MatrixLayout {
  using Type = cutlass::layout::RowMajor;
};

template <> struct MatrixLayout<true> {
  using Type = cutlass::layout::ColumnMajor;
};

} // namespace ap
