#pragma once

#include "cutlass_patch/trace_device.h"
#include "matmul.h"

namespace ap {

// Unary Epilogue
template <typename T> struct IdentityFunctor {
  struct Arguments {};

  __forceinline__ __host__ __device__ T
  operator()(T x, const Arguments &args) const {
    return x;
  }
};

template <typename T> struct ScaleFunctor {
  struct Arguments {
    float scale = static_cast<float>(1);
  };

  __forceinline__ __host__ __device__ T
  operator()(T x, const Arguments &args) const {
    return x * args.scale;
  }
};

// Variadic Epilogue
template <typename T> struct VariadicEpilogueFunctor {
  struct Arguments {
    int64_t in0_shape[3];
#if AP_USE_FLOAT16
    const half *in0_ptr{nullptr};
#else
    const T *in0_ptr{nullptr};
#endif
  };

  __forceinline__ __host__ __device__ T
  operator()(T x, const Arguments &args, const MatrixCoord &coord) const {
    int64_t offset = coord.batch * args.in0_shape[1] * args.in0_shape[2] +
                     coord.row * args.in0_shape[2] + coord.column;
    // int64_t offset = coord.column;
    T y = static_cast<T>(args.in0_ptr[offset]);
    return x + y;
  }
};

} // namespace ap
