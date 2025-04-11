#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

namespace ap {

template <typename T>
__forceinline__ __host__ __device__ T ComputePow(T base, T exponent) {
  T res = (exponent == static_cast<T>(2))
              ? (base * base)
              : ((exponent == static_cast<T>(3)) ? (base * base * base)
                                                 : (powf(base, exponent)));
  return res;
}

} // namespace ap
