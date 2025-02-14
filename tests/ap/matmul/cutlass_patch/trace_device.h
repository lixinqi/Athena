#pragma once

#if CUTLASS_DEBUG_TRACE_LEVEL

#ifndef CUTLASS_TRACE_DEVICE
#define CUTLASS_TRACE_DEVICE(format, ...)                                      \
  {                                                                            \
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&               \
        threadIdx.x == 0 && threadIdx.y == 0) {                                \
      printf("[DEVICE][%s:%d, %s]" format "\n", __FILE__, __LINE__,            \
             __FUNCTION__, ##__VA_ARGS__);                                     \
    }                                                                          \
  }
#endif

#ifndef CUTLASS_TRACE_DEVICE_TID_DETAIL
#define CUTLASS_TRACE_DEVICE_TID_DETAIL(bidz, bidx, tidx, format, ...)         \
  {                                                                            \
    if (blockIdx.x == bidx && blockIdx.y == 0 && blockIdx.z == bidz &&         \
        threadIdx.x == tidx && threadIdx.y == 0) {                             \
      printf(                                                                  \
          "[DEVICE][%s:%d, %s][bid={%d,%d,%d}, tid={%d,%d,%d}]" format "\n",   \
          __FILE__, __LINE__, __FUNCTION__, blockIdx.x, blockIdx.y,            \
          blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, ##__VA_ARGS__);   \
    }                                                                          \
  }
#endif

#ifndef CUTLASS_TRACE_DEVICE_TID
#define CUTLASS_TRACE_DEVICE_TID(format, ...)                                  \
  {                                                                            \
    CUTLASS_TRACE_DEVICE_TID_DETAIL(0, 0, 0, format, ##__VA_ARGS__)            \
    CUTLASS_TRACE_DEVICE_TID_DETAIL(0, 0, 1, format, ##__VA_ARGS__)            \
    CUTLASS_TRACE_DEVICE_TID_DETAIL(0, 1, 0, format, ##__VA_ARGS__)            \
  }
#endif

#else

#ifndef CUTLASS_TRACE_DEVICE
#define CUTLASS_TRACE_DEVICE(format, ...)
#endif

#ifndef CUTLASS_TRACE_DEVICE_TID
#define CUTLASS_TRACE_DEVICE_TID(format, ...)
#endif

#endif
