#pragma once

#include "all_tuning_configs.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace ap {

enum SwizzleType { kCommon = 0, kStreamK, kBatched };

template <SwizzleType ST, int SwizzleFactor = 1> struct ThreadBlockSwizzle {
  using Type =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
};

template <int SwizzleFactor>
struct ThreadBlockSwizzle<SwizzleType::kStreamK, SwizzleFactor> {
  using Type = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
};

template <int SwizzleFactor>
struct ThreadBlockSwizzle<SwizzleType::kBatched, SwizzleFactor> {
  using Type =
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
};

struct DefaultConfig {
  static constexpr int kConfigId = 8;
  static constexpr int kSwizzleFactor = 1;
  static constexpr SwizzleType kSwizzleType = SwizzleType::kCommon;
};

} // namespace ap
