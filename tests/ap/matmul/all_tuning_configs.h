#pragma once

#include "cutlass/gemm_coord.h"

namespace ap {

template <typename ElementT, int SwizzleFactor, int Id = 0>
struct GemmTuningConfigs {
  using TShape = cutlass::gemm::GemmShape<256, 128, 32>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = Id;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 1> {
  using TShape = cutlass::gemm::GemmShape<128, 256, 32>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 1;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 2> {
  using TShape = cutlass::gemm::GemmShape<256, 64, 32>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 2;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 3> {
  using TShape = cutlass::gemm::GemmShape<256, 64, 32>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 4;
  static constexpr int kId = 3;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 4> {
  using TShape = cutlass::gemm::GemmShape<64, 256, 32>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 4;
  static constexpr int kId = 4;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 5> {
  using TShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 5;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 6> {
  using TShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 4;
  static constexpr int kId = 6;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 7> {
  using TShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 5;
  static constexpr int kId = 7;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 8> {
  using TShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 6;
  static constexpr int kId = 8;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 9> {
  using TShape = cutlass::gemm::GemmShape<64, 128, 32>;
  using WShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 6;
  static constexpr int kId = 9;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 10> {
  using TShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 10;
  static constexpr int kId = 10;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 11> {
  using TShape = cutlass::gemm::GemmShape<256, 128, 64>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 11;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 12> {
  using TShape = cutlass::gemm::GemmShape<128, 256, 64>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 12;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 13> {
  using TShape = cutlass::gemm::GemmShape<256, 64, 64>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 4;
  static constexpr int kId = 13;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 14> {
  using TShape = cutlass::gemm::GemmShape<64, 256, 64>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 4;
  static constexpr int kId = 14;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 15> {
  using TShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 4;
  static constexpr int kId = 15;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 16> {
  using TShape = cutlass::gemm::GemmShape<256, 64, 64>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 16;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 17> {
  using TShape = cutlass::gemm::GemmShape<64, 256, 64>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 17;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 18> {
  using TShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 18;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 19> {
  using TShape = cutlass::gemm::GemmShape<128, 64, 64>;
  using WShape = cutlass::gemm::GemmShape<64, 32, 64>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 19;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 20> {
  using TShape = cutlass::gemm::GemmShape<64, 128, 64>;
  using WShape = cutlass::gemm::GemmShape<32, 64, 64>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 20;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 21> {
  using TShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WShape = cutlass::gemm::GemmShape<32, 32, 64>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 5;
  static constexpr int kId = 21;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 22> {
  using TShape = cutlass::gemm::GemmShape<32, 64, 64>;
  using WShape = cutlass::gemm::GemmShape<16, 32, 64>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 5;
  static constexpr int kId = 22;
};

template <typename ElementT, int SwizzleFactor>
struct GemmTuningConfigs<ElementT, SwizzleFactor, 23> {
  using TShape = cutlass::gemm::GemmShape<16, 64, 64>;
  using WShape = cutlass::gemm::GemmShape<16, 32, 64>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 5;
  static constexpr int kId = 23;
};

// Specialization for float
template <int SwizzleFactor, int Id>
struct GemmTuningConfigs<float, SwizzleFactor, Id> {
  using TShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 4;
  static constexpr int kId = Id;
};

template <int SwizzleFactor> struct GemmTuningConfigs<float, SwizzleFactor, 1> {
  using TShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 1;
};

template <int SwizzleFactor> struct GemmTuningConfigs<float, SwizzleFactor, 2> {
  using TShape = cutlass::gemm::GemmShape<256, 64, 16>;
  using WShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 2;
};

template <int SwizzleFactor> struct GemmTuningConfigs<float, SwizzleFactor, 3> {
  using TShape = cutlass::gemm::GemmShape<256, 64, 16>;
  using WShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 3;
};

template <int SwizzleFactor> struct GemmTuningConfigs<float, SwizzleFactor, 4> {
  using TShape = cutlass::gemm::GemmShape<256, 64, 16>;
  using WShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 4;
};

template <int SwizzleFactor> struct GemmTuningConfigs<float, SwizzleFactor, 5> {
  using TShape = cutlass::gemm::GemmShape<256, 64, 16>;
  using WShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 5;
};

template <int SwizzleFactor> struct GemmTuningConfigs<float, SwizzleFactor, 6> {
  using TShape = cutlass::gemm::GemmShape<256, 64, 16>;
  using WShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 6;
};

template <int SwizzleFactor> struct GemmTuningConfigs<float, SwizzleFactor, 7> {
  using TShape = cutlass::gemm::GemmShape<256, 64, 16>;
  using WShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 7;
};

template <int SwizzleFactor> struct GemmTuningConfigs<float, SwizzleFactor, 8> {
  using TShape = cutlass::gemm::GemmShape<64, 256, 16>;
  using WShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 8;
};

template <int SwizzleFactor> struct GemmTuningConfigs<float, SwizzleFactor, 9> {
  using TShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 4;
  static constexpr int kId = 9;
};

template <int SwizzleFactor>
struct GemmTuningConfigs<float, SwizzleFactor, 10> {
  using TShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 4;
  static constexpr int kId = 10;
};

template <int SwizzleFactor>
struct GemmTuningConfigs<float, SwizzleFactor, 11> {
  using TShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 11;
};

template <int SwizzleFactor>
struct GemmTuningConfigs<float, SwizzleFactor, 12> {
  using TShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 12;
};

template <int SwizzleFactor>
struct GemmTuningConfigs<float, SwizzleFactor, 13> {
  using TShape = cutlass::gemm::GemmShape<256, 64, 32>;
  using WShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 13;
};

template <int SwizzleFactor>
struct GemmTuningConfigs<float, SwizzleFactor, 14> {
  using TShape = cutlass::gemm::GemmShape<64, 256, 32>;
  using WShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 14;
};

template <int SwizzleFactor>
struct GemmTuningConfigs<float, SwizzleFactor, 15> {
  using TShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 15;
};

template <int SwizzleFactor>
struct GemmTuningConfigs<float, SwizzleFactor, 16> {
  using TShape = cutlass::gemm::GemmShape<64, 128, 32>;
  using WShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 16;
};

template <int SwizzleFactor>
struct GemmTuningConfigs<float, SwizzleFactor, 17> {
  using TShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using IShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
  static constexpr int kNumStages = 3;
  static constexpr int kId = 17;
};

}; // namespace ap
