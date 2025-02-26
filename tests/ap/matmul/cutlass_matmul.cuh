#pragma once

#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/util/device_memory.h"

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"

#include "cutlass_patch/epilogue/thread/linear_combination_unary.h"
#include "cutlass_patch/epilogue/thread/linear_combination_variadic.h"
#include "cutlass_patch/gemm/device/gemm_universal_with_variadic.h"

#include "matmul.h"
#include "default_config_id.h"

namespace ap {

// Operation performed by GEMM
template <typename ElementT>
struct GemmOperation {
  using Type = cutlass::arch::OpMultiplyAdd;
};

template <>
struct GemmOperation<float> {
  using Type = cutlass::arch::OpMultiplyAddFastF32;
};

static cutlass::gemm::GemmUniversalMode GetGemmMode(int batch_count) {
  return batch_count > 1 ? cutlass::gemm::GemmUniversalMode::kBatched : cutlass::gemm::GemmUniversalMode::kGemm;
}

static void* GetWorkspace(size_t workspace_size) {
  static cutlass::device_memory::allocation<uint8_t> workspace;
  if (workspace.size() < workspace_size) {
    workspace.reset(workspace_size);
  }
  return workspace.get();
}

template <typename GemmFunc>
cutlass::Status SetMaxDynamicSharedMemorySize() {
  cudaError_t cudart_result;

  // If requires more than 48KB: configure for extended, dynamic shared memory
  if constexpr (GemmFunc::kSharedStorageSize >= (48 << 10)) {
    cudart_result = cudaFuncSetAttribute(
      cutlass::Kernel2<typename GemmFunc::GemmKernel>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      GemmFunc::kSharedStorageSize);
    if (cudart_result != cudaSuccess) {
      CUTLASS_TRACE_HOST("cudaFuncSetAttribute() returned error " << cudaGetErrorString(cudart_result));
      return cutlass::Status::kErrorInternal;
    }
  }

#if AP_ENABLE_DEBUG
  // Update SM occupancy member
  int sm_occupancy = -1;
  cudart_result = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    &sm_occupancy,
    cutlass::Kernel2<typename GemmFunc::GemmKernel>,
    GemmFunc::GemmKernel::kThreadCount,
    GemmFunc::kSharedStorageSize,
    cudaOccupancyDisableCachingOverride);
  if (cudart_result != cudaSuccess) {
    CUTLASS_TRACE_HOST("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags() returned error " << cudaGetErrorString(cudart_result));
    return cutlass::Status::kErrorInternal;
  }
  CUTLASS_TRACE_HOST("sm_occupancy: (" << sm_occupancy_ << ") "
      "smem_size: (" << GemmFunc::kSharedStorageSize << ") "
      "GemmKernel::kThreadCount: (" << GemmFunc::GemmKernel::kThreadCount << ")");
#endif
  return cutlass::Status::kSuccess;
}

template <typename ElementT,
          typename ElementComputeT,
          bool TransposeA = false,
          bool TransposeB = false,
          int ConfigId = DefaultConfig::kConfigId,
          int SwizzleFactor = DefaultConfig::kSwizzleFactor,
          bool Batched = DefaultConfig::kBatched>
void CutlassMatmul(const GemmEpilogueParams& params) {
  using ElementAccumulator = typename CutlassDataType<ElementComputeT>::Type; // <- data type of accumulator
  using ElementComputeEpilogue = ElementAccumulator;              // <- data type of epilogue operations
  using ElementInputA = typename CutlassDataType<ElementT>::Type; // <- data type of elements in input matrix A
  using ElementInputB = typename CutlassDataType<ElementT>::Type; // <- data type of elements in input matrix B
  using ElementOutput = typename CutlassDataType<ElementT>::Type; // <- data type of elements in output matrix D

  // Epilogue operation as LinearCombination:
  //  alpha * accumulator + beta * source
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;  // <- alpha x AB, no bias

  using GemmFunc = cutlass::gemm::device::GemmUniversal<
      ElementInputA,
      typename MatrixLayout<TransposeA>::Type,
      ElementInputB,
      typename MatrixLayout<TransposeB>::Type,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::TShape,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::WShape,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::IShape,
      EpilogueOutputOp,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::SwizzleThreadBlock,
      GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::kNumStages,
      128 / cutlass::sizeof_bits<ElementInputA>::value, // AlignA
      128 / cutlass::sizeof_bits<ElementInputB>::value, // AlignB
      typename GemmOperation<ElementT>::Type            // Operation performed by GEMM
  >;

  CHECK_CUTLASS(SetMaxDynamicSharedMemorySize<GemmFunc>());

  /// Arguments
  cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};

  const ElementInputA *input = reinterpret_cast<const ElementInputA *>(params.input);
  const ElementInputB *weight = reinterpret_cast<const ElementInputB *>(params.weight);
  ElementOutput *output = reinterpret_cast<ElementOutput *>(params.output);

  ElementComputeEpilogue alpha = static_cast<ElementComputeEpilogue>(1);
  ElementComputeEpilogue beta = static_cast<ElementComputeEpilogue>(0);

  typename GemmFunc::Arguments arguments{
      GetGemmMode(params.batch_count),
      problem_size,                         // <- problem size of matrix multiplication
      params.batch_count,                   // <- batch_count or k-dimension split factor
      {alpha, beta},                        // <- epilogue params, alpha, beta
      input,                                // <- input, ptr_A
      weight,                               // <- input, ptr_B
      nullptr,                              // <- input, ptr_C or bias
      output,                               // <- output, ptr_D
      params.shape_args.batch_stride_A,
      params.shape_args.batch_stride_B,
      params.shape_args.batch_stride_C,
      params.shape_args.batch_stride_D,
      params.shape_args.lda,
      params.shape_args.ldb,
      params.shape_args.ldc_bias,
      params.shape_args.ldd
  };

  size_t workspace_size = GemmFunc::get_workspace_size(arguments);
  void* workspace = workspace_size > 0 ? GetWorkspace(workspace_size) : nullptr;

  GemmFunc device_gemm;

  CHECK_CUTLASS(device_gemm.can_implement(arguments));
  CHECK_CUTLASS(device_gemm.initialize(arguments, workspace, params.stream));

  //
  // Run the GEMM
  //
  CHECK_CUTLASS(device_gemm.run(params.stream));
#if AP_ENABLE_DEBUG
  CHECK_CUDA(cudaStreamSynchronize(params.stream));
#endif
}


template <typename ElementT,
          typename ElementComputeT,
          template<typename T> class UnaryFunctor,
          bool TransposeA = false,
          bool TransposeB = false,
          int ConfigId = DefaultConfig::kConfigId,
          int SwizzleFactor = DefaultConfig::kSwizzleFactor,
          bool Batched = DefaultConfig::kBatched>
void CutlassMatmulAddUnary(const GemmEpilogueParams& params, const typename UnaryFunctor<ElementComputeT>::Arguments& unary_args) {
  using ElementAccumulator = typename CutlassDataType<ElementComputeT>::Type; // <- data type of accumulator
  using ElementComputeEpilogue = ElementAccumulator;              // <- data type of epilogue operations
  using ElementInputA = typename CutlassDataType<ElementT>::Type; // <- data type of elements in input matrix A
  using ElementInputB = typename CutlassDataType<ElementT>::Type; // <- data type of elements in input matrix B
  using ElementOutput = typename CutlassDataType<ElementT>::Type; // <- data type of elements in output matrix D

  // Epilogue operation as LinearCombinationUnary:
  //  d_ij = unary_op(alpha * sum_k(a_ik * b_kj) + c_ij)
  //
  // - sum_k(a_ik * b_kj), the intermedia result of matrix product, A * B
  // - c_ij, the bias
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationUnary<
      UnaryFunctor,
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x AB + bias

  using GemmFunc = cutlass::gemm::device::GemmUniversal<
      ElementInputA,
      typename MatrixLayout<TransposeA>::Type,
      ElementInputB,
      typename MatrixLayout<TransposeB>::Type,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::TShape,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::WShape,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::IShape,
      EpilogueOutputOp,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::SwizzleThreadBlock,
      GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::kNumStages,
      128 / cutlass::sizeof_bits<ElementInputA>::value, // AlignA
      128 / cutlass::sizeof_bits<ElementInputB>::value, // AlignB
      typename GemmOperation<ElementT>::Type  // Operation performed by GEMM
  >;

  CHECK_CUTLASS(SetMaxDynamicSharedMemorySize<GemmFunc>());

  /// Arguments
  cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};

  const ElementInputA *input = reinterpret_cast<const ElementInputA *>(params.input);
  const ElementInputB *weight = reinterpret_cast<const ElementInputB *>(params.weight);
  const ElementOutput *bias = reinterpret_cast<const ElementOutput *>(params.bias);
  ElementOutput *output = reinterpret_cast<ElementOutput *>(params.output);

  ElementComputeEpilogue alpha = static_cast<ElementComputeEpilogue>(1);
  ElementComputeEpilogue beta = bias ? static_cast<ElementComputeEpilogue>(1) : static_cast<ElementComputeEpilogue>(0);

  typename GemmFunc::Arguments arguments{
      GetGemmMode(params.batch_count),
      problem_size,                         // <- problem size of matrix multiplication
      params.batch_count,                   // <- batch_count or k-dimension split factor
      {alpha, beta, unary_args},            // <- epilogue params, alpha, beta and other arguments
      input,                                // <- input, ptr_A
      weight,                               // <- input, ptr_B
      bias,                                 // <- input, ptr_C or bias
      output,                               // <- output, ptr_D
      params.shape_args.batch_stride_A,
      params.shape_args.batch_stride_B,
      params.shape_args.batch_stride_C,
      params.shape_args.batch_stride_D,
      params.shape_args.lda,
      params.shape_args.ldb,
      params.shape_args.ldc_bias,
      params.shape_args.ldd
  };

  size_t workspace_size = GemmFunc::get_workspace_size(arguments);
  void* workspace = workspace_size > 0 ? GetWorkspace(workspace_size) : nullptr;

  GemmFunc device_gemm;

  CHECK_CUTLASS(device_gemm.can_implement(arguments));
  CHECK_CUTLASS(device_gemm.initialize(arguments, workspace, params.stream));

  //
  // Run the GEMM
  //
  CHECK_CUTLASS(device_gemm.run(params.stream));
#if AP_ENABLE_DEBUG
  CHECK_CUDA(cudaStreamSynchronize(params.stream));
#endif
}

template <typename ElementT,
          typename ElementComputeT,
          int ConfigId = DefaultConfig::kConfigId,
          int SwizzleFactor = DefaultConfig::kSwizzleFactor,
          bool Batched = DefaultConfig::kBatched>
void CutlassMatmulAddBroadcast(const GemmBroadcastEpilogueParams& params) {
  using ElementAccumulator = typename CutlassDataType<ElementComputeT>::Type; // <- data type of accumulator
  using ElementComputeEpilogue = ElementAccumulator;              // <- data type of epilogue operations
  using ElementInputA = typename CutlassDataType<ElementT>::Type; // <- data type of elements in input matrix A
  using ElementInputB = typename CutlassDataType<ElementT>::Type; // <- data type of elements in input matrix B
  using ElementOutputC = typename CutlassDataType<ElementT>::Type;// <- data type of elements in output matrix D
  using ElementOutputZ = ElementOutputC;
  using ElementOutputT = ElementOutputC;

  // Epilogue operation as LinearCombinationBiasElementwise:
  //  Y = GEMM(AB, C)
  //  T[i, j] = BinaryOp(Y[i, j], Broadcast[i])
  //  Z[i, j] = Elementwise(T[i, j])
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    ElementOutputC,
    ElementAccumulator,
    ElementComputeEpilogue,
    ElementOutputZ,
    ElementOutputT,
    128 / cutlass::sizeof_bits<ElementOutputC>::value
  >;

  // Epilogue operation as LinearCombinationResidualBlock:
  //  Y = GEMM(AB, C1)
  //  UnaryOp(BinaryOp2(BinaryOp1(ActivationOp(Y), residual1), residual2))
  // using EpilogueOp = cutlass::epilogue::thread::LinearCombinationResidualBlock<
  //   ElementOutput,                        // Element type for output matrix
  //   ElementAccumulator,                   // Element type from internal accumulation
  //   ElementCompute,                       // Element type from internal accumulation
  //   ElementC,                             // Element type for C1/C2/D matrix operands
  //   AlignmentC,                           // Memory access granularity of C and D matrix in units of elements
  //   cutlass::epilogue::thread::Identity,  // Activation
  //   cutlass::plus,                        // Binary operation 1
  //   cutlass::epilogue::thread::Identity,  // Unary operation
  //   cutlass::plus                         // Binary operation 2
  //   >;

  using GemmFunc = cutlass::gemm::device::GemmUniversalWithBroadcast<
      ElementInputA,
      cutlass::layout::RowMajor,
      ElementInputB,
      cutlass::layout::RowMajor,
      ElementOutputC,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::TShape,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::WShape,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::IShape,
      EpilogueOutputOp,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::SwizzleThreadBlock,
      GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::kNumStages,
      128 / cutlass::sizeof_bits<ElementInputA>::value, // AlignA
      128 / cutlass::sizeof_bits<ElementInputB>::value, // AlignB
      typename GemmOperation<ElementT>::Type  // Operation performed by GEMM
  >;

  CHECK_CUTLASS(SetMaxDynamicSharedMemorySize<GemmFunc>());

  /// Arguments
  cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};

  const ElementInputA *input = reinterpret_cast<const ElementInputA *>(params.input);
  const ElementInputB *weight = reinterpret_cast<const ElementInputB *>(params.weight);
  const ElementOutputC *bias = reinterpret_cast<const ElementOutputC *>(params.bias);
  ElementOutputZ *output = reinterpret_cast<ElementOutputZ *>(params.output);
  ElementOutputC *broadcast = reinterpret_cast<ElementOutputC *>(params.broadcast);
  ElementOutputT *broadcast_out = reinterpret_cast<ElementOutputT *>(params.broadcast_out);

  const int64_t batch_stride_Broadcast = params.need_broadcast ? problem_size.m() : problem_size.m() * problem_size.n();
  const int64_t ldr_broadcast = params.need_broadcast ? 0 : problem_size.n();

  ElementComputeEpilogue alpha = static_cast<ElementComputeEpilogue>(1);
  ElementComputeEpilogue beta = static_cast<ElementComputeEpilogue>(1);

  typename GemmFunc::Arguments arguments{
      GetGemmMode(params.batch_count),
      problem_size,                         // <- problem size of matrix multiplication
      params.batch_count,                   // <- batch_count or k-dimension split factor
      {alpha, beta},                        // <- epilogue params, alpha, beta
      input,                                // <- input, ptr_A, A, shape={M, K}
      weight,                               // <- input, ptr_B, B, shape={K, N}
      bias,                                 // <- input, ptr_C, shape={M, N} or {1, N}
      output,                               // <- output, ptr_D, Z, shape={M, N}
      broadcast,                            // <- input, ptr_Vector, Broadcast, shape={M, 1}
      broadcast_out,                        // <- output, ptr_Tensor, T
      params.shape_args.batch_stride_A,
      params.shape_args.batch_stride_B,
      params.shape_args.batch_stride_C,
      params.shape_args.batch_stride_D,
      batch_stride_Broadcast,               // <- batch_stride_Vector, need broadcast
      problem_size.m() * problem_size.n(),  // <- batch_stride_Tensor
      params.shape_args.lda,
      params.shape_args.ldb,
      params.shape_args.ldc_bias,
      params.shape_args.ldd,
      ldr_broadcast,                        // <- ldr, must be zero
      problem_size.n()                      // <- ldt
  };

  size_t workspace_size = GemmFunc::get_workspace_size(arguments);
  void* workspace = workspace_size > 0 ? GetWorkspace(workspace_size) : nullptr;

  GemmFunc device_gemm;

  CHECK_CUTLASS(device_gemm.can_implement(arguments));
  CHECK_CUTLASS(device_gemm.initialize(arguments, workspace, params.stream));

  //
  // Run the GEMM
  //
  CHECK_CUTLASS(device_gemm(params.stream));
#if AP_ENABLE_DEBUG
  CHECK_CUDA(cudaStreamSynchronize(params.stream));
#endif
}

template <typename ElementT,
          typename ElementComputeT,
          template<typename T> class VariadicFunctor,
          int ConfigId = DefaultConfig::kConfigId,
          int SwizzleFactor = DefaultConfig::kSwizzleFactor,
          bool Batched = DefaultConfig::kBatched>
void CutlassMatmulAddVariadic(const GemmEpilogueParams& params, const typename VariadicFunctor<ElementComputeT>::Arguments& variadic_args) {
  using ElementAccumulator = typename CutlassDataType<ElementComputeT>::Type; // <- data type of accumulator
  using ElementComputeEpilogue = ElementAccumulator;              // <- data type of epilogue operations
  using ElementInputA = typename CutlassDataType<ElementT>::Type; // <- data type of elements in input matrix A
  using ElementInputB = typename CutlassDataType<ElementT>::Type; // <- data type of elements in input matrix B
  using ElementOutput = typename CutlassDataType<ElementT>::Type;// <- data type of elements in output matrix D

  // Epilogue operation as LinearCombination:
  //  alpha * accumulator + beta * source
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationVariadic<
      VariadicFunctor,
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x AB + bias

  using GemmFunc = cutlass::gemm::device::GemmUniversalWithVariadic<
      ElementInputA,
      cutlass::layout::RowMajor,
      ElementInputB,
      cutlass::layout::RowMajor,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::TShape,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::WShape,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::IShape,
      EpilogueOutputOp,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::SwizzleThreadBlock,
      GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::kNumStages,
      128 / cutlass::sizeof_bits<ElementInputA>::value, // AlignA
      128 / cutlass::sizeof_bits<ElementInputB>::value, // AlignB
      typename GemmOperation<ElementT>::Type  // Operation performed by GEMM
  >;

  CHECK_CUTLASS(SetMaxDynamicSharedMemorySize<GemmFunc>());

  /// Arguments
  cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};

  const ElementInputA *input = reinterpret_cast<const ElementInputA *>(params.input);
  const ElementInputB *weight = reinterpret_cast<const ElementInputB *>(params.weight);
  const ElementOutput *bias = reinterpret_cast<const ElementOutput *>(params.bias);
  ElementOutput *output = reinterpret_cast<ElementOutput *>(params.output);

  ElementComputeEpilogue alpha = static_cast<ElementComputeEpilogue>(1);
  ElementComputeEpilogue beta = bias ? static_cast<ElementComputeEpilogue>(1) : static_cast<ElementComputeEpilogue>(0);

  typename GemmFunc::Arguments arguments{
      GetGemmMode(params.batch_count),
      problem_size,                         // <- problem size of matrix multiplication
      params.batch_count,                   // <- batch_count or k-dimension split factor
      {alpha, beta, variadic_args},         // <- epilogue params, alpha, beta
      input,                                // <- input, ptr_A, A, shape={M, K}
      weight,                               // <- input, ptr_B, B, shape={K, N}
      bias,                                 // <- input, ptr_C, shape={M, N} or {1, N}
      output,                               // <- output, ptr_D, Z, shape={M, N}
      params.shape_args.batch_stride_A,
      params.shape_args.batch_stride_B,
      params.shape_args.batch_stride_C,
      params.shape_args.batch_stride_D,
      params.shape_args.lda,
      params.shape_args.ldb,
      params.shape_args.ldc_bias,
      params.shape_args.ldd
  };

  size_t workspace_size = GemmFunc::get_workspace_size(arguments);
  void* workspace = workspace_size > 0 ? GetWorkspace(workspace_size) : nullptr;

  GemmFunc device_gemm;

  CHECK_CUTLASS(device_gemm.can_implement(arguments));
  CHECK_CUTLASS(device_gemm.initialize(arguments, workspace, params.stream));

  //
  // Run the GEMM
  //
  CHECK_CUTLASS(device_gemm(params.stream));
#if AP_ENABLE_DEBUG
  CHECK_CUDA(cudaStreamSynchronize(params.stream));
#endif
}

}
