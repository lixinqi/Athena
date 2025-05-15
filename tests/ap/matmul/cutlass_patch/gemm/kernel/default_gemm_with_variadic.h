/*! \file
  \brief
    Defines a GEMM with Reduction based on an existing UniversalGemm kernel.

*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/kernel/gemm_universal.h"

#include "cutlass_patch/epilogue/threadblock/default_epilogue_with_variadic.h"
#include "cutlass_patch/epilogue/threadblock/epilogue_with_variadic.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator      - must satisfy concept of
    /// 'EpilogueWithVariadicOp'
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    ///
    typename Enable = void>
struct DefaultGemmWithVariadic {

  using GemmBase = typename DefaultGemmUniversal<
      ElementA_, LayoutA_, TransformA, kAlignmentA, ElementB_, LayoutB_,
      TransformB, kAlignmentB, ElementC_, LayoutC_, ElementAccumulator,
      OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape,
      EpilogueOutputOp, ThreadblockSwizzle, Stages, Operator>::GemmKernel;

  // Define epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::
      DefaultEpilogueWithVariadicTensorOp<
          typename GemmBase::Epilogue::Shape,
          typename GemmBase::Epilogue::WarpMmaOperator,
          GemmBase::Epilogue::kPartitionsK, ElementC_, EpilogueOutputOp,
          GemmBase::Epilogue::kElementsPerAccess>::Epilogue;

  /// Universal kernel without StreamkFeature member type
  template <class SwizzleT, class SwizzleEnable = void>
  class SelectBase : public kernel::GemmUniversal<typename GemmBase::Mma,
                                                  Epilogue, SwizzleT> {};

  /// Universal kernel with StreamkFeature member type
  template <class SwizzleT>
  class SelectBase<SwizzleT, typename SwizzleT::StreamkFeature>
      : public kernel::GemmUniversalStreamk<typename GemmBase::Mma, Epilogue,
                                            SwizzleT> {};

  /// Select kernel by ThreadblockSwizzle's support for StreamkFeature
  using GemmKernel = SelectBase<ThreadblockSwizzle>;
};

/// Partial specialization: ArchTag = cutlass::arch::Sm70
///
///
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator      - must satisfy concept of
    /// 'EpilogueWithVariadicOp'
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    ///
    typename Enable>
struct DefaultGemmWithVariadic<
    ElementA_, LayoutA_, TransformA, kAlignmentA, ElementB_, LayoutB_,
    TransformB, kAlignmentB, ElementC_, LayoutC_, ElementAccumulator,
    OperatorClass, cutlass::arch::Sm70, ThreadblockShape, WarpShape,
    InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, Stages, Operator,
    Enable> {

  using GemmBase = typename DefaultGemmUniversal<
      ElementA_, LayoutA_, TransformA, kAlignmentA, ElementB_, LayoutB_,
      TransformB, kAlignmentB, ElementC_, LayoutC_, ElementAccumulator,
      OperatorClass, cutlass::arch::Sm70, ThreadblockShape, WarpShape,
      InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, Stages,
      Operator>::GemmKernel;

  // Define epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::
      DefaultEpilogueWithVariadicVoltaTensorOp<
          typename GemmBase::Epilogue::Shape,
          typename GemmBase::Epilogue::WarpMmaOperator,
          GemmBase::Epilogue::kPartitionsK, ElementC_, EpilogueOutputOp,
          GemmBase::Epilogue::kElementsPerAccess>::Epilogue;

  // Compose the GEMM kernel
  using GemmKernel =
      GemmUniversal<typename GemmBase::Mma, Epilogue, ThreadblockSwizzle>;
};

} // namespace kernel
} // namespace gemm
} // namespace cutlass
