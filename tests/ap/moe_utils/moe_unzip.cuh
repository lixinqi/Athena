#pragma once

#include <cuda_bf16.h>
#include <stdio.h>

#define CUMSUM_BLOCK_SIZE 48   // cumsum开销和并行度之间的tradeoff的结果，勿动
#define CUMSUM_INVALID_TAG -1  // 用于标记无效的cumsum，尝试过-114514但失败了

namespace ap {

// cuda结构体拷贝
template <typename T, int N>
struct alignas(16) VectorType {
  T data[N];
};

// 128Byte对齐的结构体
template <>
struct alignas(16) VectorType<float, 4> {
  float4 data;  // Built-in CUDA vector type
};

template <>
struct alignas(16) VectorType<__nv_bfloat16, 8> {
  __nv_bfloat16 data[8];
};

// template <>
// struct alignas(16) VectorType<__nv_fp8_e4m3, 16> {
//   __nv_fp8_e4m3 data[16];
// };

template <>
struct alignas(16) VectorType<uint8_t, 16> {
  uint8_t data[16];
};


// Helper function to perform vectorized memory copy
template <typename T>
__device__ __forceinline__ void vectorized_memcpy(const T* src,
                                                  T* dst,
                                                  int num_elements) {
  constexpr int vector_size_in_bytes = 16;
  const int elements_per_vector = vector_size_in_bytes / sizeof(T);

  // 已知单行token向量化不会超过4G大小，用int节省整数开销
  int num_vectors = num_elements / elements_per_vector;
  int remaining_elements = num_elements % elements_per_vector;

  using VecType = VectorType<T, elements_per_vector>;
  const VecType* src_vec = reinterpret_cast<const VecType*>(src);
  VecType* dst_vec = reinterpret_cast<VecType*>(dst);

// 已知paddle框架中的显存分配均为256Bytes对齐，所以默认align
#pragma unroll
  for (int idx = threadIdx.x; idx < num_vectors; idx += blockDim.x) {
    dst_vec[idx] = src_vec[idx];
  }

  // 剩余无法向量化处理的元素
  if (remaining_elements > 0) {
    int offset = num_vectors * elements_per_vector;
    for (int i = threadIdx.x; i < remaining_elements; i += blockDim.x) {
      dst[offset + i] = src[offset + i];
    }
  }
}

// 多阶段算法，控制每block处理的行数来权衡额外开销
//  首先解析routemap来更新专家当前所收到的token数，然后check前一个block给的前缀和并更新给下一个block
//  随后，目的行号的信息已获取，立即开始搬运工作，直至任务完全完成
template <typename X_T,
          typename routemap_T,
          typename probs_T,
          int topk,
          int num_experts,
          bool has_scale>
__global__ void tokens_unzip_stable_kernel(
    const X_T *__restrict__ X,
    const routemap_T *__restrict__ routemap_topk,
    const probs_T *__restrict__ probs_topk,
    const float *__restrict__ XScale,
    X_T *__restrict__ X_unzipped,
    int *__restrict__ zipped_expertwise_rowmap,
    probs_T *__restrict__ probs_unzipped,
    float *__restrict__ XScale_unzipped,
    int *global_expertwise_block_cumsum,
    const int total_zipped_tokens_num,
    const int max_tokens_per_expert,
    const int token_length,
    const int scale_length) {
  if (threadIdx.x == 0 && blockIdx.x) {
    printf("run into tokens_unzip_stable_kernel");
  }
  const int block_row_base = blockIdx.x * CUMSUM_BLOCK_SIZE;
  int cumsum_offset[num_experts];
  int expert_offset[num_experts];
  int local_cumsum[num_experts];
#pragma unroll
  for (int i = 0; i < num_experts; i++) {
    cumsum_offset[i] =
        (blockIdx.x == 0)
            ? 0
            : CUMSUM_INVALID_TAG;  // 除了第0个block，其他的都以非法值初始化,因为atomic忙等要用
    expert_offset[i] = i * max_tokens_per_expert;
    local_cumsum[i] = 0;
  }
  const int base_row_idx = blockIdx.x * CUMSUM_BLOCK_SIZE;
  __shared__ int shared_expert_rowmap[CUMSUM_BLOCK_SIZE][num_experts];
  __shared__ probs_T shared_expert_probmap[CUMSUM_BLOCK_SIZE][num_experts];

  // --------------------- thread0 单线程任务传递 -------------------------
  if (threadIdx.x == 0) [[unlikely]] {
    int local_expert_rowmap[CUMSUM_BLOCK_SIZE][num_experts];
    probs_T local_expert_probs[CUMSUM_BLOCK_SIZE][num_experts];
#pragma unroll
    for (int i = 0; i < CUMSUM_BLOCK_SIZE; i++) {
#pragma unroll
      for (int j = 0; j < num_experts; j++) {
        local_expert_rowmap[i][j] =
            -1;  // 以非法值初始化，方便后续shared mem写入
        local_expert_probs[i][j] = (probs_T)0;
      }
    }
    // 将乱序访存限制在寄存器级别，后续shared_mem规整写入
    for (int row = block_row_base; row < block_row_base + CUMSUM_BLOCK_SIZE;
         row++) {
      if (row >= total_zipped_tokens_num) break;
      const int internal_row = row - block_row_base;
#pragma unroll
      for (int k = 0; k < topk; k++) {
        const int expert = routemap_topk[row * topk + k];
        if (expert == -1) continue;
        local_expert_rowmap[internal_row][expert] =
            local_cumsum[expert] + expert_offset[expert];
        local_expert_probs[internal_row][expert] = probs_topk[row * topk + k];
        local_cumsum[expert] += 1;
      }
    }
// -------------------------- 块间通信逻辑 -----------------------------
#pragma unroll
    for (int i = 0; i < num_experts; i++) {
      if (blockIdx.x != 0) [[likely]] {
        while (cumsum_offset[i] == CUMSUM_INVALID_TAG) [[likely]] {
          cumsum_offset[i] = atomicExch(
              &global_expertwise_block_cumsum[blockIdx.x * num_experts + i],
              CUMSUM_INVALID_TAG);
        }
      }
      const int proposed_offset = cumsum_offset[i] + local_cumsum[i];
      global_expertwise_block_cumsum[(blockIdx.x + 1) * num_experts + i] =
          proposed_offset;
    }  // 至此，给下一个block的cumsum已经更新完毕，下一个block可以开始cumsum的计算了

// -------------------------- 块内通信逻辑 -----------------------------
#pragma unroll
    for (int i = 0; i < CUMSUM_BLOCK_SIZE; i++) {
#pragma unroll
      for (int j = 0; j < num_experts; j++) {
        const int proposed_row =
            (local_expert_rowmap[i][j] == -1)
                ? -1
                : (local_expert_rowmap[i][j] + cumsum_offset[j]);
        shared_expert_rowmap[i][j] = proposed_row;
        shared_expert_probmap[i][j] = local_expert_probs[i][j];
      }
    }
  }  // 至此，本线程块内的shared_mem已经规整完毕，接下来是向量化的数据搬运
  __syncthreads();  // 其余线程等到了thread0，工作安排在shared_mem上
  // ------------------------- 所有block内线程 -------------------------
  for (int row = block_row_base; row < block_row_base + CUMSUM_BLOCK_SIZE;
       row++) {
    if (row >= total_zipped_tokens_num) return;
    const int internal_row = row - block_row_base;
#pragma unroll
    for (int expert = 0; expert < num_experts; expert++) {
      const int unzipped_row_idx = shared_expert_rowmap[internal_row][expert];
      if (threadIdx.x == 0) {
        zipped_expertwise_rowmap[row * num_experts + expert] = unzipped_row_idx;
      }
      if (unzipped_row_idx == -1) continue;
      // 更新三个核心数据结构
      if (threadIdx.x == 0) {
        probs_unzipped[unzipped_row_idx] =
            shared_expert_probmap[internal_row][expert];
      }
      if constexpr (has_scale) {
        vectorized_memcpy(&XScale[row * scale_length],
                          &XScale_unzipped[unzipped_row_idx * scale_length],
                          scale_length);
      }
      vectorized_memcpy(&X[row * token_length],
                        &X_unzipped[unzipped_row_idx * token_length],
                        token_length);
    }
  }
}

void tokens_unzip_stable(
    cudaStream_t stream,
    const __nv_bfloat16 *__restrict__ X,
    const float *__restrict__ XScale,
    const int *__restrict__ routemap_topk,
    const float *__restrict__ probs_topk,
    const int max_tokens_per_expert,
    const int topk,
    const int num_experts,
    __nv_bfloat16 *__restrict__ X_unzipped,
    int *__restrict__ zipped_expertwise_rowmap,
    float *__restrict__ probs_unzipped,
    float *__restrict__ XScale_unzipped,
    int *__restrict__ global_expertwise_block_cumsum,
    int rows,
    int output_rows,
    int cols) {
  cudaMemsetAsync(X_unzipped,
                  0,
                  2 * output_rows * cols,
                  stream);
  // ------------ 前缀和辅助数组相关逻辑，“推”式block通信 -------------------
  const int cumsum_blocknum =
      (rows + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
  // 设置为非法值CUMSUM_INVALID_TAG，用于线程块等待时使用
  cudaMemsetAsync(global_expertwise_block_cumsum,
                  CUMSUM_INVALID_TAG,
                  sizeof(int) * (cumsum_blocknum + 1) * num_experts,
                  stream);

  dim3 grid, block;
  grid.x =
      (rows + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
  block.x = 256;

  // topk == 8 && num_experts == 4

  auto kernel = tokens_unzip_stable_kernel<__nv_bfloat16,
                                           int,
                                           float,
                                           8,
                                           4,
                                           false>;
  const int max_tokens_per_expert_round =
      ((max_tokens_per_expert + 127) / 128) * 128;
  kernel<<<grid, block, 0, stream>>>(
      X,
      routemap_topk,
      probs_topk,
      nullptr,
      X_unzipped,
      zipped_expertwise_rowmap,
      probs_unzipped,
      nullptr,
      global_expertwise_block_cumsum,
      rows,
      max_tokens_per_expert_round,
      cols,
      0);
}

}
