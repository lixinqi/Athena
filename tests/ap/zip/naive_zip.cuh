#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>

namespace ap {

template <int topk, int num_experts, bool MP = true>
__global__ void zip_naive_kernel(
    const half *__restrict__ unzipped_tokens_in,
    const int *__restrict__ zipped_expertwise_rowmap,
    const int *__restrict__ expert_routemap_topk,
    const float *__restrict__ unzipped_token_probs,
    half *__restrict__ zipped_tokens_out ,
    float *__restrict__ zipped_probs_topk,
    const int token_length,
    const int total_zipped_tokens_num) {
  const int this_row = blockIdx.x;
  if (this_row >= total_zipped_tokens_num) return;

  const __nv_bfloat16 *unzipped_tokens =
      reinterpret_cast<const __nv_bfloat16 *>(unzipped_tokens_in);
  __nv_bfloat16 *zipped_tokens =
      reinterpret_cast<__nv_bfloat16 *>(zipped_tokens_out);

  int local_row_fetchlist[num_experts];

// -------------------------初始化任务表 ------------------------
#pragma unroll
  for (int expert = 0; expert < num_experts; ++expert) {
    const int fetch_row =
        zipped_expertwise_rowmap[this_row * num_experts + expert];
    local_row_fetchlist[expert] = fetch_row;
  }

#pragma unroll
  for (int k = 0; k < topk; ++k) {
    const int expert_idx = expert_routemap_topk[this_row * topk + k];
    if (expert_idx < 0) [[likely]]
      continue;
    const int expert_fetch_row = local_row_fetchlist[expert_idx];
    zipped_probs_topk[this_row * topk + k] =
        unzipped_token_probs[expert_fetch_row];
  }
  constexpr int vecSize = 2;  // __nv_bfloat162 = 2 x bfloat16
  const int num_full_vec = token_length / vecSize;
  const int remaining_elems = token_length % vecSize;
  const int thread_stride = blockDim.x * vecSize;

  if constexpr (MP) {
    // ------------------------ 手动混合精度 ---------------------------------
    // 齐整区域向量化搬移
    for (int x_offset = threadIdx.x * vecSize;
         x_offset < num_full_vec * vecSize;
         x_offset += thread_stride) {
      float2 sum = {0.0f, 0.0f};
      __nv_bfloat162 *out_ptr = reinterpret_cast<__nv_bfloat162 *>(
          &zipped_tokens[this_row * token_length + x_offset]);
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        const int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        // 手动类型提升
        float2 token_vec =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(
                &unzipped_tokens[fetch_row * token_length + x_offset]));
        sum.x = __fadd_rn(token_vec.x, sum.x);
        sum.y = __fadd_rn(token_vec.y, sum.y);
      }
      // 类型下降为原有精度
      *out_ptr = __float22bfloat162_rn(sum);
    }

    // 剩余元素处理
    for (int i = num_full_vec * vecSize + threadIdx.x; i < token_length;
         i += blockDim.x) {
      float sum = 0.0f;
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        float token_val =
            __bfloat162float(unzipped_tokens[fetch_row * token_length + i]);
        sum = __fadd_rn(token_val, sum);
      }
      zipped_tokens[this_row * token_length + i] = __float2bfloat16_rn(sum);
    }
  } else {
    // ------------------------ BF16 intrinsics 加权累加 -----------------------
    // 齐整区域向量化搬移
    for (int x_offset = threadIdx.x * vecSize;
         x_offset < num_full_vec * vecSize;
         x_offset += thread_stride) {
      __nv_bfloat162 sum = {0, 0};
      __nv_bfloat162 *out_ptr = reinterpret_cast<__nv_bfloat162 *>(
          &zipped_tokens[this_row * token_length + x_offset]);
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        const int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        __nv_bfloat162 token_vec = *reinterpret_cast<const __nv_bfloat162 *>(
            &unzipped_tokens[fetch_row * token_length + x_offset]);
        sum = __hadd2(sum, token_vec);
      }
      *out_ptr = sum;
    }

    // 剩余元素处理
    for (int i = num_full_vec * vecSize + threadIdx.x; i < token_length;
         i += blockDim.x) {
      __nv_bfloat16 sum = (__nv_bfloat16)0;
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        __nv_bfloat16 token_val = unzipped_tokens[fetch_row * token_length + i];
        sum = __hadd(sum, token_val);
      }
      zipped_tokens[this_row * token_length + i] = sum;
    }
  }
}

} // namespace ap