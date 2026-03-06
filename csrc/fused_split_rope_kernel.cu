// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding_split_rope(
    scalar_t* __restrict__ arr, const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr, int rot_offset, int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if constexpr (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = VLLM_LDG(cos_ptr + x_index);
    sin = VLLM_LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = VLLM_LDG(cos_ptr + x_index / 2);
    sin = VLLM_LDG(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
__global__ void fused_split_rope_kernel(
    const int64_t* __restrict__ positions,  // [num_tokens]
    scalar_t* __restrict__ qkv,  // [num_tokens, (num_heads_q + 2 *
                                 // num_heads_kv) * head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, rot_dim]
    const int rot_dim, const int num_heads_q, const int num_heads_kv,
    const int head_size, const int total_hidden_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  // Pointer to the start of this token's QKV data
  scalar_t* token_qkv = qkv + token_idx * total_hidden_size;

  // Apply RoPE to Q heads
  const int q_offset = 0;
  const int nq = num_heads_q * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i % embed_dim;
    scalar_t* head_ptr = token_qkv + q_offset + head_idx * head_size;
    apply_token_rotary_embedding_split_rope<scalar_t, IS_NEOX>(
        head_ptr, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  // Apply RoPE to K heads
  const int k_offset = num_heads_q * head_size;
  const int nk = num_heads_kv * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i % embed_dim;
    scalar_t* head_ptr = token_qkv + k_offset + head_idx * head_size;
    apply_token_rotary_embedding_split_rope<scalar_t, IS_NEOX>(
        head_ptr, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }
}

}  // namespace vllm

void fused_split_rope(
    torch::Tensor& qkv,        // [num_tokens, (num_heads_q + 2 *
                                // num_heads_kv) * head_size]
    torch::Tensor& positions,   // [num_tokens]
    int64_t num_heads_q,        // Number of query heads
    int64_t num_heads_kv,       // Number of key/value heads
    int64_t head_size,          // Head dimension
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox) {
  int64_t num_tokens = positions.numel();

  // Validate inputs
  TORCH_CHECK(positions.dim() == 1,
              "positions must be 1D: [num_tokens]");
  TORCH_CHECK(qkv.dim() == 2,
              "qkv must be 2D: [num_tokens, hidden_size]");
  TORCH_CHECK(qkv.size(0) == num_tokens,
              "qkv and positions must have the same number of tokens");

  int64_t total_hidden_size = (num_heads_q + 2 * num_heads_kv) * head_size;
  TORCH_CHECK(qkv.size(1) == total_hidden_size,
              "qkv hidden size must equal "
              "(num_heads_q + 2 * num_heads_kv) * head_size, got ",
              qkv.size(1), " expected ", total_hidden_size);

  int rot_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rot_dim <= head_size,
              "rot_dim must be <= head_size");

  int64_t max_heads = std::max(num_heads_q, num_heads_kv);

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(max_heads * rot_dim / 2, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      qkv.scalar_type(), "fused_split_rope", [&] {
        if (is_neox) {
          vllm::fused_split_rope_kernel<scalar_t, true>
              <<<grid, block, 0, stream>>>(
                  positions.data_ptr<int64_t>(), qkv.data_ptr<scalar_t>(),
                  cos_sin_cache.data_ptr<scalar_t>(), rot_dim,
                  static_cast<int>(num_heads_q),
                  static_cast<int>(num_heads_kv),
                  static_cast<int>(head_size),
                  static_cast<int>(total_hidden_size));
        } else {
          vllm::fused_split_rope_kernel<scalar_t, false>
              <<<grid, block, 0, stream>>>(
                  positions.data_ptr<int64_t>(), qkv.data_ptr<scalar_t>(),
                  cos_sin_cache.data_ptr<scalar_t>(), rot_dim,
                  static_cast<int>(num_heads_q),
                  static_cast<int>(num_heads_kv),
                  static_cast<int>(head_size),
                  static_cast<int>(total_hidden_size));
        }
      });
}
