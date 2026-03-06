# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float16]
IS_NEOX = [True, False]
SEEDS = [13]
CUDA_DEVICES = ["cuda:0"]


def _compute_cos_sin_cache(
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute cos/sin cache without instantiating RotaryEmbedding."""
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
    )
    t = torch.arange(max_position_embeddings, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache.to(dtype)


def _apply_split_rope_reference(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_size: int,
    rotary_dim: int,
    is_neox: bool,
) -> torch.Tensor:
    """Reference implementation: split then apply RoPE via static method."""
    q_size = num_heads_q * head_size
    kv_size = num_heads_kv * head_size

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q, k = RotaryEmbedding.forward_static(
        positions, q, k, head_size, rotary_dim, cos_sin_cache, is_neox
    )
    return torch.cat([q, k, v], dim=-1)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_split_rope custom op requires cuda platform",
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("rotary_ratio", [1.0, 0.5])
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim",
    [(16, 4, 128), (32, 8, 64), (8, 8, 128)],
)
@torch.inference_mode()
def test_fused_split_rope_matches_reference(
    default_vllm_config,
    device: str,
    dtype: torch.dtype,
    is_neox: bool,
    seed: int,
    rotary_ratio: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    torch.set_default_device(device)
    set_random_seed(seed)
    num_tokens = 4

    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv_base = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    qkv_fused = qkv_base.clone()
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)

    rotary_dim = int(head_dim * rotary_ratio)
    cos_sin_cache = _compute_cos_sin_cache(
        rotary_dim, 4096, 10000.0, dtype
    ).to(device)

    # Reference: split + forward_static
    ref_result = _apply_split_rope_reference(
        qkv=qkv_base,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_size=head_dim,
        rotary_dim=rotary_dim,
        is_neox=is_neox,
    )

    # Run opcheck for correctness validation
    opcheck(
        torch.ops._C.fused_split_rope,
        (
            qkv_fused.clone(),
            positions,
            num_heads,
            num_kv_heads,
            head_dim,
            cos_sin_cache,
            is_neox,
        ),
    )

    # Fused: in-place on qkv_fused
    torch.ops._C.fused_split_rope(
        qkv_fused,
        positions,
        num_heads,
        num_kv_heads,
        head_dim,
        cos_sin_cache,
        is_neox,
    )

    if dtype == torch.float16:
        ATOL, RTOL = (2e-3, 2e-3)
    else:
        ATOL, RTOL = (1e-2, 1e-2)

    torch.testing.assert_close(
        qkv_fused,
        ref_result,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_split_rope custom op requires cuda platform",
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@torch.inference_mode()
def test_fused_split_rope_v_unchanged(
    default_vllm_config,
    dtype: torch.dtype,
    is_neox: bool,
):
    """Verify that the V portion of QKV is not modified."""
    device = "cuda:0"
    torch.set_default_device(device)
    set_random_seed(42)

    num_heads, num_kv_heads, head_dim = 16, 4, 128
    num_tokens = 4
    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)

    # Save V portion before
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    v_before = qkv[:, q_size + kv_size :].clone()

    cos_sin_cache = _compute_cos_sin_cache(
        head_dim, 4096, 10000.0, dtype
    ).to(device)

    torch.ops._C.fused_split_rope(
        qkv,
        positions,
        num_heads,
        num_kv_heads,
        head_dim,
        cos_sin_cache,
        is_neox,
    )

    v_after = qkv[:, q_size + kv_size :]
    torch.testing.assert_close(v_before, v_after, atol=0, rtol=0)
