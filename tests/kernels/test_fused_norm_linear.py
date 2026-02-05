# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for fused RMSNorm + Linear kernel.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_norm_linear import (
    fused_add_rms_norm_linear,
    fused_rms_norm_linear,
)


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference implementation of RMSNorm."""
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * weight.float()
    return x.to(orig_dtype)


def fused_add_rms_norm_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation of fused add + RMSNorm."""
    residual = x + residual
    normed = rms_norm_ref(residual, weight, eps)
    return normed, residual


@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
@pytest.mark.parametrize("hidden_size", [256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("output_size", [256, 512, 1024, 2048])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rms_norm_linear(
    batch_size: int,
    hidden_size: int,
    output_size: int,
    dtype: torch.dtype,
):
    """Test fused RMSNorm + Linear without residual."""
    torch.manual_seed(42)
    device = "cuda"
    eps = 1e-6

    # Create inputs
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    norm_weight = torch.randn(hidden_size, dtype=dtype, device=device)
    linear_weight = torch.randn(output_size, hidden_size, dtype=dtype, device=device)

    # Reference implementation
    normed = rms_norm_ref(x, norm_weight, eps)
    expected = torch.nn.functional.linear(normed, linear_weight)

    # Fused implementation
    actual, _ = fused_rms_norm_linear(x, norm_weight, linear_weight, eps)

    # Check correctness
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
@pytest.mark.parametrize("hidden_size", [256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("output_size", [256, 512, 1024, 2048])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rms_norm_linear(
    batch_size: int,
    hidden_size: int,
    output_size: int,
    dtype: torch.dtype,
):
    """Test fused Add + RMSNorm + Linear with residual."""
    torch.manual_seed(42)
    device = "cuda"
    eps = 1e-6

    # Create inputs
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    norm_weight = torch.randn(hidden_size, dtype=dtype, device=device)
    linear_weight = torch.randn(output_size, hidden_size, dtype=dtype, device=device)

    # Reference implementation
    residual_ref = residual.clone()
    normed, residual_ref = fused_add_rms_norm_ref(x, residual_ref, norm_weight, eps)
    expected = torch.nn.functional.linear(normed, linear_weight)

    # Fused implementation
    residual_fused = residual.clone()
    actual, residual_out = fused_add_rms_norm_linear(
        x, residual_fused, norm_weight, linear_weight, eps
    )

    # Check correctness
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(residual_out, residual_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("hidden_size", [256, 512, 1024])
@pytest.mark.parametrize("output_size", [256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rms_norm_linear_with_bias(
    batch_size: int,
    hidden_size: int,
    output_size: int,
    dtype: torch.dtype,
):
    """Test fused RMSNorm + Linear with bias."""
    torch.manual_seed(42)
    device = "cuda"
    eps = 1e-6

    # Create inputs
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    norm_weight = torch.randn(hidden_size, dtype=dtype, device=device)
    linear_weight = torch.randn(output_size, hidden_size, dtype=dtype, device=device)
    bias = torch.randn(output_size, dtype=dtype, device=device)

    # Reference implementation
    normed = rms_norm_ref(x, norm_weight, eps)
    expected = torch.nn.functional.linear(normed, linear_weight, bias)

    # Fused implementation
    actual, _ = fused_rms_norm_linear(x, norm_weight, linear_weight, eps, bias=bias)

    # Check correctness
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("seq_len", [1, 64, 256, 512])
@pytest.mark.parametrize("hidden_size", [1024, 2048, 4096])
def test_fused_rms_norm_linear_3d_input(
    seq_len: int,
    hidden_size: int,
):
    """Test fused RMSNorm + Linear with 3D input (batch, seq, hidden)."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    batch_size = 2
    output_size = 1024

    # Create inputs
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    norm_weight = torch.randn(hidden_size, dtype=dtype, device=device)
    linear_weight = torch.randn(output_size, hidden_size, dtype=dtype, device=device)

    # Reference implementation
    x_2d = x.view(-1, hidden_size)
    normed = rms_norm_ref(x_2d, norm_weight, eps)
    expected = torch.nn.functional.linear(normed, linear_weight)
    expected = expected.view(batch_size, seq_len, output_size)

    # Fused implementation
    actual, _ = fused_rms_norm_linear(x, norm_weight, linear_weight, eps)

    # Check correctness
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    # Run a quick test
    print("Running quick tests...")

    # Test without residual
    test_fused_rms_norm_linear(16, 1024, 1024, torch.bfloat16)
    print("✓ test_fused_rms_norm_linear passed")

    # Test with residual
    test_fused_add_rms_norm_linear(16, 1024, 1024, torch.bfloat16)
    print("✓ test_fused_add_rms_norm_linear passed")

    # Test with bias
    test_fused_rms_norm_linear_with_bias(16, 1024, 1024, torch.bfloat16)
    print("✓ test_fused_rms_norm_linear_with_bias passed")

    # Test 3D input
    test_fused_rms_norm_linear_3d_input(64, 1024)
    print("✓ test_fused_rms_norm_linear_3d_input passed")

    print("\nAll tests passed!")
