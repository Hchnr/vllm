# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused RMSNorm + Linear (GEMM) kernel implementation using Triton.

This module provides a fused kernel that combines:
1. Optional residual addition
2. RMSNorm computation
3. Linear transformation (GEMM)

The fusion reduces memory bandwidth by avoiding intermediate tensor storage.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_add_rms_norm_rstd_kernel(
    X,  # [M, K] input tensor
    Residual,  # [M, K] residual tensor (optional, in-place updated)
    Rstd,  # [M] output reciprocal std
    stride_x_row: tl.int64,
    stride_res_row: tl.int64,
    M: tl.int64,
    K: tl.int64,
    eps: tl.float32,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    """
    Compute the reciprocal standard deviation (rstd) for RMSNorm.
    If HAS_RESIDUAL, also performs: residual = x + residual (in-place)

    Each program handles one row.
    """
    row = tl.program_id(0)

    # Compute row pointers
    X_row = X + row * stride_x_row
    if HAS_RESIDUAL:
        Res_row = Residual + row * stride_res_row

    # Accumulate sum of squares (use scalar accumulator)
    sum_sq = 0.0

    for block_start in range(0, K, BLOCK_K):
        cols = block_start + tl.arange(0, BLOCK_K)
        mask = cols < K

        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)

        if HAS_RESIDUAL:
            res = tl.load(Res_row + cols, mask=mask, other=0.0).to(tl.float32)
            x = x + res
            # Truncate to input dtype to match reference precision
            x = x.to(INPUT_DTYPE).to(tl.float32)
            # Store updated residual
            tl.store(Res_row + cols, x, mask=mask)

        sum_sq = sum_sq + tl.sum(x * x)

    # Compute rstd = 1 / sqrt(mean(x^2) + eps)
    mean_sq = sum_sq / K
    rstd = tl.rsqrt(mean_sq + eps)

    # Store rstd for this row
    tl.store(Rstd + row, rstd)


@triton.jit
def _fused_rms_norm_gemm_kernel(
    # Input tensors
    X,  # [M, K] input (or residual after fused add)
    W,  # [N, K] weight matrix (transposed for row-major)
    NormWeight,  # [K] RMSNorm weight
    Rstd,  # [M] precomputed rstd
    Out,  # [M, N] output
    Bias,  # [N] optional bias
    # Dimensions
    M: tl.int64,
    K: tl.int64,
    N: tl.int64,
    # Strides
    stride_xm: tl.int64,
    stride_xk: tl.int64,
    stride_wn: tl.int64,
    stride_wk: tl.int64,
    stride_om: tl.int64,
    stride_on: tl.int64,
    # Compile-time constants
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    """
    Fused RMSNorm + GEMM kernel.

    Computes: Out = RMSNorm(X, NormWeight, Rstd) @ W.T + Bias

    The RMSNorm is applied on-the-fly during matrix multiplication:
    x_normed = x * rstd * norm_weight
    """
    # Program ID and grid mapping (swizzled for better L2 cache utilization)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Load rstd for this block of rows
    rstd = tl.load(Rstd + offs_m, mask=offs_m < M, other=1.0)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main GEMM loop
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < K

        # Load X block: [BLOCK_M, BLOCK_K]
        x_ptrs = X + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & (k_mask[None, :]),
            other=0.0,
        ).to(tl.float32)

        # Load norm weight: [BLOCK_K]
        norm_w = tl.load(NormWeight + k_offs, mask=k_mask, other=1.0).to(tl.float32)

        # Apply RMSNorm: x_normed = x * rstd * norm_weight
        # Truncate to input dtype to match reference implementation precision
        x_normed = x * rstd[:, None] * norm_w[None, :]
        x_normed = x_normed.to(INPUT_DTYPE).to(tl.float32)

        # Load W block: [BLOCK_K, BLOCK_N] (W is stored as [N, K], so we transpose)
        w_ptrs = W + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w = tl.load(
            w_ptrs,
            mask=(k_mask[:, None]) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)

        # Accumulate: [BLOCK_M, BLOCK_N] += [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        accumulator += tl.dot(x_normed, w)

    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        accumulator += bias[None, :]

    # Convert to output dtype and store
    c = accumulator.to(Out.dtype.element_ty)
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, c, mask=out_mask)


def _get_autotune_config():
    """Get autotuning configurations for the GEMM kernel."""
    return [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
    ]


def fused_rms_norm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-6,
    residual: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Fused RMSNorm + Linear operation.

    If residual is provided:
        residual = x + residual  (in-place update)
        out = Linear(RMSNorm(residual))
    Else:
        out = Linear(RMSNorm(x))

    Args:
        x: Input tensor of shape [*, K]
        norm_weight: RMSNorm weight of shape [K]
        linear_weight: Linear weight of shape [N, K] (transposed)
        eps: Epsilon for numerical stability
        residual: Optional residual tensor of shape [*, K]
        bias: Optional bias tensor of shape [N]

    Returns:
        out: Output tensor of shape [*, N]
        residual: Updated residual tensor (or None if not provided)
    """
    # Handle arbitrary batch dimensions
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    M, K = x_2d.shape
    N = linear_weight.shape[0]

    # Ensure contiguous
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    has_residual = residual is not None
    if has_residual:
        residual_2d = residual.view(-1, residual.shape[-1])
        if not residual_2d.is_contiguous():
            residual_2d = residual_2d.contiguous()
    else:
        residual_2d = None

    # Ensure weight is contiguous
    if not linear_weight.is_contiguous():
        linear_weight = linear_weight.contiguous()
    if not norm_weight.is_contiguous():
        norm_weight = norm_weight.contiguous()

    # Allocate output tensors
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Determine block size for rstd kernel
    BLOCK_K_RSTD = min(triton.next_power_of_2(K), 8192)

    # Launch rstd computation kernel
    grid_rstd = (M,)
    input_dtype = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16
    _fused_add_rms_norm_rstd_kernel[grid_rstd](
        x_2d if not has_residual else x_2d,
        residual_2d if has_residual else x_2d,  # dummy if no residual
        rstd,
        x_2d.stride(0),
        residual_2d.stride(0) if has_residual else x_2d.stride(0),
        M,
        K,
        eps,
        HAS_RESIDUAL=has_residual,
        BLOCK_K=BLOCK_K_RSTD,
        INPUT_DTYPE=input_dtype,
        num_warps=min(max(BLOCK_K_RSTD // 256, 1), 8),
    )

    # For GEMM, use the residual (which now contains x + residual) if available
    gemm_input = residual_2d if has_residual else x_2d

    # GEMM kernel configuration
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    GROUP_SIZE_M = 8
    num_stages = 3
    num_warps = 4

    # Adjust for larger matrices
    if M >= 256 and N >= 256:
        BLOCK_M = 128
        BLOCK_N = 128
        num_warps = 8

    # Launch GEMM kernel
    grid_gemm = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _fused_rms_norm_gemm_kernel[grid_gemm](
        gemm_input,
        linear_weight,
        norm_weight,
        rstd,
        out,
        bias if bias is not None else norm_weight,  # dummy if no bias
        M,
        K,
        N,
        gemm_input.stride(0),
        gemm_input.stride(1),
        linear_weight.stride(0),
        linear_weight.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        INPUT_DTYPE=input_dtype,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    # Reshape output
    out = out.view(*orig_shape[:-1], N)

    # Return updated residual if it was provided
    if has_residual:
        residual_out = residual_2d.view(orig_shape)
        return out, residual_out
    return out, None


def fused_add_rms_norm_linear(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-6,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper for fused_rms_norm_linear with residual.

    Computes:
        residual = x + residual  (in-place)
        out = Linear(RMSNorm(residual))

    Args:
        x: Input tensor of shape [*, K]
        residual: Residual tensor of shape [*, K] (modified in-place)
        norm_weight: RMSNorm weight of shape [K]
        linear_weight: Linear weight of shape [N, K]
        eps: Epsilon for numerical stability
        bias: Optional bias tensor of shape [N]

    Returns:
        out: Output tensor of shape [*, N]
        residual: Updated residual tensor
    """
    out, residual_out = fused_rms_norm_linear(
        x, norm_weight, linear_weight, eps, residual, bias
    )
    assert residual_out is not None
    return out, residual_out
