# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused RMSNorm + QKV Linear layer for attention.

This module provides a layer that combines:
1. RMSNorm
2. QKV projection

The fusion is achieved using Triton kernels from fused_norm_linear.py.
"""

from typing import Any

import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm.distributed import (
    divide,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_norm_linear import (
    fused_add_rms_norm_linear,
    fused_rms_norm_linear,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ModelWeightParameter,
    PackedColumnParameter,
)
from vllm.model_executor.utils import set_weight_attrs


class FusedRMSNormQKVParallelLinear(LinearBase):
    """
    Fused RMSNorm + QKV Linear layer for attention.

    This layer combines RMSNorm and QKV projection into a single fused operation,
    reducing memory bandwidth by avoiding intermediate tensor storage.

    Args:
        hidden_size: Input hidden state size of the transformer.
        head_size: Size of each attention head.
        total_num_heads: Total number of attention query heads.
        total_num_kv_heads: Total number of attention key/value heads.
        rms_norm_eps: Epsilon for RMSNorm.
        bias: If true, add bias to the linear projection.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configuration.
        prefix: The name of the layer in the state dict.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        rms_norm_eps: float = 1e-6,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        v_head_size: int | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        self.rms_norm_eps = rms_norm_eps

        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads

        # Tensor parallel configuration
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)

        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1

        # Calculate output sizes
        self.q_size = self.num_heads * self.head_size
        self.k_size = self.num_kv_heads * self.head_size
        self.v_size = self.num_kv_heads * self.v_head_size
        self.output_size = self.q_size + self.k_size + self.v_size

        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.v_head_size * tp_size,  # v_proj
        ]

        # Determine params dtype
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        # Initialize quantization method
        if quant_config is None:
            self.quant_method: LinearMethodBase = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

        # RMSNorm weight (not part of quantization)
        self.norm_weight = Parameter(
            torch.ones(hidden_size, dtype=params_dtype),
            requires_grad=False,
        )

        # Create QKV weights using the quant method
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=hidden_size,
            output_partition_sizes=[self.q_size, self.k_size, self.v_size],
            input_size=hidden_size,
            output_size=self.output_size * tp_size,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

        # Bias (optional)
        self.has_bias = bias
        if bias:
            self.bias = Parameter(
                torch.zeros(self.output_size, dtype=params_dtype),
                requires_grad=False,
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        """Load weights for the fused layer."""
        param_data = param.data

        # Handle norm weight
        if "norm" in param.weight_loader_attr.get("name", ""):
            param_data.copy_(loaded_weight)
            return

        # Handle QKV weights
        if loaded_shard_id is None:
            # Already fused weight
            param_data.copy_(loaded_weight)
            return

        # Get shard info
        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)

        if shard_offset is None or shard_size is None:
            raise ValueError(f"Unknown shard_id: {loaded_shard_id}")

        param_data[shard_offset : shard_offset + shard_size].copy_(loaded_weight)

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int | None:
        shard_offset_mapping = {
            "q": 0,
            "k": self.q_size,
            "v": self.q_size + self.k_size,
        }
        return shard_offset_mapping.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int | None:
        shard_size_mapping = {
            "q": self.q_size,
            "k": self.k_size,
            "v": self.v_size,
        }
        return shard_size_mapping.get(loaded_shard_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass with fused RMSNorm + QKV projection.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            residual: Optional residual tensor for fused add

        Returns:
            qkv: QKV projection output of shape [batch, seq_len, qkv_size]
            residual: Updated residual tensor (if input residual was provided)
        """
        # Check if we can use fused kernel (only for unquantized)
        if isinstance(self.quant_method, UnquantizedLinearMethod):
            return self._forward_fused(hidden_states, residual)
        else:
            # Fallback to separate ops for quantized weights
            return self._forward_unfused(hidden_states, residual)

    def _forward_fused(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fused forward using Triton kernel."""
        if residual is not None:
            qkv, residual_out = fused_add_rms_norm_linear(
                hidden_states,
                residual,
                self.norm_weight,
                self.weight,
                self.rms_norm_eps,
                self.bias if self.has_bias else None,
            )
            return qkv, residual_out
        else:
            qkv, _ = fused_rms_norm_linear(
                hidden_states,
                self.norm_weight,
                self.weight,
                self.rms_norm_eps,
                None,
                self.bias if self.has_bias else None,
            )
            return qkv, None

    def _forward_unfused(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fallback forward for quantized weights."""
        # Apply residual add + RMSNorm
        if residual is not None:
            hidden_states = hidden_states + residual
            residual = hidden_states.clone()

        # RMSNorm
        hidden_states_fp32 = hidden_states.float()
        variance = hidden_states_fp32.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.rms_norm_eps)
        hidden_states = hidden_states * self.norm_weight

        # QKV projection using quant method
        qkv = self.quant_method.apply(self, hidden_states, self.bias)

        return qkv, residual

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_size={self.head_size}, "
            f"rms_norm_eps={self.rms_norm_eps}"
        )
