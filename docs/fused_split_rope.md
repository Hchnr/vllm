# Fused Split RoPE 自定义算子

## 1. 背景与动机

在 Llama、Qwen2.5 等主流 LLM 的 Attention 层中，QKV 投影后需要执行两步操作：

```python
# 原始流程
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)   # Step 1: 拆分
q, k = self.rotary_emb(positions, q, k)                     # Step 2: 旋转位置编码
```

本工作将 **Split + RoPE 融合为一个 CUDA kernel**，直接在 QKV tensor 上原地完成 Q/K 的旋转编码，V 部分保持不变：

```python
# 融合流程
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = self.rotary_emb.forward_fused_split_rope(          # 一步完成
    positions, qkv, self.num_heads, self.num_kv_heads)
```

## 2. 实现方案

### 2.1 改动文件

| 文件 | 作用 |
|------|------|
| `csrc/fused_split_rope_kernel.cu` | CUDA kernel 实现 |
| `CMakeLists.txt` | 将 `.cu` 文件加入编译 |
| `csrc/ops.h` | C++ 函数声明 |
| `csrc/torch_bindings.cpp` | 注册为 PyTorch 自定义算子 `_C.fused_split_rope` |
| `vllm/_custom_ops.py` | Python 层封装 |
| `vllm/model_executor/layers/rotary_embedding/base.py` | 新增 `forward_fused_split_rope()` 方法 |
| `vllm/model_executor/models/llama.py` | Llama 模型调用新接口 |
| `tests/kernels/core/test_fused_split_rope.py` | 正确性测试 |
| `test_fused_split_rope_bench.py` | 性能 benchmark 脚本 |

### 2.2 Kernel 设计

**并行策略**：每个 CUDA block 处理一个 token，block 内线程通过 stride loop 协作处理所有 Q/K head 的旋转维度。

**QKV 内存布局**（单个 token 视角）：

```
|<--- Q: num_heads_q × head_size --->|<--- K: num_heads_kv × head_size --->|<--- V: 不处理 --->|
```

**核心计算**：对 Q 和 K 区域的每对旋转维度应用二维旋转矩阵：

```
x' = x · cos(θ) - y · sin(θ)
y' = y · cos(θ) + x · sin(θ)
```

支持 GPT-NeoX（前后半配对）和 GPT-J（相邻元素配对）两种旋转风格，通过编译期模板参数 `IS_NEOX` 消除运行时分支。

### 2.3 注册链路

```
CUDA Kernel (.cu)
  → C++ 入口函数 fused_split_rope()
    → ops.h 声明
      → torch_bindings.cpp 注册为 torch.ops._C.fused_split_rope
        → _custom_ops.py 提供 Python 接口
          → RotaryEmbedding.forward_fused_split_rope() 封装调用
            → LlamaAttention.forward() 使用
```

## 3. 性能收益分析

### 3.1 收益来源

| 来源 | 说明 |
|------|------|
| 减少 Python 开销 | 原始流程调用 `split()` + `rotary_embedding()`，融合后只需一次 `fused_split_rope()` 调用 |
| 减少显存遍历 | 原始 RoPE kernel 需分别读取 Q、K 两个 tensor 的 stride/offset；融合后直接操作连续 QKV 内存，对 GPU cache 更友好 |
| 消除中间 Tensor | `split()` 在 RoPE 之前执行时产生 3 个新 Tensor 对象；融合后 `split()` 在 RoPE 之后执行，仅为视图操作 |

### 3.2 收益局限性

RoPE 本身是一个**计算强度极低的 memory-bound 操作**。以 Llama-3.1-70B、4096 tokens 为例：

```
数据量 ≈ 4096 × (64 + 2×8) × 128 × 2 bytes ≈ 80 MB
耗时 ≈ 335 μs
等效带宽 ≈ 238 GB/s（远低于 A800 的 ~2 TB/s 峰值）
```

瓶颈在 kernel launch 和调度，而非显存带宽，因此融合 split 的收益幅度有限。

### 3.3 Kernel 级 Benchmark 数据

测试环境：NVIDIA A800-SXM4-80GB，bf16，NeoX 风格。

| 模型配置 | tokens | 原始 (μs) | 融合 (μs) | 加速比 |
|---------|--------|----------|----------|--------|
| Llama-3.1-8B (Q32/KV8/d128) | 1 | 37.7 | 37.6 | 1.00× |
| | 32 | 38.3 | 37.9 | 1.01× |
| | 256 | 38.0 | 37.8 | 1.01× |
| | 1024 | 52.5 | 51.5 | 1.02× |
| | 4096 | 196.0 | 193.8 | 1.01× |
| Llama-3.1-70B (Q64/KV8/d128) | 1 | 37.6 | 37.5 | 1.00× |
| | 32 | 37.6 | 37.7 | 1.00× |
| | 256 | 38.0 | 37.7 | 1.01× |
| | 1024 | 78.8 | 77.2 | 1.02× |
| | 4096 | 335.4 | 333.1 | 1.01× |
| Qwen2.5-7B (Q28/KV4/d128) | 1 | 37.3 | 37.4 | 1.00× |
| | 32 | 37.5 | 37.6 | 1.00× |
| | 256 | 37.7 | 37.7 | 1.00× |
| | 1024 | 42.5 | 41.7 | 1.02× |
| | 4096 | 154.5 | 152.3 | 1.01× |
| Qwen2.5-72B (Q64/KV8/d128) | 1 | 36.6 | 36.9 | 0.99× |
| | 32 | 37.6 | 37.6 | 1.00× |
| | 256 | 38.1 | 37.4 | 1.02× |
| | 1024 | 78.8 | 77.2 | 1.02× |
| | 4096 | 335.5 | 333.1 | 1.01× |

**观察**：

- 小 tokens（1~256）：耗时被 kernel launch 固定开销（~37 μs）主导，融合无明显收益。
- 大 tokens（1024~4096）：稳定获得 **1~2% 加速**，每步节省约 **1.5~2.2 μs**。
- 在端到端推理的 decode 阶段，该节省会在每个 layer 的每个 step 中累积。

## 4. 为什么不进一步融合 qkv_proj

一个自然的想法是将 `qkv_proj → split → RoPE` 三步全部融合为一个 kernel。然而这在实践中**不可行且不划算**：

### 4.1 计算特性不匹配

| | qkv_proj (GEMM) | RoPE |
|---|---|---|
| 计算类型 | compute-bound | memory-bound |
| 算术强度 | 极高（O(n²) / O(n)） | 极低（逐元素） |
| 实现方式 | cuBLAS / CUTLASS 极致优化 | 简单自定义 kernel |
| 并行策略 | Tile-based，SM 间数据复用 | 每 token 一个 block |

将两者写进一个 kernel 意味着**放弃 cuBLAS/CUTLASS 的矩阵乘优化**（包括 warp-level MMA 指令、多级 tiling、软件流水线等），得不偿失。

### 4.2 CUTLASS Epilogue 方案的局限

CUTLASS 支持自定义 epilogue（GEMM 输出写回前的逐元素操作），理论上可以在 epilogue 中对 Q/K 应用 RoPE。但存在以下困难：

- **需要额外数据**：epilogue 需访问 `positions` 数组和 `cos_sin_cache`，增加了寄存器压力和 shared memory 需求。
- **QKV 混合输出**：GEMM 的输出列中 Q、K 需要 RoPE，V 不需要，epilogue 要根据列索引做条件判断，而 CUTLASS tile 的列切分不一定和 Q/K/V 的边界对齐。
- **架构绑定**：CUTLASS epilogue 与 GPU 架构（sm80/sm89/sm90）强耦合，可移植性差。
- **收益有限**：GEMM 是 compute-bound，RoPE 读写的数据量相比 GEMM 的计算量可以忽略，省下的一次显存读写不会显著改善端到端延迟。

### 4.3 更好的融合方向

vllm 中已有的 `fused_qk_norm_rope` 提供了一个正确的融合参考——它融合的是 **RMSNorm + RoPE**，两者都是 memory-bound 的逐元素操作，特性一致，融合后减少显存读写轮次收益明显。

```
✓ memory-bound + memory-bound  →  适合融合（如 RMSNorm + RoPE）
✗ compute-bound + memory-bound →  不适合融合（如 GEMM + RoPE）
```

可以进一步探索的方向：

| 方向 | 说明 |
|------|------|
| torch.compile fusion pass | 仿照 `QKNormRoPEFusionPass`，自动匹配 `split + rotary_embedding` 并替换为 `fused_split_rope`，所有模型自动受益 |
| FlashAttention 内置 RoPE | 在 attention kernel 内部完成 RoPE，属于更大粒度的融合 |
| LayerNorm + split_rope 融合 | Attention 前的 LayerNorm 与 split_rope 都是 memory-bound，可以进一步融合 |

## 5. 使用方式

### 5.1 在模型中调用（推荐）

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = self.rotary_emb.forward_fused_split_rope(
    positions, qkv, self.num_heads, self.num_kv_heads)
attn_output = self.attn(q, k, v)
```

**前提**：`qkv` shape 为 `[num_tokens, (num_heads_q + 2 * num_heads_kv) * head_size]`。

### 5.2 直接调用底层 ops

```python
from vllm import _custom_ops as ops

# 原地修改 qkv 的 Q/K 部分，V 不变
ops.fused_split_rope(
    qkv, positions, num_heads_q, num_heads_kv,
    head_size, cos_sin_cache, is_neox=True)

q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
```

### 5.3 适用模型

任何使用 fused QKV projection 且**不带 QK Norm** 的模型均可使用，包括但不限于：

- Llama 3 / 3.1 / 3.2 全系列
- Qwen2.5（`qk_norm=False`，默认配置）
- Mistral / Mixtral

**不适用**的情况：带 QK Norm 的模型（如 BAGEL），因为 RMSNorm 必须在 split 之后、RoPE 之前对 Q/K 分别执行，打断了融合的前提。此类模型应使用 `fused_qk_norm_rope`。

## 6. 测试与 Benchmark

### 6.1 正确性测试

```bash
pytest tests/kernels/core/test_fused_split_rope.py -v
```

测试覆盖范围：
- 多种 head 配置：`(16,4,128)`, `(32,8,64)`, `(8,8,128)`
- 部分旋转（`rotary_ratio=0.5`）和全旋转（`rotary_ratio=1.0`）
- NeoX 和 GPT-J 两种旋转风格
- fp16 和 bf16 两种精度
- V 区域不被修改的验证

### 6.2 性能 Benchmark

```bash
python test_fused_split_rope_bench.py
```

Benchmark 脚本对比 `split + rotary_embedding`（原始）与 `fused_split_rope`（融合）两条路径的 kernel 执行延迟。

**模型配置**（覆盖不同 GQA 比例）：

| 模型 | Q heads | KV heads | head_size | GQA 比例 |
|------|---------|----------|-----------|---------|
| Llama-3.1-8B | 32 | 8 | 128 | 4:1 |
| Llama-3.1-70B | 64 | 8 | 128 | 8:1 |
| Qwen2.5-7B | 28 | 4 | 128 | 7:1 |
| Qwen2.5-72B | 64 | 8 | 128 | 8:1 |

**tokens 数量**（覆盖真实推理场景）：

| tokens | 场景 |
|--------|------|
| 1 | 单请求 decode |
| 32 | 小批量 decode |
| 256 | 中批量 decode |
| 1024 | prefill |
| 4096 | 长序列 prefill |

脚本使用 `torch.utils.benchmark` 的 `blocked_autorange` 方法，每组配置至少运行 1 秒以获得稳定的中位数延迟，最终输出逐项对比和汇总表格。
