"""Benchmark: fused_split_rope vs split + rotary_embedding (original)."""

import torch
import torch.utils.benchmark as benchmark
from vllm import _custom_ops as ops

# ============================================================
# 模型配置：选用真实 LLM 的典型参数
# ============================================================
MODEL_CONFIGS = {
    "Llama-3.1-8B":  dict(num_heads_q=32, num_heads_kv=8,  head_size=128),
    "Llama-3.1-70B": dict(num_heads_q=64, num_heads_kv=8,  head_size=128),
    "Qwen2.5-7B":    dict(num_heads_q=28, num_heads_kv=4,  head_size=128),
    "Qwen2.5-72B":   dict(num_heads_q=64, num_heads_kv=8,  head_size=128),
}

# num_tokens 模拟不同场景：
#   1     - 单请求 decode
#   32    - 小批量 decode
#   256   - 中等批量
#   1024  - prefill 阶段
#   4096  - 长序列 prefill
NUM_TOKENS_LIST = [1, 32, 256, 1024, 4096]

DTYPE = torch.bfloat16
DEVICE = "cuda:0"
BASE = 500000.0
MAX_POS = 8192
WARMUP_ITERS = 50
BENCHMARK_ITERS = 200


def build_cos_sin_cache(rotary_dim: int, max_pos: int, base: float,
                        dtype: torch.dtype, device: str) -> torch.Tensor:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float,
                              device=device) / rotary_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float, device=device)
    freqs = torch.outer(t, inv_freq)
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return cache.to(dtype)


def original_split_rope(qkv: torch.Tensor, positions: torch.Tensor,
                        num_heads_q: int, num_heads_kv: int,
                        head_size: int, cos_sin_cache: torch.Tensor,
                        is_neox: bool):
    """原始方式：先 split，再分别对 Q/K 调用 rotary_embedding。"""
    q_size = num_heads_q * head_size
    kv_size = num_heads_kv * head_size
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    # rotary_embedding 是 in-place 操作
    ops.rotary_embedding(positions, q, k, head_size, cos_sin_cache, is_neox)
    return q, k, v


def fused_split_rope(qkv: torch.Tensor, positions: torch.Tensor,
                     num_heads_q: int, num_heads_kv: int,
                     head_size: int, cos_sin_cache: torch.Tensor,
                     is_neox: bool):
    """融合方式：fused_split_rope in-place 修改 QKV，然后 split。"""
    ops.fused_split_rope(qkv, positions, num_heads_q, num_heads_kv,
                         head_size, cos_sin_cache, is_neox)
    q_size = num_heads_q * head_size
    kv_size = num_heads_kv * head_size
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    return q, k, v


def main():
    torch.set_default_device(DEVICE)
    results = []

    for model_name, cfg in MODEL_CONFIGS.items():
        num_heads_q = cfg["num_heads_q"]
        num_heads_kv = cfg["num_heads_kv"]
        head_size = cfg["head_size"]
        rotary_dim = head_size  # 常见情况：全维度旋转

        cos_sin_cache = build_cos_sin_cache(
            rotary_dim, MAX_POS, BASE, DTYPE, DEVICE)

        total_dim = (num_heads_q + 2 * num_heads_kv) * head_size

        for num_tokens in NUM_TOKENS_LIST:
            positions = torch.randint(0, MAX_POS, (num_tokens,),
                                      dtype=torch.long, device=DEVICE)

            # --- Benchmark original ---
            def run_original():
                qkv = torch.randn(num_tokens, total_dim, dtype=DTYPE,
                                  device=DEVICE)
                original_split_rope(qkv, positions, num_heads_q, num_heads_kv,
                                    head_size, cos_sin_cache, True)

            t_orig = benchmark.Timer(
                stmt="fn()",
                globals={"fn": run_original},
                label="RoPE",
                sub_label=f"{model_name} tokens={num_tokens}",
                description="split+rope (original)",
            )

            # --- Benchmark fused ---
            def run_fused():
                qkv = torch.randn(num_tokens, total_dim, dtype=DTYPE,
                                  device=DEVICE)
                fused_split_rope(qkv, positions, num_heads_q, num_heads_kv,
                                 head_size, cos_sin_cache, True)

            t_fused = benchmark.Timer(
                stmt="fn()",
                globals={"fn": run_fused},
                label="RoPE",
                sub_label=f"{model_name} tokens={num_tokens}",
                description="fused_split_rope",
            )

            r_orig = t_orig.blocked_autorange(
                min_run_time=1.0, callback=None)
            r_fused = t_fused.blocked_autorange(
                min_run_time=1.0, callback=None)
            results.append(r_orig)
            results.append(r_fused)

            speedup = r_orig.median / r_fused.median
            print(f"{model_name:20s} | tokens={num_tokens:5d} | "
                  f"original={r_orig.median*1e6:8.1f}us | "
                  f"fused={r_fused.median*1e6:8.1f}us | "
                  f"speedup={speedup:.2f}x")

    # 打印汇总表格
    print("\n" + "=" * 80)
    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    main()

'''
On NVIDIA A800-SXM4-80GB:

Llama-3.1-8B         | tokens=    1 | original=    37.7us | fused=    37.6us | speedup=1.00x
Llama-3.1-8B         | tokens=   32 | original=    38.3us | fused=    37.9us | speedup=1.01x
Llama-3.1-8B         | tokens=  256 | original=    38.0us | fused=    37.8us | speedup=1.01x
Llama-3.1-8B         | tokens= 1024 | original=    52.5us | fused=    51.5us | speedup=1.02x
Llama-3.1-8B         | tokens= 4096 | original=   196.0us | fused=   193.8us | speedup=1.01x
Llama-3.1-70B        | tokens=    1 | original=    37.6us | fused=    37.5us | speedup=1.00x
Llama-3.1-70B        | tokens=   32 | original=    37.6us | fused=    37.7us | speedup=1.00x
Llama-3.1-70B        | tokens=  256 | original=    38.0us | fused=    37.7us | speedup=1.01x
Llama-3.1-70B        | tokens= 1024 | original=    78.8us | fused=    77.2us | speedup=1.02x
Llama-3.1-70B        | tokens= 4096 | original=   335.4us | fused=   333.1us | speedup=1.01x
Qwen2.5-7B           | tokens=    1 | original=    37.3us | fused=    37.4us | speedup=1.00x
Qwen2.5-7B           | tokens=   32 | original=    37.5us | fused=    37.6us | speedup=1.00x
Qwen2.5-7B           | tokens=  256 | original=    37.7us | fused=    37.7us | speedup=1.00x
Qwen2.5-7B           | tokens= 1024 | original=    42.5us | fused=    41.7us | speedup=1.02x
Qwen2.5-7B           | tokens= 4096 | original=   154.5us | fused=   152.3us | speedup=1.01x
Qwen2.5-72B          | tokens=    1 | original=    36.6us | fused=    36.9us | speedup=0.99x
Qwen2.5-72B          | tokens=   32 | original=    37.6us | fused=    37.6us | speedup=1.00x
Qwen2.5-72B          | tokens=  256 | original=    38.1us | fused=    37.4us | speedup=1.02x
Qwen2.5-72B          | tokens= 1024 | original=    78.8us | fused=    77.2us | speedup=1.02x
Qwen2.5-72B          | tokens= 4096 | original=   335.5us | fused=   333.1us | speedup=1.01x

================================================================================
[----------------------------------- RoPE -----------------------------------]
                                 |  split+rope (original)  |  fused_split_rope
1 threads: -------------------------------------------------------------------
      Llama-3.1-8B tokens=1      |           37.7          |        37.6      
      Llama-3.1-8B tokens=32     |           38.3          |        37.9      
      Llama-3.1-8B tokens=256    |           38.0          |        37.8      
      Llama-3.1-8B tokens=1024   |           52.5          |        51.5      
      Llama-3.1-8B tokens=4096   |          196.0          |       193.8      
      Llama-3.1-70B tokens=1     |           37.6          |        37.5      
      Llama-3.1-70B tokens=32    |           37.6          |        37.7      
      Llama-3.1-70B tokens=256   |           38.0          |        37.7      
      Llama-3.1-70B tokens=1024  |           78.8          |        77.2      
      Llama-3.1-70B tokens=4096  |          335.4          |       333.1      
      Qwen2.5-7B tokens=1        |           37.3          |        37.4      
      Qwen2.5-7B tokens=32       |           37.5          |        37.6      
      Qwen2.5-7B tokens=256      |           37.7          |        37.7      
      Qwen2.5-7B tokens=1024     |           42.5          |        41.7      
      Qwen2.5-7B tokens=4096     |          154.5          |       152.3      
      Qwen2.5-72B tokens=1       |           36.6          |        36.9      
      Qwen2.5-72B tokens=32      |           37.6          |        37.6      
      Qwen2.5-72B tokens=256     |           38.1          |        37.4      
      Qwen2.5-72B tokens=1024    |           78.8          |        77.2      
      Qwen2.5-72B tokens=4096    |          335.5          |       333.1      

Times are in microseconds (us).
'''