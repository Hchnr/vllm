#!/bin/bash
# Benchmark script to compare fused vs non-fused RMSNorm+Linear kernels

MODEL="/nfs/hcr/models/Qwen/Qwen3-0.6B"
INPUT_LEN=102400
OUTPUT_LEN=102400
NUM_PROMPTS=128

echo "========================================"
echo "Model: $MODEL"
echo "Input length: $INPUT_LEN"
echo "Output length: $OUTPUT_LEN"
echo "Num prompts: $NUM_PROMPTS"
echo "========================================"

echo ""
echo "========== Baseline (no fused) =========="
VLLM_FUSED_NORM_LINEAR=0 vllm bench throughput \
    --model $MODEL \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
    --num-prompts $NUM_PROMPTS

echo ""
echo "========== Fused RMSNorm+Linear =========="
VLLM_FUSED_NORM_LINEAR=1 vllm bench throughput \
    --model $MODEL \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
    --num-prompts $NUM_PROMPTS

echo ""
echo "========== Benchmark Complete =========="
