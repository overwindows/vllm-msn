#!/bin/bash
# Gemma 4 26B MoE with FP8 Quantization + MTP Assistant on A100 40GB
# Multi-Token Prediction for 2-3x speedup

VLLM_CMD=llm_analyzer_gemma4_moe_fp8_mtp.py

# Set logging levels
export PYTHONWARNINGS="ignore"
export LOGURU_LEVEL="ERROR"
export VLLM_LOG_LEVEL="ERROR"
export PYTHONUNBUFFERED=1

# A100 40GB Configuration
export TORCH_CUDA_ARCH_LIST="8.0"

# Attention Backend: FLASH_ATTN is optimal for MoE models on A100
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# FP8 Quantization for MoE
export VLLM_USE_FLASHINFER_MOE_FP8=1
export VLLM_MOE_BACKEND=auto

# Model Configuration
MODEL_PATH=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it
ASSISTANT_PATH=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant

# Speculative Decoding with MTP
# The assistant model predicts 4-6 tokens ahead for faster inference
NUM_SPECULATIVE_TOKENS=5  # Typical range: 3-7

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/nvmedata/chenw/vllm-ra python3 $VLLM_CMD \
    --input_path /nvmedata/chenw/genz/genz_users_20k_format.tsv \
    --output_path /nvmedata/chenw/genz/genz_users_interests_gemma4_fp8_mtp.jsonl \
    --model_path $MODEL_PATH \
    --speculative_model $ASSISTANT_PATH \
    --num_speculative_tokens $NUM_SPECULATIVE_TOKENS \
    --batch_size 128 \
    --max_model_len 8192 \
    --quantization fp8 \
    --kv_cache_dtype fp8_e5m2 \
    --gpu_memory_utilization 0.85 \
    --max_num_batched_tokens 8192 \
    --max_num_seqs 256

# Expected Performance with MTP:
# - Throughput: 80-120 req/sec (vs 40-60 without MTP)
# - Latency: 30-60ms per request (vs 50-100ms)
# - Memory: ~37-38GB (assistant adds ~2-3GB)
# - Speedup: 2-3x compared to standard decoding
