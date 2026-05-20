VLLM_CMD=llm_analyzer_vllm_oaas_async_relay.py
# VLLM_CMD=llm_analyzer_vllm_oaas_async.py

# Set logging levels
export PYTHONWARNINGS="ignore"
export LOGURU_LEVEL="ERROR"
export VLLM_LOG_LEVEL="ERROR"
export PYTHONUNBUFFERED=1
# run these in the same shell that starts vLLM
export TORCH_CUDA_ARCH_LIST="8.0"      # A100 architecture
# A100 Optimization: Use FLASHINFER for best performance
export VLLM_ATTENTION_BACKEND=FLASHINFER
# Alternative: FLASH_ATTN (if FLASHINFER has issues)
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# Or leave unset for autodetect


# MODEL_PATH=/nvmedata/hf_checkpoints/Qwen3-8B/
MODEL_PATH=/nvmedata/hf_checkpoints/Llama-2-7b-chat-hf-bf16

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/nvmedata/chenw/vllm-ra python3 $VLLM_CMD \
    --input_path /nvmedata/chenw/genz/genz_users_20k_format.tsv \
    --output_path /nvmedata/chenw/genz/genz_users_interests_vllm_oaas_async.jsonl \
    --model_path $MODEL_PATH \
    --batch_size 256 \
    --enable_relay_attention