VLLM_CMD=llm_analyzer_vllm_oaas_async_relay.py
# VLLM_CMD=llm_analyzer_vllm_oaas_async.py

# Set logging levels
export PYTHONWARNINGS="ignore"
export LOGURU_LEVEL="ERROR"
export VLLM_LOG_LEVEL="ERROR"
export PYTHONUNBUFFERED=1
# run these in the same shell that starts vLLM
export TORCH_CUDA_ARCH_LIST="8.0"      # any arch ≤ 9.0 works
# export VLLM_ATTENTION_BACKEND=FLASHINFER   # autodetect also fine
# Use FLASH_ATTN instead of FLASHINFER for better CUDA 12.9 compatibility
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# If FLASH_ATTN still has issues, comment out the line above to use default backend


# MODEL_PATH=/nvmedata/hf_checkpoints/Qwen3-8B/
MODEL_PATH=/nvmedata/hf_checkpoints/Llama-2-7b-chat-hf-bf16

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/nvmedata/chenw/vllm-ra python3 $VLLM_CMD \
    --input_path /nvmedata/chenw/genz/genz_users_20k_format.tsv \
    --output_path /nvmedata/chenw/genz/genz_users_interests_vllm_oaas_async.jsonl \
    --model_path $MODEL_PATH \
    --batch_size 256 \
    --enable_relay_attention