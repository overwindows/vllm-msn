#!/bin/bash
# Enhanced Experiment Runner for Ablation Study
# Supports all configuration options needed for systematic experiments

set -e

# =============================================================================
# Configuration Parsing
# =============================================================================

# Defaults
EXPERIMENT_ID="E001"
BACKEND="FLASHINFER"
BATCH_SIZE=128
USE_FP8="true"
USE_CUDA_GRAPHS="false"
USE_MTP="false"
GPU_MEM_UTIL="0.75"
USE_TEXT_ONLY="false"
KV_CACHE_DTYPE="fp8_e5m2"
DRY_RUN="false"
MAX_BATCHED_TOKENS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp|--experiment)
            EXPERIMENT_ID="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --batch|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --fp8)
            USE_FP8="true"
            shift
            ;;
        --no-fp8)
            USE_FP8="false"
            shift
            ;;
        --cuda-graphs)
            USE_CUDA_GRAPHS="true"
            shift
            ;;
        --no-cuda-graphs)
            USE_CUDA_GRAPHS="false"
            shift
            ;;
        --mtp)
            USE_MTP="true"
            shift
            ;;
        --no-mtp)
            USE_MTP="false"
            shift
            ;;
        --gpu-mem)
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        --text-only|--text-only-model)
            USE_TEXT_ONLY="true"
            shift
            ;;
        --kv-cache-dtype)
            KV_CACHE_DTYPE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --max-batched-tokens)
            MAX_BATCHED_TOKENS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --exp, --experiment ID       Experiment ID (default: E001)"
            echo "  --backend TYPE               FLASHINFER|FLASH_ATTN (default: FLASHINFER)"
            echo "  --batch, --batch-size N      Batch size (default: 128)"
            echo "  --fp8                        Enable FP8 quantization (default)"
            echo "  --no-fp8                     Disable FP8 quantization"
            echo "  --cuda-graphs                Enable CUDA graphs"
            echo "  --no-cuda-graphs             Disable CUDA graphs (default)"
            echo "  --mtp                        Enable MTP speculative decoding"
            echo "  --no-mtp                     Disable MTP (default)"
            echo "  --gpu-mem RATIO              GPU memory utilization (default: 0.75)"
            echo "  --text-only                  Use text-only model variant"
            echo "  --kv-cache-dtype TYPE        fp8_e5m2|fp8_e4m3|auto (default: fp8_e5m2)"
            echo "  --max-batched-tokens N       Override max batched tokens"
            echo "  --dry-run                    Show config and exit"
            echo "  -h, --help                   Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --exp E002 --backend FLASHINFER --batch 128 --fp8"
            echo "  $0 --exp E005 --cuda-graphs --mtp --gpu-mem 0.75"
            echo "  $0 --exp E012 --backend FLASH_ATTN --no-fp8 --batch 64"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Calculate max_batched_tokens if not specified
if [ -z "$MAX_BATCHED_TOKENS" ]; then
    MAX_BATCHED_TOKENS=$((BATCH_SIZE * 48))  # Assume avg 48 tokens per sequence
fi

# Determine model path
if [ "$USE_TEXT_ONLY" = "true" ]; then
    MODEL_PATH="/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only"
else
    MODEL_PATH="/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it"
fi

ASSISTANT_PATH="/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant"

# Determine quantization
if [ "$USE_FP8" = "true" ]; then
    DTYPE="bfloat16"
    QUANTIZATION="fp8"
else
    DTYPE="bfloat16"
    QUANTIZATION="none"
    KV_CACHE_DTYPE="auto"  # No FP8 KV cache without FP8
fi

# =============================================================================
# Display Configuration
# =============================================================================

echo "=========================================="
echo "Gemma 4 MoE Ablation Experiment Runner"
echo "=========================================="
echo ""
echo "Experiment: ${EXPERIMENT_ID}"
echo ""
echo "Configuration:"
echo "  Model:                ${MODEL_PATH##*/}"
echo "  Backend:              ${BACKEND}"
echo "  Batch Size:           ${BATCH_SIZE}"
echo "  Max Batched Tokens:   ${MAX_BATCHED_TOKENS}"
echo "  FP8 Quantization:     ${USE_FP8}"
echo "  CUDA Graphs:          ${USE_CUDA_GRAPHS}"
echo "  MTP:                  ${USE_MTP}"
echo "  GPU Memory Util:      ${GPU_MEM_UTIL}"
echo "  KV Cache Dtype:       ${KV_CACHE_DTYPE}"
echo "  Text-Only Model:      ${USE_TEXT_ONLY}"
echo ""

if [ "$DRY_RUN" = "true" ]; then
    echo "Dry run mode - configuration displayed, exiting."
    exit 0
fi

# =============================================================================
# Setup
# =============================================================================

OUTPUT_DIR="./experiment_results/${EXPERIMENT_ID}"
mkdir -p "${OUTPUT_DIR}"

echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Activate environment
source /root/miniconda3/bin/activate vllm

# =============================================================================
# Environment Configuration
# =============================================================================

echo "Step 1: Configuring environment..."

# Backend
case "${BACKEND}" in
    FLASHINFER)
        export VLLM_ATTENTION_BACKEND=FLASHINFER
        if [ "$USE_FP8" = "true" ]; then
            export VLLM_USE_FLASHINFER_MOE_FP8=1
        fi
        echo "  ✓ FlashInfer backend configured"
        ;;
    FLASH_ATTN)
        export VLLM_ATTENTION_BACKEND=FLASH_ATTN
        unset VLLM_USE_FLASHINFER_MOE_FP8
        echo "  ✓ Flash Attention 2 backend configured"
        ;;
    *)
        echo "  ✗ Unknown backend: ${BACKEND}"
        exit 1
        ;;
esac

# Logging
export VLLM_LOGGING_LEVEL=INFO

echo ""

# =============================================================================
# Log Environment Info
# =============================================================================

echo "Step 2: Logging environment information..."
{
    echo "# Experiment ${EXPERIMENT_ID} - Environment Info"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "## Configuration"
    echo "Model: ${MODEL_PATH}"
    echo "Backend: ${BACKEND}"
    echo "Batch Size: ${BATCH_SIZE}"
    echo "Max Batched Tokens: ${MAX_BATCHED_TOKENS}"
    echo "FP8: ${USE_FP8}"
    echo "Quantization: ${QUANTIZATION}"
    echo "KV Cache Dtype: ${KV_CACHE_DTYPE}"
    echo "CUDA Graphs: ${USE_CUDA_GRAPHS}"
    echo "MTP: ${USE_MTP}"
    echo "GPU Memory Util: ${GPU_MEM_UTIL}"
    echo "Text-Only: ${USE_TEXT_ONLY}"
    echo ""
    echo "## Hardware"
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
    echo ""
    echo "## Software"
    python --version
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || echo "vLLM: dev"
    echo "Attention Backend: ${VLLM_ATTENTION_BACKEND}"
    echo "FlashInfer MoE FP8: ${VLLM_USE_FLASHINFER_MOE_FP8:-disabled}"
    echo ""
    echo "## Git Commit"
    cd /nvmedata/chenw/vllm-ra
    git log -1 --oneline
    git rev-parse HEAD
} > "${OUTPUT_DIR}/environment.txt"

echo "  ✓ Environment info saved"
echo ""

# =============================================================================
# Start Monitoring
# =============================================================================

echo "Step 3: Starting GPU monitor..."
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu \
    --format=csv -l 1 > "${OUTPUT_DIR}/gpu_trace.csv" &
MONITOR_PID=$!
echo "  ✓ GPU monitor started (PID: ${MONITOR_PID})"
echo ""

# Initial state
nvidia-smi > "${OUTPUT_DIR}/gpu_initial.txt"

# =============================================================================
# Build Python Command
# =============================================================================

echo "Step 4: Building inference command..."

PYTHON_CMD="python3 run_inference_configurable.py"

PYTHON_ARGS=(
    "--model_path" "${MODEL_PATH}"
    "--max_num_seqs" "${BATCH_SIZE}"
    "--max_num_batched_tokens" "${MAX_BATCHED_TOKENS}"
    "--gpu_memory_utilization" "${GPU_MEM_UTIL}"
    "--dtype" "${DTYPE}"
    "--output_path" "${OUTPUT_DIR}/output.jsonl"
    "--input_path" "/nvmedata/data/layer1_delta_1k_test.txt"
    "--num_test_samples" "1000"
)

# Quantization
if [ "$USE_FP8" = "true" ]; then
    PYTHON_ARGS+=("--quantization" "fp8")
    PYTHON_ARGS+=("--kv_cache_dtype" "${KV_CACHE_DTYPE}")
else
    # No quantization - explicitly set to None is handled by not passing it
    :  # No-op
fi

# CUDA graphs
if [ "$USE_CUDA_GRAPHS" = "true" ]; then
    PYTHON_ARGS+=("--enable_cuda_graphs")
else
    PYTHON_ARGS+=("--enforce_eager")
fi

# MTP
if [ "$USE_MTP" = "true" ]; then
    PYTHON_ARGS+=("--speculative_model" "${ASSISTANT_PATH}")
    PYTHON_ARGS+=("--num_speculative_tokens" "5")
fi

echo "  Command: ${PYTHON_CMD}"
echo "  Arguments:"
for arg in "${PYTHON_ARGS[@]}"; do
    echo "    $arg"
done
echo ""

# =============================================================================
# Run Experiment
# =============================================================================

echo "Step 5: Running experiment..."
echo "  Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

START_TIME=$(date +%s)

cd /nvmedata/chenw/vllm-ra/examples

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/nvmedata/chenw/vllm-ra \
time ${PYTHON_CMD} "${PYTHON_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/inference.log"

EXPERIMENT_STATUS=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "  End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Duration: ${DURATION} seconds ($((DURATION / 60)) minutes)"
echo "  Status: $([ ${EXPERIMENT_STATUS} -eq 0 ] && echo '✓ Success' || echo '✗ Failed')"
echo ""

# =============================================================================
# Stop Monitoring
# =============================================================================

echo "Step 6: Stopping monitor..."
kill ${MONITOR_PID} 2>/dev/null || true
echo "  ✓ GPU monitor stopped"
echo ""

# Final state
nvidia-smi > "${OUTPUT_DIR}/gpu_final.txt"

# =============================================================================
# Analyze Results
# =============================================================================

echo "Step 7: Analyzing results..."

{
    echo "# Experiment ${EXPERIMENT_ID} - Results"
    echo ""
    echo "## Configuration"
    echo "Experiment ID: ${EXPERIMENT_ID}"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Status: $([ ${EXPERIMENT_STATUS} -eq 0 ] && echo 'Success' || echo 'Failed')"
    echo ""
    echo "Model: ${MODEL_PATH##*/}"
    echo "Backend: ${BACKEND}"
    echo "Batch Size: ${BATCH_SIZE}"
    echo "Max Batched Tokens: ${MAX_BATCHED_TOKENS}"
    echo "FP8: ${USE_FP8}"
    echo "CUDA Graphs: ${USE_CUDA_GRAPHS}"
    echo "MTP: ${USE_MTP}"
    echo "GPU Memory Util: ${GPU_MEM_UTIL}"
    echo "KV Cache Dtype: ${KV_CACHE_DTYPE}"
    echo ""
    echo "## Performance"
    echo "Duration: ${DURATION} seconds ($((DURATION / 60)) minutes)"

    if [ -f "${OUTPUT_DIR}/inference.log" ]; then
        # Extract vLLM stats if available
        THROUGHPUT=$(grep "throughput" "${OUTPUT_DIR}/inference.log" | tail -1 || echo "N/A")
        LATENCY=$(grep "latency" "${OUTPUT_DIR}/inference.log" | tail -1 || echo "N/A")

        if [ "$THROUGHPUT" != "N/A" ]; then
            echo "Throughput: ${THROUGHPUT}"
        fi

        if [ "$LATENCY" != "N/A" ]; then
            echo "Latency: ${LATENCY}"
        fi
    fi

    echo ""
    echo "## Memory"
    if [ -f "${OUTPUT_DIR}/gpu_trace.csv" ]; then
        # Skip header and calculate stats
        PEAK_MEM=$(tail -n +2 "${OUTPUT_DIR}/gpu_trace.csv" | cut -d',' -f2 | sed 's/ MiB//' | sort -n | tail -1)
        AVG_MEM=$(tail -n +2 "${OUTPUT_DIR}/gpu_trace.csv" | cut -d',' -f2 | sed 's/ MiB//' | awk '{sum+=$1; count++} END {if(count>0) printf "%.0f", sum/count; else print "N/A"}')
        PEAK_UTIL=$(tail -n +2 "${OUTPUT_DIR}/gpu_trace.csv" | cut -d',' -f4 | sed 's/ %//' | sort -n | tail -1)
        AVG_UTIL=$(tail -n +2 "${OUTPUT_DIR}/gpu_trace.csv" | cut -d',' -f4 | sed 's/ %//' | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}')

        echo "Peak Memory: ${PEAK_MEM} MiB"
        echo "Average Memory: ${AVG_MEM} MiB"
        echo "Peak GPU Utilization: ${PEAK_UTIL}%"
        echo "Average GPU Utilization: ${AVG_UTIL}%"
    fi

    echo ""
    echo "## Files"
    echo "- Environment: ${OUTPUT_DIR}/environment.txt"
    echo "- Inference Log: ${OUTPUT_DIR}/inference.log"
    echo "- GPU Trace: ${OUTPUT_DIR}/gpu_trace.csv"
    echo "- Output: ${OUTPUT_DIR}/output.jsonl"
    echo "- GPU Initial: ${OUTPUT_DIR}/gpu_initial.txt"
    echo "- GPU Final: ${OUTPUT_DIR}/gpu_final.txt"
    echo ""
    echo "## Errors"
    ERROR_COUNT=$(grep -ic "error\|exception\|traceback" "${OUTPUT_DIR}/inference.log" 2>/dev/null || echo "0")
    if [ ${ERROR_COUNT} -gt 0 ]; then
        echo "⚠ Found ${ERROR_COUNT} error mentions"
        echo ""
        echo "Sample errors:"
        grep -i "error\|exception" "${OUTPUT_DIR}/inference.log" | head -10
    else
        echo "✓ No errors detected"
    fi

} > "${OUTPUT_DIR}/summary.md"

cat "${OUTPUT_DIR}/summary.md"
echo ""

# =============================================================================
# Final Summary
# =============================================================================

echo "=========================================="
if [ ${EXPERIMENT_STATUS} -eq 0 ]; then
    echo "✓ Experiment ${EXPERIMENT_ID} Complete!"
else
    echo "✗ Experiment ${EXPERIMENT_ID} Failed"
fi
echo "=========================================="
echo ""
echo "Results: ${OUTPUT_DIR}/"
echo ""
echo "Quick commands:"
echo "  View summary:     cat ${OUTPUT_DIR}/summary.md"
echo "  View full log:    less ${OUTPUT_DIR}/inference.log"
echo "  Check errors:     grep -i error ${OUTPUT_DIR}/inference.log"
echo "  Plot GPU usage:   python3 plot_gpu_trace.py ${OUTPUT_DIR}/gpu_trace.csv"
echo ""

exit ${EXPERIMENT_STATUS}
