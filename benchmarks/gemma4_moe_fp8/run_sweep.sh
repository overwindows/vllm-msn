#!/usr/bin/env bash
# run_sweep.sh — wrapper that sets the env vars bench_sweep.py needs
# (mirrors run_ablation.sh's env-setup for the E011-equivalent config:
# FP8 weights → FlashInfer FP8 MoE off on A100, FlashInfer sampler off).
#
# Usage:
#   ./run_sweep.sh --phase 1                 # 1-D sweeps, sc1, 2 reps
#   ./run_sweep.sh --phase 2 --reps 1        # 1D+2D, 1 rep
#   ./run_sweep.sh --phase 3 --skip-existing # full grid, resume
#   ./run_sweep.sh --list                    # show cells (no GPU)
#   ./run_sweep.sh --only SW_mns128_mk5_mnbt16384_cpD
#
# All args are passed straight through to bench_sweep.py.

set -euo pipefail
cd "$(dirname "$0")"

# Resolve a snapshot dir under an HF cache root (models--<org>--<name>/snapshots/<sha>).
_hf_snapshot() {
  local root="$1" repo="$2"
  local d="${root}/models--${repo//\//--}/snapshots"
  [[ -d "$d" ]] || return 1
  # shellcheck disable=SC2012
  ls -1d "$d"/*/ 2>/dev/null | head -1 | sed 's:/$::'
}

: "${HF_HOME:=/scratch/hf_cache}"
export HF_HOME

# Local converted text-only checkpoint
: "${GEMMA4_TEXT_ONLY_MODEL_PATH:=/scratch/hf_cache/local_models/gemma-4-26B-A4B-it-text-only}"

# Resolve HF-cache snapshots for the base + assistant models. Search both
# {HF_HOME} and {HF_HOME}/hub (transformers/HF Hub cache layouts).
_resolve_one() {
  local repo="$1" path=""
  for root in "$HF_HOME" "$HF_HOME/hub" /scratch/hf_cache /scratch/hf_cache/hub; do
    path="$(_hf_snapshot "$root" "$repo" 2>/dev/null || true)"
    [[ -n "$path" ]] && { echo "$path"; return; }
  done
}
: "${GEMMA4_MODEL_PATH:=$(_resolve_one google/gemma-4-26B-A4B-it)}"
: "${GEMMA4_ASSISTANT_MODEL_PATH:=$(_resolve_one google/gemma-4-26B-A4B-it-assistant)}"
export GEMMA4_MODEL_PATH GEMMA4_TEXT_ONLY_MODEL_PATH GEMMA4_ASSISTANT_MODEL_PATH

echo "  GEMMA4_MODEL_PATH=$GEMMA4_MODEL_PATH"
echo "  GEMMA4_TEXT_ONLY_MODEL_PATH=$GEMMA4_TEXT_ONLY_MODEL_PATH"
echo "  GEMMA4_ASSISTANT_MODEL_PATH=$GEMMA4_ASSISTANT_MODEL_PATH"
for p in "$GEMMA4_TEXT_ONLY_MODEL_PATH" "$GEMMA4_ASSISTANT_MODEL_PATH"; do
  [[ -d "$p" ]] || { echo "  ERROR: not a directory: $p"; exit 2; }
done

# Anchor uses FP8 weights → on A100 we keep FlashInfer FP8 MoE off (Marlin
# fallback). On H100 (sm_90+) we enable it. Logged either way for repro.
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"   # Gemma 4 forces TRITON_ATTN anyway; logged only
export VLLM_USE_FLASHINFER_SAMPLER="0"

COMPUTE_CAP=$(python3 -c "import torch; cc=torch.cuda.get_device_capability(); print(cc[0]*10+cc[1])" 2>/dev/null || echo "0")
if [[ "$COMPUTE_CAP" -ge 90 ]]; then
  export VLLM_USE_FLASHINFER_MOE_FP8="1"
  echo "  → H100 detected (sm_${COMPUTE_CAP}): VLLM_USE_FLASHINFER_MOE_FP8=1"
else
  export VLLM_USE_FLASHINFER_MOE_FP8="0"
  echo "  → non-H100 (sm_${COMPUTE_CAP}): VLLM_USE_FLASHINFER_MOE_FP8=0 (Marlin fallback)"
fi

LOG_DIR="ablation_results"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/sweep_${TS}.log"

echo "Sweep log → $LOG_FILE"
python3 -u bench_sweep.py "$@" 2>&1 | tee "$LOG_FILE"
