#!/usr/bin/env bash
# run_ablation.sh — Launch one or more ablation experiments.
#
# Usage:
#   ./run_ablation.sh E001               # single experiment, sc1, 2 reps
#   ./run_ablation.sh E001,E003,E006     # comma-separated list
#   ./run_ablation.sh --all              # all 15 experiments
#   ./run_ablation.sh E011 --scenario sc2 --reps 3
#
# Environment variables (can override before calling):
#   GEMMA4_MODEL_PATH            full model checkpoint path
#   GEMMA4_TEXT_ONLY_MODEL_PATH  text-only checkpoint path
#   GEMMA4_ASSISTANT_MODEL_PATH  MTP assistant model path
#
# The script resolves VLLM_ATTENTION_BACKEND and FlashInfer knobs per
# experiment ID rather than having them hard-coded in bench_ablation.py
# (vllm imports freeze env vars at import time, so they must be set here).

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Default model paths (override via env)
# ---------------------------------------------------------------------------
: "${GEMMA4_MODEL_PATH:=/mnt/models/gemma-4-26B-A4B-it}"
: "${GEMMA4_TEXT_ONLY_MODEL_PATH:=${GEMMA4_MODEL_PATH}-text-only}"
: "${GEMMA4_ASSISTANT_MODEL_PATH:=${GEMMA4_MODEL_PATH}-assistant}"
export GEMMA4_MODEL_PATH GEMMA4_TEXT_ONLY_MODEL_PATH GEMMA4_ASSISTANT_MODEL_PATH

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
EXP_IDS=""
SCENARIO="sc1"
REPS=2
RUN_ALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)          RUN_ALL=1; shift ;;
    --scenario)     SCENARIO="$2"; shift 2 ;;
    --reps)         REPS="$2"; shift 2 ;;
    --list)
      python3 bench_ablation.py --exp E001 --list
      exit 0
      ;;
    -h|--help)
      sed -n '2,22p' "$0"     # print the usage block at top of script
      exit 0
      ;;
    *)              EXP_IDS="$1"; shift ;;
  esac
done

if [[ $RUN_ALL -eq 1 ]]; then
  EXP_IDS="E001,E002,E003,E004,E005,E006,E007,E008,E009,E010,E011,E012,E013,E014,E015"
fi

if [[ -z "$EXP_IDS" ]]; then
  echo "ERROR: no experiment ID provided. Use --all or specify e.g. E001"
  exit 1
fi

# ---------------------------------------------------------------------------
# Per-experiment environment variable table.
# These must be set BEFORE vllm is imported, so this shell wrapper is the
# right place (bench_ablation.py also cross-checks and warns on mismatch).
#
# Mapping:
#   VLLM_ATTENTION_BACKEND          : FLASH_ATTN | FLASHINFER
#   VLLM_USE_FLASHINFER_MOE_FP8     : 0 | 1  (H100 only; A100 must use 0)
#   VLLM_USE_FLASHINFER_SAMPLER     : 0 | 1  (JIT compile fix for old nvcc)
# ---------------------------------------------------------------------------
set_env_for_exp() {
  local exp="$1"
  # Default: safe values
  export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
  export VLLM_USE_FLASHINFER_MOE_FP8="0"
  export VLLM_USE_FLASHINFER_SAMPLER="0"

  case "$exp" in
    E001)
      export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
      ;;
    E002)
      export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
      ;;
    E003|E004|E005|E006|E007|E008|E009|E010|E011|E013|E014|E015)
      export VLLM_ATTENTION_BACKEND="FLASHINFER"
      # Enable FlashInfer FP8 MoE only if on H100 (sm_90 / sm_90a).
      # On A100 this raises NotImplementedError; guard by checking compute cap.
      COMPUTE_CAP=$(python3 -c "import torch; cc=torch.cuda.get_device_capability(); print(cc[0]*10+cc[1])" 2>/dev/null || echo "0")
      if [[ "$COMPUTE_CAP" -ge 90 ]]; then
        export VLLM_USE_FLASHINFER_MOE_FP8="1"
        echo "  → H100 detected (sm_${COMPUTE_CAP}): enabling VLLM_USE_FLASHINFER_MOE_FP8=1"
      else
        export VLLM_USE_FLASHINFER_MOE_FP8="0"
        echo "  → non-H100 (sm_${COMPUTE_CAP}): VLLM_USE_FLASHINFER_MOE_FP8=0 (Marlin FP8 MoE fallback)"
      fi
      ;;
    E012)
      export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
      # E012 is the FA2 head-to-head at the optimal config — no FlashInfer
      ;;
    *)
      echo "WARNING: unknown exp '$exp' — using default FLASH_ATTN"
      ;;
  esac

  echo "  ENV: VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND  VLLM_USE_FLASHINFER_MOE_FP8=$VLLM_USE_FLASHINFER_MOE_FP8"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
IFS=',' read -ra EXPS <<< "$EXP_IDS"

echo "=================================================="
echo "  Ablation benchmark"
echo "  Experiments : ${EXP_IDS}"
echo "  Scenario    : ${SCENARIO}"
echo "  Reps        : ${REPS}"
echo "  $(date)"
echo "=================================================="

FAILED=()
for EXP in "${EXPS[@]}"; do
  EXP="$(echo "$EXP" | tr '[:lower:]' '[:upper:]' | tr -d ' ')"
  echo ""
  echo ">>> Setting up environment for ${EXP}"
  set_env_for_exp "$EXP"

  echo ">>> Launching bench_ablation.py --exp ${EXP} --scenario ${SCENARIO} --reps ${REPS}"
  if python3 bench_ablation.py \
      --exp "${EXP}" \
      --scenario "${SCENARIO}" \
      --reps "${REPS}"; then
    echo ">>> ${EXP} COMPLETED"
  else
    echo ">>> ${EXP} FAILED (exit $?)"
    FAILED+=("$EXP")
  fi
done

echo ""
echo "=================================================="
echo "  Ablation run finished"
echo "  Results: ablation_results/all_runs.csv"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "  FAILED experiments: ${FAILED[*]}"
  exit 1
fi
echo "  All experiments PASSED"
echo "=================================================="
