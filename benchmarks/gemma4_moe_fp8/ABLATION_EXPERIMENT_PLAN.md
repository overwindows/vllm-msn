# Gemma 4 MoE FP8 — Ablation Experiment Plan

**Hardware target**: A100 80 GB (sm_80)  
**Dataset**: `datasets/sc1_delta_v2.jsonl` (sc1 only)  
**Driver**: `bench_ablation.py` via `run_ablation.sh`  
**Results**: `ablation_results/all_runs.csv` + per-run JSON  
**Analysis**: `python3 analyze_ablation.py` → `ablation_results/summary.md`

---

## Fixed scenario parameters (sc1)

| Parameter | Value | Source |
|---|---|---|
| Dataset | `datasets/sc1_delta_v2.jsonl` | `prep_dataset.py --max-keep 1000` |
| `num_prompts` | 1 000 | ablation-sized subset of full 10 000 |
| `output_len` (max_tokens) | 8 192 | matches REPRODUCE_PRODSHAPE |
| `max_model_len` | 24 576 | matches REPRODUCE_PRODSHAPE |
| `max_num_batched_tokens` | 16 384 | matches REPRODUCE_PRODSHAPE |
| Sampling | temp=0.7, top_p=0.95, ignore_eos=False | consistent with bench_offline |
| Reps per experiment | 2 | mean ± σ across reps |
| Attention backend | **TRITON_ATTN** (forced by vLLM) | Gemma 4 heterogeneous head dims |

> **Why TRITON_ATTN is forced**: Gemma 4 has heterogeneous attention head dimensions
> (256 and 512 in different layers). vLLM cannot route a mixed-head-dim model through
> FLASH_ATTN or FLASHINFER, so it falls back to TRITON_ATTN unconditionally regardless
> of the `VLLM_ATTENTION_BACKEND` env var. Setting that var is a no-op for this model.

---

## Dataset preparation

Run once before any experiments:

```bash
cd benchmarks/gemma4_moe_fp8
python3 prep_dataset.py \
  --src /path/to/delta_prompts/ \
  --dst datasets/sc1_delta_v2.jsonl \
  --model google/gemma-4-26B-A4B-it \
  --max-tokens 16384 \
  --max-keep 1000
```

Parameter rationale:
- `--max-tokens 16384` = `max_model_len(24576) − output_len(8192)` — leaves exactly the right headroom for the 8192-token output cap
- `--max-keep 1000` — ablation uses 1000 prompts (vs 10 000 in the full prod benchmark), to keep each experiment under ~30 min on A100

---

## Experiment matrix

### Group A — Reproduce REPRODUCE_PRODSHAPE baseline

Goal: verify the sc1 numbers from REPRODUCE_PRODSHAPE.md land in the right ballpark on A100
(H100 NVL reference: bf16=1870 out tok/s, FP8=2056 out tok/s — A100 will be lower).

| ID | Label | quant | KV dtype | eager | MTP | mns | gpu_mem | model |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E001** | BF16 baseline — matches REPRODUCE_PRODSHAPE sc1 | bf16 | auto | ✓ | ✗ | 128 | 0.90 | full |
| **E002** | +FP8 weights (KV cache stays auto/bf16) | fp8 | auto | ✓ | ✗ | 128 | 0.90 | full |
| **E003** | +FP8 KV cache (fp8_e4m3) — **FAIL expected on A100** | fp8 | fp8_e4m3 | ✓ | ✗ | 128 | 0.90 | full |

**E003 note**: `fp8_e4m3` KV cache requires Triton `fp8e4nv` which is not supported on
sm_80. The expected result is a hard error. On H100 (sm_90) this is the FP8 run from
REPRODUCE_PRODSHAPE. Recording the failure on A100 IS the result.

---

### Group B — Incremental optimizations (stack-up)

Each experiment adds one technique on top of the previous best.

| ID | Label | quant | KV dtype | eager | MTP | mns | gpu_mem | model | Builds on |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E004** | +CUDA graphs | fp8 | auto | ✗ | ✗ | 128 | 0.90 | full | E002 |
| **E005** | +MTP speculative decoding (k=5) | fp8 | auto | ✗ | ✓ k=5 | 128 | 0.90 | full | E004 |
| **E006** | +text-only model (vision tower stripped) | fp8 | auto | ✗ | ✓ k=5 | 128 | 0.90 | text_only | E005 |

**E006 is the "best-so-far" config** that Groups C, D, E branch from.

---

### Group C — Batch size (`max_num_seqs`) sweep

Base config: E006 (FP8, CUDA graphs, MTP k=5, text-only, gpu_mem=0.90).  
Question: is mns=128 optimal, or does A100 saturate earlier/later?

| ID | Label | mns | All other params |
|---|---|:---:|---|
| **E007** | batch sweep: mns=64 | 64 | same as E006 |
| E006 | *(control, mns=128)* | 128 | — |
| **E008** | batch sweep: mns=192 | 192 | same as E006 |
| **E009** | batch sweep: mns=256 | 256 | same as E006 |

---

### Group D — GPU memory utilization sweep

Base config: E006 (mns=128).  
Question: does giving vLLM more KV cache headroom help on A100?

| ID | Label | gpu_mem | All other params |
|---|---|:---:|---|
| **E010** | gpu_mem sweep: 0.80 | 0.80 | same as E006 |
| E006 | *(control, gpu_mem=0.90)* | 0.90 | — |
| **E011** | gpu_mem sweep: 0.95 | 0.95 | same as E006 |

---

### Group E — Isolation (ablate single contributions)

Base config: E006. Each experiment turns off exactly one optimization to measure its
isolated contribution.

| ID | Label | What is turned off | quant | eager | MTP | mns | gpu_mem | model |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **E012** | no MTP at optimal | MTP disabled | fp8 | ✗ | ✗ | 128 | 0.90 | text_only |
| **E013** | no CUDA graphs at optimal | CUDA graphs disabled | fp8 | ✓ | ✓ k=5 | 128 | 0.90 | text_only |
| **E014** | BF16 weights at optimal | FP8 weights removed | bf16 | ✗ | ✓ k=5 | 128 | 0.90 | text_only |
| **E015** | BF16 reference (text-only, no opts) | All opts off | bf16 | ✓ | ✗ | 128 | 0.90 | text_only |

**Isolation pairs** (E006 is the "on" state, column is the "off" state):

| Contribution measured | ON | OFF | Expected sign |
|---|:---:|:---:|:---:|
| MTP k=5 | E006 | E012 | E006 > E012 |
| CUDA graphs | E006 | E013 | E006 ≥ E013 (may regress on heterogeneous batch) |
| FP8 weights | E006 | E014 | E006 > E014 |
| text-only model vs full | E006 | E005 | E006 > E005 |

---

## Complete config table

| ID | Group | quant | KV dtype | eager | MTP k | mns | gpu_mem | model |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| E001 | A | bf16 | auto | ✓ | — | 128 | 0.90 | full |
| E002 | A | fp8 | auto | ✓ | — | 128 | 0.90 | full |
| E003 | A | fp8 | fp8_e4m3 | ✓ | — | 128 | 0.90 | full |
| E004 | B | fp8 | auto | ✗ | — | 128 | 0.90 | full |
| E005 | B | fp8 | auto | ✗ | 5 | 128 | 0.90 | full |
| E006 | B | fp8 | auto | ✗ | 5 | 128 | 0.90 | text_only |
| E007 | C | fp8 | auto | ✗ | 5 | **64** | 0.90 | text_only |
| E008 | C | fp8 | auto | ✗ | 5 | **192** | 0.90 | text_only |
| E009 | C | fp8 | auto | ✗ | 5 | **256** | 0.90 | text_only |
| E010 | D | fp8 | auto | ✗ | 5 | 128 | **0.80** | text_only |
| E011 | D | fp8 | auto | ✗ | 5 | 128 | **0.95** | text_only |
| E012 | E | fp8 | auto | ✗ | **—** | 128 | 0.90 | text_only |
| E013 | E | fp8 | auto | **✓** | 5 | 128 | 0.90 | text_only |
| E014 | E | **bf16** | auto | ✗ | 5 | 128 | 0.90 | text_only |
| E015 | E | **bf16** | auto | **✓** | **—** | 128 | 0.90 | text_only |

Bold = the single parameter that differs from E006.

---

## Running the experiments

### Prepare dataset (once)
```bash
python3 prep_dataset.py \
  --src /path/to/delta_prompts/ \
  --dst datasets/sc1_delta_v2.jsonl \
  --max-tokens 16384 --max-keep 1000
```

### Run all experiments
```bash
./run_ablation.sh --all --scenario sc1 --reps 2
```

### Run a single experiment
```bash
./run_ablation.sh E001 --scenario sc1 --reps 2
```

### Run a group
```bash
./run_ablation.sh E001,E002,E003 --scenario sc1 --reps 2   # Group A
./run_ablation.sh E004,E005,E006 --scenario sc1 --reps 2   # Group B
```

### Analyze results
```bash
python3 analyze_ablation.py
# Output: ablation_results/summary.md
```

---

## Environment variables (set automatically by run_ablation.sh)

| Variable | Value | Notes |
|---|---|---|
| `VLLM_ATTENTION_BACKEND` | `FLASH_ATTN` | No-op for Gemma 4 (TRITON_ATTN forced) |
| `VLLM_USE_FLASHINFER_MOE_FP8` | `0` on A100, `1` on H100 | Auto-detected via compute cap |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | Avoids JIT failures with old nvcc on A100 |

---

## Model paths

| Variable | Default | Used by |
|---|---|---|
| `GEMMA4_MODEL_PATH` | `/mnt/models/gemma-4-26B-A4B-it` | E001–E005, E004 |
| `GEMMA4_TEXT_ONLY_MODEL_PATH` | `$GEMMA4_MODEL_PATH-text-only` | E006–E015 |
| `GEMMA4_ASSISTANT_MODEL_PATH` | `$GEMMA4_MODEL_PATH-assistant` | All MTP experiments (E005–E011, E013, E014) |

Override before running:
```bash
export GEMMA4_MODEL_PATH=/your/local/path/gemma-4-26B-A4B-it
./run_ablation.sh --all
```

---

## Expected outcomes (A100 80 GB)

H100 NVL reference numbers (from REPRODUCE_PRODSHAPE.md §6):

| Config | H100 out tok/s |
|---|---:|
| BF16 baseline (E001 equivalent) | 1 870 ± 14 |
| FP8 weights + FP8 KV (E003 equivalent, H100 only) | 2 056 ± 21 |

A100 will produce lower absolute numbers (~30–50% lower than H100 NVL). The
**ratio** between experiments (e.g. FP8 vs BF16, MTP on vs off) is the portable signal.

From the prior A100 ablation study (`examples/EXPERIMENT_PLAN_ABLATION_STUDY.md`,
different dataset/output_len but same techniques):

| Technique | A100 contribution |
|---|---:|
| FP8 weights | +5.7% |
| CUDA graphs | −2.6% (regressed on heterogeneous batch) |
| MTP k=5 | **+26.8%** (single biggest win) |
| text-only model | −1.8% (marginal on text-only workload) |

These are rough indicators — the new 8192-token output length and larger batch size
may shift these numbers significantly. The isolation experiments (Group E) will give
the definitive answers for this workload.

---

## Files in this directory

| File | Purpose |
|---|---|
| `bench_ablation.py` | Main ablation driver — EXPERIMENTS dict, run logic |
| `run_ablation.sh` | Shell wrapper — sets env vars, calls bench_ablation.py |
| `analyze_ablation.py` | Post-run analysis → ablation_results/summary.md |
| `prep_dataset.py` | One-time dataset generation from raw delta_prompts/ |
| `bench_offline.py` | Original prod-shape benchmark (REPRODUCE_PRODSHAPE) |
| `REPRODUCE_PRODSHAPE.md` | How to reproduce H100 baseline numbers |
| `ABLATION_EXPERIMENT_PLAN.md` | This file |
| `BENCHMARK_LOG.md` | Run log — record results here after each experiment |
| `datasets/sc1_delta_v2.jsonl` | Ablation dataset (generated by prep_dataset.py) |
| `ablation_results/` | Output directory — all_runs.csv + per-run JSON |
