# Gemma 4 26B MoE Ablation Study - Experiment Plan

**Objective:** Systematically measure the impact of each optimization on Gemma 4 26B MoE inference performance.

**Hardware:** NVIDIA A100 40GB
**Baseline Model:** google/gemma-4-26B-A4B-it
**Metrics:** Throughput (tokens/sec), Latency (ms/token), Memory (GB), GPU Utilization (%)

---

## Table of Contents

1. [Experiment Matrix](#experiment-matrix)
2. [Baseline Configuration](#baseline-configuration)
3. [Experiment Configurations](#experiment-configurations)
4. [Execution Plan](#execution-plan)
5. [Expected Results](#expected-results)
6. [Analysis Guidelines](#analysis-guidelines)

---

## Experiment Matrix

### Overview

```
Experiments organized in 3 groups:

GROUP A: Core Optimizations (E001-E007)
├─ Baseline → FP8 → FlashInfer → CUDA Graphs → MTP
└─ Build up optimizations cumulatively

GROUP B: Memory Optimizations (E008-E011)
├─ Vision weight removal
├─ Batch size scaling
└─ gpu_memory_utilization tuning

GROUP C: Alternative Configurations (E012-E015)
├─ Backend comparison
├─ Quantization comparison
└─ KV cache dtype comparison

Total: 15 experiments
Estimated time: 2-3 hours (including warmup and multiple runs)
```

### Quick Reference Table

| Exp | Name | FP8 | Backend | CUDA | MTP | Batch | Vision | Mem% | Expected Δ |
|-----|------|-----|---------|------|-----|-------|--------|------|-----------|
| E001 | Baseline | ✗ | FA2 | ✗ | ✗ | 64 | ✓ | 0.95 | Baseline (OOM likely) |
| E002 | +FP8 | ✓ | FA2 | ✗ | ✗ | 64 | ✓ | 0.85 | +2× (memory) |
| E003 | +FlashInfer | ✓ | FI | ✗ | ✗ | 64 | ✓ | 0.85 | +8% |
| E004 | +Batch128 | ✓ | FI | ✗ | ✗ | 128 | ✓ | 0.85 | +30% |
| E005 | +CUDAGraphs | ✓ | FI | ✓ | ✗ | 128 | ✓ | 0.75 | +50% |
| E006 | +MTP | ✓ | FI | ✓ | ✓ | 128 | ✓ | 0.75 | +2-3× |
| E007 | -Vision | ✓ | FI | ✓ | ✓ | 128 | ✗ | 0.75 | +1-2% |
| E008 | Batch192 | ✓ | FI | ✓ | ✓ | 192 | ✗ | 0.75 | +20% |
| E009 | Batch256 | ✓ | FI | ✓ | ✓ | 256 | ✗ | 0.75 | +30% (may OOM) |
| E010 | Mem70 | ✓ | FI | ✓ | ✓ | 128 | ✗ | 0.70 | +5-10% |
| E011 | Mem80 | ✓ | FI | ✓ | ✓ | 128 | ✗ | 0.80 | -5% (less headroom) |
| E012 | FA2vsFI | ✓ | FA2 | ✓ | ✓ | 128 | ✗ | 0.75 | -8% |
| E013 | NoMTP | ✓ | FI | ✓ | ✗ | 128 | ✗ | 0.75 | -2-3× |
| E014 | KV_E4M3 | ✓ | FI | ✓ | ✓ | 128 | ✗ | 0.75 | +2-3% |
| E015 | BF16Full | ✗ | FI | ✗ | ✗ | 32 | ✗ | 0.95 | -2×, OOM risk |

Legend:
- FP8: FP8 quantization enabled
- Backend: FA2 (Flash Attention 2), FI (FlashInfer)
- CUDA: CUDA graphs enabled
- MTP: Multi-Token Prediction with assistant model
- Batch: max_num_seqs
- Vision: Vision weights present
- Mem%: gpu_memory_utilization
- Expected Δ: Expected change vs previous experiment
```

---

## Baseline Configuration

### E001: Naive Baseline (Expected: OOM or very slow)

**Purpose:** Establish baseline without any optimizations

```bash
# Configuration
MODEL_PATH=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it
DTYPE=bfloat16
QUANTIZATION=None  # No quantization
TENSOR_PARALLEL=1
GPU_MEMORY_UTIL=0.95
MAX_NUM_SEQS=64  # Small batch to avoid OOM
MAX_NUM_BATCHED_TOKENS=3072
ENFORCE_EAGER=True  # No CUDA graphs

# Environment
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # Flash Attention 2
unset VLLM_USE_FLASHINFER_MOE_FP8

# Expected Issues
⚠️  Likely OOM (model ~49GB BF16 > 40GB available)
⚠️  If runs: Very slow, low throughput
⚠️  High memory usage, low GPU utilization
```

**Run Command:**
```bash
./experiment_runner.sh E001 FLASH_ATTN 64 \
    --no-fp8 \
    --no-cuda-graphs \
    --no-mtp \
    --gpu-mem 0.95
```

**Expected Metrics:**
- Throughput: ~500-800 tokens/sec (if doesn't OOM)
- Latency: ~80-120 ms/token
- Memory: 39-40 GB (very tight, likely OOM)
- GPU Util: 20-30%

---

## Experiment Configurations

### GROUP A: Core Optimizations (Cumulative)

#### E002: Add FP8 Quantization

**Purpose:** Measure FP8 quantization impact (most critical optimization)

```bash
# Changes from E001
QUANTIZATION=fp8
KV_CACHE_DTYPE=fp8_e5m2
GPU_MEMORY_UTIL=0.85
MAX_NUM_SEQS=64

# Environment
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Expected
✓ Model fits in memory (~24GB FP8)
✓ 2× memory reduction
✓ Slightly higher throughput (less memory pressure)
```

**Run Command:**
```bash
./experiment_runner.sh E002 FLASH_ATTN 64 \
    --fp8 \
    --no-cuda-graphs \
    --no-mtp \
    --gpu-mem 0.85
```

**Expected Metrics:**
- Throughput: ~800-1000 tokens/sec
- Latency: ~60-80 ms/token
- Memory: 28-32 GB
- GPU Util: 30-40%

**Key Question:** Does FP8 enable model to run? How much throughput improvement?

---

#### E003: Switch to FlashInfer Backend

**Purpose:** Measure FlashInfer vs Flash Attention 2 for sparse attention

```bash
# Changes from E002
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_FLASHINFER_MOE_FP8=1

# Expected
✓ 8-10% faster on sliding window attention (83% of layers)
✓ Better MoE FP8 kernel integration
✓ Lower memory for block-sparse patterns
```

**Run Command:**
```bash
./experiment_runner.sh E003 FLASHINFER 64 \
    --fp8 \
    --no-cuda-graphs \
    --no-mtp \
    --gpu-mem 0.85
```

**Expected Metrics:**
- Throughput: ~880-1100 tokens/sec (+8-10% vs E002)
- Latency: ~55-70 ms/token
- Memory: 27-31 GB (slightly lower)
- GPU Util: 35-45%

**Key Question:** How much does FlashInfer improve performance vs FA2?

---

#### E004: Increase Batch Size to 128

**Purpose:** Measure batch size impact on MoE amortization

```bash
# Changes from E003
MAX_NUM_SEQS=128  # 2× larger
MAX_NUM_BATCHED_TOKENS=6144

# Expected
✓ Expert loading amortized over 2× tokens
✓ 25-30% higher throughput
✓ Higher memory usage (KV cache grows)
```

**Run Command:**
```bash
./experiment_runner.sh E004 FLASHINFER 128 \
    --fp8 \
    --no-cuda-graphs \
    --no-mtp \
    --gpu-mem 0.85
```

**Expected Metrics:**
- Throughput: ~1150-1400 tokens/sec (+30% vs E003)
- Latency: ~55-65 ms/token (similar)
- Memory: 32-36 GB (+KV cache)
- GPU Util: 40-50%

**Key Question:** How much does larger batch improve throughput?

---

#### E005: Enable CUDA Graphs

**Purpose:** Measure CUDA graph compilation benefit

```bash
# Changes from E004
ENFORCE_EAGER=False
GPU_MEMORY_UTIL=0.75  # Reduce for CUDA graph overhead
MAX_NUM_SEQS=128

# Expected
✓ 40-50% faster (kernel launch overhead eliminated)
✓ +4-6 GB memory for compiled graphs
✓ Longer warmup time
```

**Run Command:**
```bash
./experiment_runner.sh E005 FLASHINFER 128 \
    --fp8 \
    --cuda-graphs \
    --no-mtp \
    --gpu-mem 0.75
```

**Expected Metrics:**
- Throughput: ~1700-2100 tokens/sec (+50% vs E004)
- Latency: ~35-45 ms/token
- Memory: 34-38 GB (+CUDA graphs)
- GPU Util: 45-55%

**Key Question:** Does CUDA graph provide expected 50% speedup?

---

#### E006: Add MTP (Multi-Token Prediction)

**Purpose:** Measure MTP speculative decoding benefit

```bash
# Changes from E005
SPECULATIVE_MODEL=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant
NUM_SPECULATIVE_TOKENS=5

# Expected
✓ 2-3× higher throughput (generates 5 tokens per iteration)
✓ +0.8 GB memory for assistant model
✓ Latency per "effective token" much lower
```

**Run Command:**
```bash
./experiment_runner.sh E006 FLASHINFER 128 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75
```

**Expected Metrics:**
- Throughput: ~4000-6000 tokens/sec (2-3× vs E005)
- Latency: ~15-25 ms/token (effective)
- Memory: 35-39 GB (+assistant)
- GPU Util: 50-60%

**Key Question:** Does MTP achieve expected 2-3× speedup? What's acceptance rate?

---

#### E007: Remove Vision Weights

**Purpose:** Measure benefit of removing unused vision components

```bash
# Changes from E006
MODEL_PATH=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only

# Create text-only model first:
python3 create_text_only_model.py \
    --model_path /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \
    --output_path /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only

# Expected
✓ -1.5 GB memory
✓ Slightly faster loading
✓ Enables larger batch sizes
```

**Run Command:**
```bash
./experiment_runner.sh E007 FLASHINFER 128 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75 \
    --text-only-model
```

**Expected Metrics:**
- Throughput: ~4100-6200 tokens/sec (+1-3% vs E006)
- Latency: ~15-24 ms/token
- Memory: 33-37 GB (-1.5 GB)
- GPU Util: 50-60%

**Key Question:** Does vision removal free up enough memory for larger batches?

---

### GROUP B: Memory Optimizations

#### E008: Test Batch Size 192

**Purpose:** Measure throughput with larger batch (enabled by vision removal)

```bash
# Changes from E007
MAX_NUM_SEQS=192
MAX_NUM_BATCHED_TOKENS=9216

# Expected
✓ 50% larger batch
✓ +20% throughput (better expert amortization)
✓ +3-4 GB memory (KV cache)
⚠️  May be tight on memory (test carefully)
```

**Run Command:**
```bash
./experiment_runner.sh E008 FLASHINFER 192 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75 \
    --text-only-model
```

**Expected Metrics:**
- Throughput: ~4900-7400 tokens/sec (+20% vs E007)
- Latency: ~13-20 ms/token
- Memory: 37-39 GB (near limit)
- GPU Util: 55-65%

**Key Question:** Can we fit batch=192? How much throughput gain?

---

#### E009: Test Batch Size 256

**Purpose:** Find maximum sustainable batch size

```bash
# Changes from E007
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=12288

# Expected
⚠️  May OOM (40-42 GB required)
✓ If fits: +30% throughput vs E007
?  May need to reduce gpu_memory_utilization
```

**Run Command:**
```bash
./experiment_runner.sh E009 FLASHINFER 256 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75 \
    --text-only-model
```

**Expected Metrics:**
- Throughput: ~5300-8000 tokens/sec (if fits)
- Latency: ~12-19 ms/token
- Memory: 39-41 GB (likely OOM)
- GPU Util: 60-70%

**Key Question:** Does batch=256 OOM? If yes, what's max batch?

---

#### E010: Lower Memory Utilization (0.70)

**Purpose:** Test if lower allocation enables better dynamic batching

```bash
# Changes from E007
GPU_MEMORY_UTIL=0.70  # More headroom
MAX_NUM_SEQS=128

# Expected
✓ More memory headroom for vLLM dynamic batching
✓ Better handling of variable sequence lengths
?  May improve average throughput in production
```

**Run Command:**
```bash
./experiment_runner.sh E010 FLASHINFER 128 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.70 \
    --text-only-model
```

**Expected Metrics:**
- Throughput: ~4300-6600 tokens/sec (+5-10% vs E007 on variable load)
- Latency: ~14-23 ms/token
- Memory: 30-34 GB (lower peak)
- GPU Util: 50-60%

**Key Question:** Does lower utilization improve dynamic batching?

---

#### E011: Higher Memory Utilization (0.80)

**Purpose:** Test if higher allocation causes issues

```bash
# Changes from E007
GPU_MEMORY_UTIL=0.80  # Less headroom
MAX_NUM_SEQS=128

# Expected
⚠️  Less headroom for CUDA graphs (may cause issues)
⚠️  Less flexibility for dynamic batching
✓ Slightly higher KV cache capacity
```

**Run Command:**
```bash
./experiment_runner.sh E011 FLASHINFER 128 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.80 \
    --text-only-model
```

**Expected Metrics:**
- Throughput: ~3900-5900 tokens/sec (-5% vs E007, may be unstable)
- Latency: ~16-25 ms/token
- Memory: 36-39 GB (higher, riskier)
- GPU Util: 50-60%

**Key Question:** Does higher utilization cause OOM or instability?

---

### GROUP C: Alternative Configurations

#### E012: Compare Flash Attention 2 (with all optimizations)

**Purpose:** Validate FlashInfer is better than FA2

```bash
# Changes from E007
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
unset VLLM_USE_FLASHINFER_MOE_FP8

# Expected
✗ 8-10% slower than FlashInfer
✗ Higher memory usage for sliding window
✓ Validates our FlashInfer choice
```

**Run Command:**
```bash
./experiment_runner.sh E012 FLASH_ATTN 128 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75 \
    --text-only-model
```

**Expected Metrics:**
- Throughput: ~3700-5700 tokens/sec (-8% vs E007)
- Latency: ~16-27 ms/token
- Memory: 35-39 GB
- GPU Util: 45-55%

**Key Question:** Confirm FlashInfer is faster than FA2?

---

#### E013: Disable MTP (measure MTP contribution)

**Purpose:** Isolate MTP contribution to performance

```bash
# Changes from E007
# Remove MTP
SPECULATIVE_MODEL=None
NUM_SPECULATIVE_TOKENS=0

# Expected
✗ 2-3× lower throughput (no speculative decoding)
✓ -0.8 GB memory
✓ Confirms MTP value
```

**Run Command:**
```bash
./experiment_runner.sh E013 FLASHINFER 128 \
    --fp8 \
    --cuda-graphs \
    --no-mtp \
    --gpu-mem 0.75 \
    --text-only-model
```

**Expected Metrics:**
- Throughput: ~1700-2100 tokens/sec (-60% vs E007)
- Latency: ~35-45 ms/token (2-3× slower)
- Memory: 32-36 GB
- GPU Util: 45-55%

**Key Question:** Confirm MTP provides 2-3× speedup?

---

#### E014: Test FP8_E4M3 KV Cache

**Purpose:** Compare E5M2 vs E4M3 KV cache formats

```bash
# Changes from E007
KV_CACHE_DTYPE=fp8_e4m3  # Higher precision mantissa

# Expected
✓ Slightly better accuracy (more mantissa bits)
✓ 2-3% faster (better tensor core utilization?)
?  May have rounding differences
```

**Run Command:**
```bash
./experiment_runner.sh E014 FLASHINFER 128 \
    --fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75 \
    --text-only-model
```

**Expected Metrics:**
- Throughput: ~4100-6400 tokens/sec (+1-3% vs E007)
- Latency: ~15-24 ms/token
- Memory: 33-37 GB (same)
- GPU Util: 50-60%

**Key Question:** Does E4M3 provide better accuracy or performance?

---

#### E015: Full BF16 (Reference Baseline)

**Purpose:** Compare to full-precision baseline (if it fits)

```bash
# Changes
DTYPE=bfloat16
QUANTIZATION=None
KV_CACHE_DTYPE=auto
MAX_NUM_SEQS=32  # Very small to avoid OOM
GPU_MEMORY_UTIL=0.95
ENFORCE_EAGER=True
NO MTP

# Expected
⚠️  May OOM (49 GB model)
✗ 2× slower (if fits)
✓ Reference for accuracy
```

**Run Command:**
```bash
./experiment_runner.sh E015 FLASHINFER 32 \
    --no-fp8 \
    --no-cuda-graphs \
    --no-mtp \
    --gpu-mem 0.95 \
    --text-only-model
```

**Expected Metrics:**
- Throughput: ~400-600 tokens/sec (may OOM)
- Latency: ~80-130 ms/token
- Memory: 39-40 GB (very tight)
- GPU Util: 25-35%

**Key Question:** Can full BF16 even run? How much slower than FP8?

---

## Execution Plan

### Phase 1: Setup (30 minutes)

```bash
# 1. Create text-only model variant
cd /nvmedata/chenw/vllm-ra/examples
python3 create_text_only_model.py \
    --model_path /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \
    --output_path /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only

# 2. Prepare experiment runner
chmod +x experiment_runner.sh

# 3. Create results directory
mkdir -p experiment_results/ablation_study
cd experiment_results/ablation_study

# 4. Test baseline (quick smoke test)
../../experiment_runner.sh E001 FLASH_ATTN 64 --no-fp8 --no-cuda-graphs --no-mtp --gpu-mem 0.95 --dry-run
```

### Phase 2: Core Experiments (GROUP A) - 60 minutes

```bash
# Run experiments E001-E007 sequentially
for exp in E001 E002 E003 E004 E005 E006 E007; do
    echo "=== Running ${exp} ==="
    ../../experiment_runner.sh ${exp} ...
    sleep 30  # Cool down between experiments
done
```

**Critical checkpoints:**
- E001: May OOM (acceptable, just document)
- E002: Must succeed (FP8 enables model to run)
- E005: Must leave 4-6 GB for CUDA graphs
- E006: MTP should show 2-3× speedup

### Phase 3: Memory Experiments (GROUP B) - 45 minutes

```bash
# Run experiments E008-E011
for exp in E008 E009 E010 E011; do
    echo "=== Running ${exp} ==="
    ../../experiment_runner.sh ${exp} ...
    # Check for OOM, adjust if needed
    sleep 30
done
```

**Critical checkpoints:**
- E008: Monitor memory closely (may be tight)
- E009: Likely to OOM (document max batch size)
- E010-E011: Compare dynamic batching behavior

### Phase 4: Comparisons (GROUP C) - 45 minutes

```bash
# Run experiments E012-E015
for exp in E012 E013 E014 E015; do
    echo "=== Running ${exp} ==="
    ../../experiment_runner.sh ${exp} ...
    sleep 30
done
```

**Critical checkpoints:**
- E012: Should be slower than E007 (confirms FlashInfer)
- E013: Should be 2-3× slower (confirms MTP value)
- E015: May OOM (document if so)

### Phase 5: Analysis (30 minutes)

```bash
# Generate comparison report
python3 ../../analyze_experiments.py \
    --results-dir experiment_results/ablation_study \
    --output ablation_study_report.md

# Create visualizations
python3 ../../plot_experiments.py \
    --results-dir experiment_results/ablation_study \
    --output ablation_study_plots/
```

---

## Expected Results

### Cumulative Improvement (GROUP A)

```
Experiment       Throughput    vs E001    Cumulative    Key Optimization
────────────────────────────────────────────────────────────────────────
E001 Baseline    ~700 tok/s    1.0×       1.0×          (may OOM)
E002 +FP8        ~900 tok/s    1.3×       1.3×          FP8 quantization
E003 +FlashInfer ~1000 tok/s   1.1×       1.4×          Better backend
E004 +Batch128   ~1300 tok/s   1.3×       1.9×          Larger batch
E005 +CUDA       ~1900 tok/s   1.5×       2.7×          CUDA graphs
E006 +MTP        ~5000 tok/s   2.6×       7.1×          Speculative decoding
E007 -Vision     ~5100 tok/s   1.02×      7.3×          Memory headroom

Total speedup: ~7× vs baseline!
```

### Memory Optimization (GROUP B)

```
Experiment    Batch  Memory    Throughput    vs E007    Notes
──────────────────────────────────────────────────────────────
E007 (base)   128    35 GB     5100 tok/s    1.0×       Baseline
E008 +50%     192    38 GB     6100 tok/s    1.2×       +20%
E009 +100%    256    41 GB     OOM / 7000    1.4×       May OOM
E010 Mem70    128    32 GB     5400 tok/s    1.06×      Better dynamics
E011 Mem80    128    38 GB     4900 tok/s    0.96×      Less stable
```

### Component Contributions

```
Component              Impact       Speedup    Memory    Critical?
─────────────────────────────────────────────────────────────────
FP8 Quantization       Critical     1.3×       -50%      ✓ YES
FlashInfer Backend     Major        1.1×       -5%       ✓ YES
Batch Size (64→128)    Major        1.3×       +20%      ✓ YES
CUDA Graphs            Major        1.5×       +15%      ✓ YES
MTP Speculative        Critical     2.6×       +2%       ✓ YES
Vision Removal         Minor        1.02×      -4%       ○ Optional
Batch (128→192)        Major        1.2×       +10%      ○ If fits
KV Cache Format        Negligible   1.01×      0%        ✗ No
Memory Util Tuning     Minor        1.05×      Varies    ○ Situational

Key takeaway: FP8 + CUDA + MTP = 5× speedup (70% of total)
```

---

## Analysis Guidelines

### For Each Experiment, Record:

**Performance Metrics:**
```yaml
throughput:
  total_tokens_sec: 5000
  tokens_per_sequence: 39.1

latency:
  mean_ms: 25.5
  p50_ms: 24.8
  p95_ms: 32.1
  p99_ms: 41.2

memory:
  peak_allocated_gb: 36.2
  peak_reserved_gb: 38.1
  kv_cache_gb: 9.4
  cuda_graphs_gb: 4.8

gpu_utilization:
  mean_pct: 52.3
  peak_pct: 68.9
  memory_bandwidth_pct: 87.2
```

**Qualitative Observations:**
- Stability (any OOM, crashes, hangs?)
- Warmup time (CUDA graphs take longer)
- Accuracy (any degradation noticed?)
- Error rates (MTP acceptance rate)

### Key Comparisons:

**1. FP8 vs BF16 (E002 vs E015)**
```
Metric           BF16(E015)   FP8(E002)    Delta
──────────────────────────────────────────────────
Throughput       700 tok/s    900 tok/s    +29%
Memory           39 GB        32 GB        -18%
Fits in 40GB?    Barely       ✓ Yes        N/A

Conclusion: FP8 essential for A100 40GB
```

**2. FlashInfer vs Flash Attn 2 (E003 vs E012)**
```
Metric           FA2(E012)    FI(E003)     Delta
──────────────────────────────────────────────────
Throughput       4600 tok/s   5000 tok/s   +9%
Memory           36 GB        35 GB        -3%
Attention time   12 ms        11 ms        -8%

Conclusion: FlashInfer 8-10% faster for Gemma 4
```

**3. MTP Impact (E006 vs E013)**
```
Metric           No MTP(E013) With MTP(E006) Delta
────────────────────────────────────────────────────
Throughput       1900 tok/s   5000 tok/s     +2.6×
Latency          40 ms/tok    25 ms/tok      -38%
Memory           35 GB        36 GB          +3%

Conclusion: MTP provides 2.6× speedup for 0.8 GB cost
```

**4. Batch Size Scaling (E007 vs E008 vs E009)**
```
Batch    Throughput    Per-token    Memory    Fits?
────────────────────────────────────────────────────
128      5100 tok/s    39.8 t/s     35 GB     ✓
192      6100 tok/s    31.8 t/s     38 GB     ? (tight)
256      7000 tok/s    27.3 t/s     41 GB     ✗ (OOM)

Conclusion: Optimal batch = 192 (if fits) or 128 (safe)
```

### Success Criteria:

```
✓ E001 may OOM - acceptable (establishes need for FP8)
✓ E002 must run - FP8 enables model on A100 40GB
✓ E003 should be 8-10% faster than E012 - validates FlashInfer
✓ E005 should be 50% faster than E004 - CUDA graphs work
✓ E006 should be 2-3× faster than E005 - MTP works
✓ E007 should free 1.5 GB - vision removal successful
✓ E008 should fit in memory - optimal batch determination
✓ E009 may OOM - documents maximum batch size
✓ Overall: 5-7× speedup vs baseline

❌ Failure conditions:
- E002 OOMs (FP8 should fit)
- E005 no speedup (CUDA graphs not working)
- E006 no speedup (MTP not working)
- E012 faster than E003 (wrong backend choice)
```

---

## Next Steps After Experiments

1. **Analyze results:**
   ```bash
   python3 analyze_experiments.py --results-dir experiment_results/ablation_study
   ```

2. **Generate visualizations:**
   - Throughput vs experiment (bar chart)
   - Memory usage vs batch size (line chart)
   - Latency distribution (box plot)
   - Cumulative speedup (stacked bar)

3. **Update documentation:**
   - Add results to EXPERIMENT_LOG_002_ABLATION_STUDY.md
   - Update GEMMA4_MOE_OPTIMIZATION_GUIDE.md with validated numbers
   - Document optimal configuration in README

4. **Production configuration:**
   ```bash
   # Based on results, create optimal config
   # Expected: E007 or E008 configuration
   cp experiment_results/ablation_study/E007_config.yaml \
      production_config.yaml
   ```

5. **Report summary:**
   - Key findings (which optimizations matter most)
   - Recommended configuration for A100 40GB
   - Optimal batch size
   - Memory vs throughput trade-offs

---

## Quick Start (TL;DR)

```bash
# 1. Create text-only model
python3 create_text_only_model.py --model_path ... --output_path ...

# 2. Run all experiments (automated)
./run_ablation_study.sh

# 3. Analyze results
python3 analyze_experiments.py --results-dir experiment_results/ablation_study

# 4. View report
cat experiment_results/ablation_study/ablation_study_report.md
```

**Expected time:** 2-3 hours total
**Expected result:** 5-7× speedup, optimal config identified

---

**Document Version:** 1.0
**Status:** Ready for execution
**Next:** Run experiments and document results
