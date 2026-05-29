# Gemma 4 MoE FP8 — A100 80 GB Sweep Plan

Multi-axis sweep around the **E011 optimum** identified in §4 of the main
ablation. Goal: characterize the response surface for the four engine
knobs that the earlier ablation only probed one-step from baseline.

## Anchor (E011, mean across 2 reps)

| metric | value |
|---|---|
| out tok/s | 1771.5 ± 31.2 |
| req/s     | 1.577 |
| elapsed   | 634 s (1000 prompts) |

Anchor config:

| knob | value |
|---|---|
| `quantization` | `fp8` |
| `enforce_eager` | `False` (CUDA graphs on) |
| `model_variant` | `text_only` |
| `gpu_memory_utilization` | `0.95` |
| `mtp` / `mtp_k` | `True` / `5` |
| `max_num_seqs` (mns) | `128` |
| `max_num_batched_tokens` (mnbt) | `16384` |
| `enable_chunked_prefill` (cp) | vLLM default (V1 → on) |

Scenario: `sc1` (1000 prompts from `sc1_delta_v2.jsonl`,
`output_len_cap=8192`, `max_model_len=24576`).

## Axes

| axis | values | anchor |
|---|---|---|
| `mns`  | 32, 64, 96, **128**, 192, 256, 384 | 128 |
| `mtp_k` (0 = MTP off) | **0**, 3, 4, **5**, 6, 7, 8 | 5 |
| `mnbt` | 4096, 8192, 12288, **16384**, 20480, 24576 | 16384 |
| `cp`   | on (default), off | on |

`mtp_k` doubles as the speculative-token count (vLLM `spec_tokens`).
On the text-only Gemma 4 path the MTP draft model is
`google/gemma-4-26B-A4B-it-assistant`; `mtp_k=0` disables MTP entirely.

## Phases

Phases are cumulative — running Phase 2 covers Phase 1's cells too,
Phase 3 covers Phase 2. `--skip-existing` re-uses CSV rows so phases can
be resumed.

### Phase 0 — anchor confirmation (1 cell, ~25 min)

Re-run E011 under the new harness to verify identical numbers and serve
as a calibration baseline.

### Phase 1 — 1-D sweeps (20 cells, ~8 h)

One axis at a time, holding the other three at the anchor. Captures the
shape of the response surface along each axis individually.

### Phase 2 — 1-D + targeted 2-D interaction slices (36 cells, ~15 h)

Adds three 3 × 3 / 3 × 4 grids most likely to show interactions:

- `mns ∈ {64, 128, 256}` × `mtp_k ∈ {0, 3, 5, 7}` — does optimal batch
  shift when verification compute grows?
- `mns ∈ {64, 128, 256}` × `mnbt ∈ {8192, 16384, 24576}` — larger
  batches need larger prefill token budget.
- `mtp_k ∈ {0, 3, 5, 7}` × `mnbt ∈ {8192, 16384, 24576}` — verification
  needs prefill budget too.

### Phase 3 — full 3-D grid (59 cells, ~25 h)

`mns ∈ {64, 128, 192, 256}` × `mtp_k ∈ {0, 3, 5, 7}` ×
`mnbt ∈ {8192, 16384, 24576}` (with `cp=default`). 48 grid cells, minus
~13 already covered by Phases 1 + 2.

## Hypotheses (so the data is interpretable)

- **mns**: throughput should rise sharply 32 → 128, plateau 128 → 256,
  start to decline at 384 due to KV-cache pressure under
  `gpu_mem=0.95`. The 1-D sweep around 128 was effectively the only
  signal we had from the original ablation (`E007 mns=64`,
  `E008 mns=192`, `E009 mns=256` all within ~5 % of `E006 mns=128`).
- **mtp_k**: classic "draft length sweet spot" — too few wastes the
  verifier, too many causes acceptance-rate collapse. Expected peak
  somewhere in 4–6 on this prefix-heavy workload.
- **mnbt**: should saturate once it ≥ `mns × prefill_chunk`, then become
  irrelevant.
- **cp**: chunked prefill is V1's default and is expected to help only
  when long prompts collide with active decodes. With
  `max_model_len=24576` and a 1000-prompt batch, expect a modest win
  from chunked prefill; the off/on delta isolates that.

## Output

All cells append to
[`ablation_results/all_runs.csv`](ablation_results/all_runs.csv) with
`exp_id` of the form `SW_mns{NNN}_mk{K}_mnbt{N}_cp{0|1|D}` so they
coexist with the original `E001…E016` rows. A new log file
`ablation_results/sweep_<TS>.log` captures the full session.

## Run

```bash
cd benchmarks/gemma4_moe_fp8
chmod +x run_sweep.sh

# Dry-run: list cells, no GPU
./run_sweep.sh --phase 1 --list

# Phase 0 (anchor only): sanity check
./run_sweep.sh --phase 0

# Phase 1 (recommended starting point):
./run_sweep.sh --phase 1

# Resume / extend without re-running completed cells:
./run_sweep.sh --phase 3 --skip-existing
```

Re-run the analyzer after any phase to refresh `summary.md`:

```bash
python3 analyze_ablation.py
```
