#!/usr/bin/env python3
"""Ablation-study benchmark for Gemma 4 26B MoE FP8.

Runs one experiment from a 15-config ablation matrix using vllm.LLM offline.
Replicates the config design from examples/EXPERIMENT_PLAN_ABLATION_STUDY.md
but drives it through the bench_offline.py throughput framework so results
are directly comparable to the sc1/sc2 H100 sweep.

IMPORTANT — environment variables must be set BEFORE this script is imported
(before vllm is imported).  Do NOT call this script directly.  Use
run_ablation.sh which sets VLLM_ATTENTION_BACKEND etc. and then execs this.

Usage (via run_ablation.sh):
    run_ablation.sh E001            # single experiment
    run_ablation.sh --all           # all 15 experiments sequentially
    run_ablation.sh E006 E011 E013  # subset

Direct single-experiment call (env vars must already be set):
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \\
    python3 bench_ablation.py --exp E001 --scenario sc1 --reps 2
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Output paths (sibling to bench_offline.py results)
# ---------------------------------------------------------------------------
OUT_DIR = Path("ablation_results")
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "all_runs.csv"

CSV_FIELDS = [
    "ts", "exp_id", "label", "scenario",
    "quantization", "kv_cache_dtype", "attention_backend",
    "enforce_eager", "mtp", "mtp_k",
    "max_num_seqs", "gpu_memory_utilization", "model_variant",
    "num_prompts", "output_len_cap", "max_model_len", "max_num_batched_tokens",
    "rep", "seed",
    "elapsed_time",
    "requests_per_second",
    "prompt_tokens_total", "output_tokens_total", "total_tokens",
    "prompt_tps", "output_tps", "total_tps",
    "out_len_mean", "out_len_stdev", "out_len_p50", "out_len_p90", "out_len_max",
    "finish_stop", "finish_length", "finish_other",
]

# ---------------------------------------------------------------------------
# Scenario definitions (datasets produced by prep_dataset.py)
# Same as bench_offline.py — keeps results comparable.
# ---------------------------------------------------------------------------
SCENARIOS = {
    "sc1": dict(
        dataset="datasets/sc1_delta_v2.jsonl",
        num_prompts=1000,
        output_len=8192,
        max_model_len=24576,
        max_num_batched_tokens=16384,
    ),
    "sc2": dict(
        dataset="datasets/sc2_personal_v2.jsonl",
        num_prompts=500,
        output_len=8192,
        max_model_len=49152,
        max_num_batched_tokens=16384,
    ),
}

# ---------------------------------------------------------------------------
# Model paths
# Adjust MODEL_BASE and MODEL_TEXT_ONLY to your local checkpoint locations.
# ---------------------------------------------------------------------------
MODEL_BASE = os.environ.get("GEMMA4_MODEL_PATH", "google/gemma-4-26B-A4B-it")
MODEL_TEXT_ONLY = os.environ.get(
    "GEMMA4_TEXT_ONLY_MODEL_PATH",
    MODEL_BASE + "-text-only",   # created by examples/create_text_only_model.py
)
MODEL_ASSISTANT = os.environ.get(
    "GEMMA4_ASSISTANT_MODEL_PATH",
    MODEL_BASE + "-assistant",
)

# ---------------------------------------------------------------------------
# 15-experiment ablation matrix
# Derived from examples/EXPERIMENT_PLAN_ABLATION_STUDY.md (A100 run 2026-05-21)
# and adapted for H100 NVL where noted.
#
# Each entry:
#   label            : short human-readable tag
#   quantization     : None | "fp8"
#   kv_cache_dtype   : "auto" | "fp8_e4m3" | "fp8_e5m2"
#   attention_backend: set via VLLM_ATTENTION_BACKEND (env var before import)
#   enforce_eager    : True = no CUDA graphs; False = CUDA graphs enabled
#   mtp              : True = MTP speculative decoding enabled
#   mtp_k            : number of speculative tokens (ignored if mtp=False)
#   max_num_seqs     : max concurrent sequences
#   gpu_memory_utilization
#   model_variant    : "full" | "text_only"
#
# H100-specific notes vs the A100 run:
#   - VLLM_USE_FLASHINFER_MOE_FP8=1 is now SET for all FlashInfer FP8 runs
#     (Hopper sm_90 has native FP8 tensor cores — the A100 couldn't use this).
#   - E014 (fp8_e4m3 KV cache) is expected to SUCCEED on H100 instead of fail.
#   - E001 (BF16) will likely have a different memory ceiling with 80 GB HBM3.
# ---------------------------------------------------------------------------
EXPERIMENTS: dict[str, dict] = {
    "E001": dict(
        label="baseline (BF16 / FA2, full model, no opts)",
        quantization=None,
        kv_cache_dtype="auto",
        attention_backend="FLASH_ATTN",
        enforce_eager=True,
        mtp=False,   mtp_k=0,
        max_num_seqs=64,
        gpu_memory_utilization=0.95,
        model_variant="full",
    ),
    "E002": dict(
        label="+FP8 weights (FA2)",
        quantization="fp8",
        kv_cache_dtype="auto",
        attention_backend="FLASH_ATTN",
        enforce_eager=True,
        mtp=False,   mtp_k=0,
        max_num_seqs=64,
        gpu_memory_utilization=0.85,
        model_variant="full",
    ),
    "E003": dict(
        label="swap attention backend → FlashInfer (+FP8 MoE on H100)",
        quantization="fp8",
        kv_cache_dtype="auto",
        attention_backend="FLASHINFER",
        enforce_eager=True,
        mtp=False,   mtp_k=0,
        max_num_seqs=64,
        gpu_memory_utilization=0.85,
        model_variant="full",
    ),
    "E004": dict(
        label="+batch 128",
        quantization="fp8",
        kv_cache_dtype="auto",
        attention_backend="FLASHINFER",
        enforce_eager=True,
        mtp=False,   mtp_k=0,
        max_num_seqs=128,
        gpu_memory_utilization=0.85,
        model_variant="full",
    ),
    "E005": dict(
        label="+CUDA graphs (full + piecewise)",
        quantization="fp8",
        kv_cache_dtype="auto",
        attention_backend="FLASHINFER",
        enforce_eager=False,
        mtp=False,   mtp_k=0,
        max_num_seqs=128,
        gpu_memory_utilization=0.75,
        model_variant="full",
    ),
    "E006": dict(
        label="+MTP speculative decoding (k=5)",
        quantization="fp8",
        kv_cache_dtype="auto",
        attention_backend="FLASHINFER",
        enforce_eager=False,
        mtp=True,    mtp_k=5,
        max_num_seqs=128,
        gpu_memory_utilization=0.75,
        model_variant="full",
    ),
    "E007": dict(
        label="swap to text-only model (vision stripped)",
        quantization="fp8",
        kv_cache_dtype="auto",
        attention_backend="FLASHINFER",
        enforce_eager=False,
        mtp=True,    mtp_k=5,
        max_num_seqs=128,
        gpu_memory_utilization=0.75,
        model_variant="text_only",
    ),
    "E008": dict(
        label="batch 192",
        quantization="fp8",
        kv_cache_dtype="auto",
        attention_backend="FLASHINFER",
        enforce_eager=False,
        mtp=True,    mtp_k=5,
        max_num_seqs=192,
        gpu_memory_utilization=0.75,
        model_variant="text_only",
    ),
    "E009": dict(
        label="batch 256",
        quantization="fp8",
        kv_cache_dtype="auto",
        attention_backend="FLASHINFER",
        enforce_eager=False,
        mtp=True,    mtp_k=5,
        max_num_seqs=256,
        gpu_memory_utilization=0.75,
        model_variant="text_only",
    ),
    "E010": dict(
        label="gpu_memory_utilization=0.70",
        quantization="fp8",
        kv_cache_dtype="auto",
        attention_backend="FLASHINFER",
        enforce_eager=False,
        mtp=True,    mtp_k=5,
        max_num_seqs=128,
        gpu_memory_utilization=0.70,
        model_variant="text_only",
    ),
    "E011": dict(
        label="gpu_memory_utilization=0.80  [best on A100]",
        quantization="fp8",
        kv_cache_dtype="auto",
        attention_backend="FLASHINFER",
        enforce_eager=False,
        mtp=True,    mtp_k=5,
        max_num_seqs=128,
        gpu_memory_utilization=0.80,
        model_variant="text_only",
    ),
    "E012": dict(
        label="swap attention back to FA2 at optimal config",
        quantization="fp8",
        kv_cache_dtype="auto",
        attention_backend="FLASH_ATTN",
        enforce_eager=False,
        mtp=True,    mtp_k=5,
        max_num_seqs=128,
        gpu_memory_utilization=0.75,
        model_variant="text_only",
    ),
    "E013": dict(
        label="disable MTP at optimal config (isolates MTP contribution)",
        quantization="fp8",
        kv_cache_dtype="auto",
        attention_backend="FLASHINFER",
        enforce_eager=False,
        mtp=False,   mtp_k=0,
        max_num_seqs=128,
        gpu_memory_utilization=0.75,
        model_variant="text_only",
    ),
    "E014": dict(
        label="FP8 E4M3 KV cache (expected FAIL on A100, should work on H100)",
        quantization="fp8",
        kv_cache_dtype="fp8_e4m3",
        attention_backend="FLASHINFER",
        enforce_eager=False,
        mtp=True,    mtp_k=5,
        max_num_seqs=128,
        gpu_memory_utilization=0.75,
        model_variant="text_only",
    ),
    "E015": dict(
        label="BF16 reference baseline (text-only, no opts)",
        quantization=None,
        kv_cache_dtype="auto",
        attention_backend="FLASHINFER",
        enforce_eager=True,
        mtp=False,   mtp_k=0,
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        model_variant="text_only",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_csv_header():
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv_row(row: dict):
    with CSV_PATH.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def load_prompts(dataset_path: str, n: int) -> list[str]:
    prompts: list[str] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompts.append(d["prompt"])
            if len(prompts) >= n:
                break
    if not prompts:
        raise FileNotFoundError(
            f"Dataset empty or not found: {dataset_path}\n"
            "Run prep_dataset.py first to generate the JSONL datasets."
        )
    return prompts


def render_chat(tok, raw_prompts: list[str]) -> list[str]:
    out = []
    for p in raw_prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        out.append(text)
    return out


def percentile(sorted_vals: list, q: float):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_experiment(
    *,
    exp_id: str,
    exp_cfg: dict,
    scenario: str,
    sc_cfg: dict,
    reps: int,
) -> list[dict]:
    """Build engine for one experiment config, run `reps` generations."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model = MODEL_TEXT_ONLY if exp_cfg["model_variant"] == "text_only" else MODEL_BASE

    print(
        f"\n{'='*70}\n"
        f"  Experiment : {exp_id}  ({exp_cfg['label']})\n"
        f"  Scenario   : {scenario}  ({sc_cfg['num_prompts']} prompts, "
        f"max_model_len={sc_cfg['max_model_len']})\n"
        f"  Model      : {model}\n"
        f"  quantization={exp_cfg['quantization']}  "
        f"kv_cache_dtype={exp_cfg['kv_cache_dtype']}  "
        f"enforce_eager={exp_cfg['enforce_eager']}\n"
        f"  max_num_seqs={exp_cfg['max_num_seqs']}  "
        f"gpu_memory_utilization={exp_cfg['gpu_memory_utilization']}  "
        f"mtp={exp_cfg['mtp']} k={exp_cfg['mtp_k']}\n"
        f"  VLLM_ATTENTION_BACKEND={os.environ.get('VLLM_ATTENTION_BACKEND', 'unset')}\n"
        f"  VLLM_USE_FLASHINFER_MOE_FP8="
        f"{os.environ.get('VLLM_USE_FLASHINFER_MOE_FP8', 'unset')}\n"
        f"{'='*70}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    raw_prompts = load_prompts(sc_cfg["dataset"], sc_cfg["num_prompts"])
    prompts = render_chat(tok, raw_prompts)
    print(f"loaded {len(prompts)} prompts from {sc_cfg['dataset']}", flush=True)

    # Build LLM kwargs
    llm_kwargs: dict = dict(
        model=model,
        trust_remote_code=True,
        max_model_len=sc_cfg["max_model_len"],
        max_num_seqs=exp_cfg["max_num_seqs"],
        max_num_batched_tokens=sc_cfg["max_num_batched_tokens"],
        gpu_memory_utilization=exp_cfg["gpu_memory_utilization"],
        enforce_eager=exp_cfg["enforce_eager"],
        seed=0,
    )
    if exp_cfg["quantization"]:
        llm_kwargs["quantization"] = exp_cfg["quantization"]
    if exp_cfg["kv_cache_dtype"] != "auto":
        llm_kwargs["kv_cache_dtype"] = exp_cfg["kv_cache_dtype"]
    if exp_cfg["mtp"]:
        llm_kwargs["speculative_model"] = MODEL_ASSISTANT
        llm_kwargs["num_speculative_tokens"] = exp_cfg["mtp_k"]

    t_engine = time.time()
    llm = LLM(**llm_kwargs)
    print(f"engine built in {time.time()-t_engine:.1f}s", flush=True)

    rows: list[dict] = []
    for rep in range(1, reps + 1):
        seed = rep
        sampling = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=sc_cfg["output_len"],
            seed=seed,
            ignore_eos=False,
        )
        tag = f"{exp_id}_{scenario}_rep{rep}"
        print(f"\n--- RUN {tag} seed={seed} ---", flush=True)
        t0 = time.time()
        outputs = llm.generate(prompts, sampling, use_tqdm=True)
        elapsed = time.time() - t0

        out_lens: list[int] = []
        prompt_total = 0
        output_total = 0
        finish_counts = {"stop": 0, "length": 0, "other": 0}
        for o in outputs:
            prompt_total += len(o.prompt_token_ids)
            for comp in o.outputs:
                n_out = len(comp.token_ids)
                output_total += n_out
                out_lens.append(n_out)
                fr = (comp.finish_reason or "other").lower()
                if fr == "stop":
                    finish_counts["stop"] += 1
                elif fr == "length":
                    finish_counts["length"] += 1
                else:
                    finish_counts["other"] += 1

        total = prompt_total + output_total
        out_lens_sorted = sorted(out_lens)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "exp_id": exp_id,
            "label": exp_cfg["label"],
            "scenario": scenario,
            "quantization": str(exp_cfg["quantization"]),
            "kv_cache_dtype": exp_cfg["kv_cache_dtype"],
            "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND", "unset"),
            "enforce_eager": exp_cfg["enforce_eager"],
            "mtp": exp_cfg["mtp"],
            "mtp_k": exp_cfg["mtp_k"],
            "max_num_seqs": exp_cfg["max_num_seqs"],
            "gpu_memory_utilization": exp_cfg["gpu_memory_utilization"],
            "model_variant": exp_cfg["model_variant"],
            "num_prompts": len(prompts),
            "output_len_cap": sc_cfg["output_len"],
            "max_model_len": sc_cfg["max_model_len"],
            "max_num_batched_tokens": sc_cfg["max_num_batched_tokens"],
            "rep": rep,
            "seed": seed,
            "elapsed_time": round(elapsed, 3),
            "requests_per_second": round(len(prompts) / elapsed, 4),
            "prompt_tokens_total": prompt_total,
            "output_tokens_total": output_total,
            "total_tokens": total,
            "prompt_tps": round(prompt_total / elapsed, 2),
            "output_tps": round(output_total / elapsed, 2),
            "total_tps": round(total / elapsed, 2),
            "out_len_mean": round(statistics.mean(out_lens), 2) if out_lens else None,
            "out_len_stdev": round(statistics.stdev(out_lens), 2) if len(out_lens) > 1 else 0.0,
            "out_len_p50": int(percentile(out_lens_sorted, 0.5)) if out_lens else None,
            "out_len_p90": int(percentile(out_lens_sorted, 0.9)) if out_lens else None,
            "out_len_max": max(out_lens) if out_lens else None,
            "finish_stop": finish_counts["stop"],
            "finish_length": finish_counts["length"],
            "finish_other": finish_counts["other"],
        }
        append_csv_row(row)
        rows.append(row)

        per_run = OUT_DIR / f"{tag}.json"
        with per_run.open("w") as f:
            json.dump({**row, "out_lens": out_lens}, f, indent=2)

        print(
            f"  elapsed={elapsed:.1f}s  req/s={row['requests_per_second']:.3f}  "
            f"out_tok/s={row['output_tps']:.0f}  total_tok/s={row['total_tps']:.0f}  "
            f"out_len(mean±sd)={row['out_len_mean']}±{row['out_len_stdev']}  "
            f"finish=stop:{finish_counts['stop']}/len:{finish_counts['length']}",
            flush=True,
        )

    _summarize(exp_id, exp_cfg["label"], scenario, rows)

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return rows


def _summarize(exp_id: str, label: str, scenario: str, rows: list[dict]):
    if not rows:
        print(f"[SUMMARY] {exp_id} {scenario}: no successful runs", flush=True)
        return

    def m(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None, None
        if len(vals) == 1:
            return vals[0], 0.0
        return statistics.mean(vals), statistics.stdev(vals)

    el_m, el_s = m("elapsed_time")
    r_m, r_s = m("requests_per_second")
    o_m, o_s = m("output_tps")
    t_m, t_s = m("total_tps")
    ol_m, _ = m("out_len_mean")
    print(
        f"\n[SUMMARY] {exp_id} | {label}\n"
        f"  scenario={scenario}  reps={len(rows)}\n"
        f"  elapsed_time   : {el_m:.2f} ± {el_s:.2f} s\n"
        f"  requests/sec   : {r_m:.4f} ± {r_s:.4f}\n"
        f"  output tokens/s: {o_m:.2f} ± {o_s:.2f}\n"
        f"  total tokens/s : {t_m:.2f} ± {t_s:.2f}\n"
        f"  mean out_len   : {ol_m:.1f}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run one or more ablation experiments via vllm.LLM offline benchmark."
    )
    ap.add_argument(
        "--exp", required=True,
        help="Experiment ID(s), comma-separated. E.g. E001 or E001,E003,E006",
    )
    ap.add_argument(
        "--scenario", default="sc1", choices=list(SCENARIOS.keys()),
        help="Dataset scenario (default: sc1)",
    )
    ap.add_argument("--reps", type=int, default=2,
                    help="Repetitions per experiment (default: 2)")
    ap.add_argument("--list", action="store_true",
                    help="Print the experiment matrix and exit")
    args = ap.parse_args()

    if args.list:
        print(f"{'ID':<6}  {'label'}")
        print("-" * 70)
        for eid, ecfg in EXPERIMENTS.items():
            print(f"{eid:<6}  {ecfg['label']}")
        return 0

    ensure_csv_header()
    sc_cfg = SCENARIOS[args.scenario]
    exp_ids = [x.strip() for x in args.exp.split(",")]

    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            print(f"ERROR: unknown experiment ID '{exp_id}'. "
                  f"Valid: {list(EXPERIMENTS.keys())}", file=sys.stderr)
            return 1
        exp_cfg = EXPERIMENTS[exp_id]

        # Validate that VLLM_ATTENTION_BACKEND matches the experiment's expectation.
        env_backend = os.environ.get("VLLM_ATTENTION_BACKEND", "")
        if env_backend and env_backend != exp_cfg["attention_backend"]:
            print(
                f"WARNING: VLLM_ATTENTION_BACKEND={env_backend} but experiment "
                f"{exp_id} expects {exp_cfg['attention_backend']}.  "
                f"Proceeding with env var (env takes precedence).",
                flush=True,
            )
        elif not env_backend:
            # Set it so the experiment config is explicit in logs.
            os.environ["VLLM_ATTENTION_BACKEND"] = exp_cfg["attention_backend"]

        try:
            run_experiment(
                exp_id=exp_id,
                exp_cfg=exp_cfg,
                scenario=args.scenario,
                sc_cfg=sc_cfg,
                reps=args.reps,
            )
        except Exception as e:
            print(f"!!! experiment {exp_id} FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
