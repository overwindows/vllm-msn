#!/usr/bin/env python3
"""Custom offline throughput driver using `vllm.LLM` directly.

Same engine as `vllm bench throughput`, but lets us use our real prompts
(any length) and capture per-request stats for stdev / percentiles.

Workflow:
  - For each (scenario, max_num_seqs) config, instantiate a fresh LLM.
  - For each rep in 1..reps, call llm.generate(...) on the prompts (one rep at a time).
  - Time the wall clock around generate() to get the run-level throughput.
  - Aggregate per-request output lengths from the RequestOutput list.
  - Append a CSV row, write a per-run JSON, print mean +/- stdev per config.

Key knobs match the vllm-bench-throughput defaults we discussed.
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

OUT_DIR = Path("bench_results")
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "all_runs.csv"

CSV_FIELDS = [
    "ts", "scenario", "num_prompts", "output_len_cap",
    "max_model_len", "max_num_batched_tokens", "gpu_mem_util",
    "max_num_seqs", "rep", "seed",
    "elapsed_time",
    "requests_per_second",
    "prompt_tokens_total", "output_tokens_total", "total_tokens",
    "prompt_tps", "output_tps", "total_tps",
    "out_len_mean", "out_len_stdev", "out_len_p50", "out_len_p90", "out_len_max",
    "finish_stop", "finish_length", "finish_other",
]

SCENARIOS = {
    "sc1": dict(
        dataset="datasets/sc1_delta_v2.jsonl",
        num_prompts=1000,
        output_len=8192,
        max_model_len=24576,
        max_num_batched_tokens=16384,
        gpu_mem_util=0.95,
    ),
    "sc2": dict(
        dataset="datasets/sc2_personal_v2.jsonl",
        num_prompts=500,
        output_len=8192,
        max_model_len=49152,
        max_num_batched_tokens=16384,
        gpu_mem_util=0.95,
    ),
}


def ensure_csv_header():
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv_row(row: dict):
    with CSV_PATH.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def load_prompts(dataset_path: str, n: int) -> list[str]:
    prompts = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompts.append(d["prompt"])
            if len(prompts) >= n:
                break
    return prompts


def render_chat(tok, raw_prompts: list[str]) -> list[str]:
    """Wrap each prompt as a single user-message chat and render via Gemma 4 chat template.
    Mirrors what `vllm bench throughput --dataset-name custom` would do internally.
    """
    out = []
    for p in raw_prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        out.append(text)
    return out


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def run_one_config(*, scenario: str, cfg: dict, max_num_seqs: int, reps: int, model: str):
    """Build a fresh LLM with this max_num_seqs, run `reps` generations.

    Returns the list of per-rep dicts.
    """
    # Defer heavy imports until after we know we'll use them.
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n=== build LLM scenario={scenario} max_num_seqs={max_num_seqs} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    raw_prompts = load_prompts(cfg["dataset"], cfg["num_prompts"])
    prompts = render_chat(tok, raw_prompts)
    print(f"loaded {len(prompts)} prompts from {cfg['dataset']}", flush=True)

    t_engine = time.time()
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=cfg["max_model_len"],
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=cfg["max_num_batched_tokens"],
        gpu_memory_utilization=cfg["gpu_mem_util"],
        seed=0,
        # disable_log_stats=True,  # if available; harmless otherwise
    )
    print(f"engine built in {time.time()-t_engine:.1f} s", flush=True)

    rows = []
    for rep in range(1, reps + 1):
        seed = rep  # different seed per rep
        sampling = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=cfg["output_len"],
            seed=seed,
            ignore_eos=False,
        )
        tag = f"{scenario}_mns{max_num_seqs}_rep{rep}"
        print(f"\n--- RUN {tag} seed={seed} ---", flush=True)
        t0 = time.time()
        outputs = llm.generate(prompts, sampling, use_tqdm=True)
        elapsed = time.time() - t0

        # Aggregate per-request stats
        out_lens = []
        prompt_total = 0
        output_total = 0
        finish_counts = {"stop": 0, "length": 0, "other": 0}
        for o in outputs:
            prompt_total += len(o.prompt_token_ids)
            # Each prompt produced n=1 completion (default)
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
            "scenario": scenario,
            "num_prompts": len(prompts),
            "output_len_cap": cfg["output_len"],
            "max_model_len": cfg["max_model_len"],
            "max_num_batched_tokens": cfg["max_num_batched_tokens"],
            "gpu_mem_util": cfg["gpu_mem_util"],
            "max_num_seqs": max_num_seqs,
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

        # Per-run JSON
        per_run = OUT_DIR / f"{tag}.json"
        with per_run.open("w") as f:
            json.dump({**row, "out_lens": out_lens}, f, indent=2)

        print(
            f"  elapsed={elapsed:.1f}s  req/s={row['requests_per_second']:.3f}  "
            f"out_tok/s={row['output_tps']:.0f}  total_tok/s={row['total_tps']:.0f}  "
            f"out_len(mean+/-sd)={row['out_len_mean']}+/-{row['out_len_stdev']}  "
            f"finish=stop:{finish_counts['stop']}/len:{finish_counts['length']}",
            flush=True,
        )

    # Print config summary
    summarize(scenario, max_num_seqs, rows)

    # Hand the engine to the GC before the next config
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return rows


def summarize(scenario: str, mns: int, rows: list[dict]):
    if not rows:
        print(f"[SUMMARY] {scenario} mns={mns}: no successful runs", flush=True)
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
    ol_m, ol_s = m("out_len_mean")
    n = len(rows)
    print(
        f"\n[SUMMARY] scenario={scenario}  max_num_seqs={mns}  reps={n}\n"
        f"  elapsed_time   : {el_m:.2f} +/- {el_s:.2f} s\n"
        f"  requests/sec   : {r_m:.4f} +/- {r_s:.4f}\n"
        f"  output tokens/s: {o_m:.2f} +/- {o_s:.2f}\n"
        f"  total tokens/s : {t_m:.2f} +/- {t_s:.2f}\n"
        f"  mean out_len   : {ol_m:.1f} +/- {ol_s:.1f}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-num-seqs", required=True,
                    help="Comma list, e.g. 64,128,256,512,1024")
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    args = ap.parse_args()

    ensure_csv_header()
    cfg = SCENARIOS[args.scenario]
    sweep = [int(x) for x in args.max_num_seqs.split(",")]
    for mns in sweep:
        try:
            run_one_config(scenario=args.scenario, cfg=cfg, max_num_seqs=mns,
                           reps=args.reps, model=args.model)
        except Exception as e:
            print(f"!!! config max_num_seqs={mns} failed: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
