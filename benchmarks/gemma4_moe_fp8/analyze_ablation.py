#!/usr/bin/env python3
"""Summarize ablation_results/all_runs.csv and compare to A100 baselines.

Prints a ranked comparison table and writes ablation_results/summary.md.

Usage:
    python3 analyze_ablation.py [--csv ablation_results/all_runs.csv]
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

# ---------------------------------------------------------------------------
# A100 reference results from examples/EXPERIMENT_PLAN_ABLATION_STUDY.md
# Used to contextualize H100 gains.
# ---------------------------------------------------------------------------
A100_RESULTS: dict[str, dict] = {
    "E001": dict(output_tps=654.9,  label="BF16 baseline (FA2)"),
    "E002": dict(output_tps=692.1,  label="+FP8 weights"),
    "E003": dict(output_tps=696.5,  label="+FlashInfer attn"),
    "E004": dict(output_tps=789.1,  label="+batch 128"),
    "E005": dict(output_tps=768.4,  label="+CUDA graphs"),
    "E006": dict(output_tps=974.6,  label="+MTP k=5"),
    "E007": dict(output_tps=957.3,  label="text-only model"),
    "E008": dict(output_tps=968.4,  label="batch 192"),
    "E009": dict(output_tps=970.9,  label="batch 256"),
    "E010": dict(output_tps=955.1,  label="gpu_mem=0.70"),
    "E011": dict(output_tps=983.7,  label="gpu_mem=0.80 [A100 BEST]"),
    "E012": dict(output_tps=973.6,  label="FA2 at optimal"),
    "E013": dict(output_tps=781.4,  label="no MTP"),
    "E014": dict(output_tps=None,   label="FP8 KV cache — FAIL on A100"),
    "E015": dict(output_tps=477.1,  label="BF16 ref (text, no opts)"),
}


def load_csv(csv_path: Path) -> dict[str, list[dict]]:
    """Load CSV and group rows by (exp_id, scenario)."""
    groups: dict[str, list[dict]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = f"{row['exp_id']}_{row['scenario']}"
            groups.setdefault(key, []).append(row)
    return groups


def summarize_group(rows: list[dict]) -> dict:
    """Compute mean ± stdev for numeric metrics across reps."""
    def nums(field: str) -> list[float]:
        vals = []
        for r in rows:
            v = r.get(field, "")
            if v not in ("", "None", None):
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        return vals

    def stat(field: str):
        vals = nums(field)
        if not vals:
            return None, None
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, stdev

    first = rows[0]
    return {
        "exp_id": first["exp_id"],
        "label": first["label"],
        "scenario": first["scenario"],
        "reps": len(rows),
        "output_tps_mean": stat("output_tps")[0],
        "output_tps_stdev": stat("output_tps")[1],
        "total_tps_mean": stat("total_tps")[0],
        "requests_per_second_mean": stat("requests_per_second")[0],
        "elapsed_mean": stat("elapsed_time")[0],
        "out_len_mean": stat("out_len_mean")[0],
        "quantization": first["quantization"],
        "attention_backend": first["attention_backend"],
        "enforce_eager": first["enforce_eager"],
        "mtp": first["mtp"],
        "mtp_k": first["mtp_k"],
        "max_num_seqs": first["max_num_seqs"],
        "gpu_memory_utilization": first["gpu_memory_utilization"],
        "model_variant": first["model_variant"],
    }


def render_table(summaries: list[dict], scenario: str) -> str:
    # Sort by exp_id
    rows = [s for s in summaries if s["scenario"] == scenario]
    rows.sort(key=lambda r: r["exp_id"])
    if not rows:
        return f"_No data for scenario {scenario}_\n"

    # Find H100 baseline (E001)
    h100_base = next((r["output_tps_mean"] for r in rows if r["exp_id"] == "E001"), None)
    a100_base = A100_RESULTS.get("E001", {}).get("output_tps")

    lines: list[str] = []
    lines.append(f"### Scenario: {scenario}\n")
    lines.append(
        f"| Exp | Label | out tok/s (H100) | ±σ | vs H100 E001 | A100 ref | H100/A100 |"
        f" Backend | eager | MTP | seqs | mem% |"
    )
    lines.append("|-----|-------|:---:|:---:|:---:|:---:|:---:|---------|-------|-----|------|------|")

    for r in rows:
        eid = r["exp_id"]
        otps = r["output_tps_mean"]
        otps_s = r["output_tps_stdev"] or 0.0
        otps_str = f"{otps:.1f}" if otps else "FAIL"
        otps_sig = f"{otps_s:.1f}" if otps else "—"
        # vs H100 baseline
        vs_h100 = f"{otps/h100_base:.3f}×" if (otps and h100_base) else "—"
        # A100 reference
        a100 = A100_RESULTS.get(eid, {})
        a100_tps = a100.get("output_tps")
        a100_str = f"{a100_tps:.1f}" if a100_tps else "FAIL/NA"
        # H100 vs A100
        if otps and a100_tps:
            ratio = otps / a100_tps
            h100_vs_a100 = f"{ratio:.2f}×"
        else:
            h100_vs_a100 = "—"

        backend = r["attention_backend"] or "?"
        eager = "✓" if str(r["enforce_eager"]).lower() in ("true", "1") else "✗ (CG)"
        mtp_str = f"✓ k={r['mtp_k']}" if str(r["mtp"]).lower() in ("true", "1") else "✗"
        seqs = r["max_num_seqs"]
        mem = r["gpu_memory_utilization"]

        lines.append(
            f"| {eid} | {r['label'][:55]} | {otps_str} | {otps_sig} | {vs_h100} "
            f"| {a100_str} | {h100_vs_a100} | {backend} | {eager} | {mtp_str} | {seqs} | {mem} |"
        )

    # Best row
    best = max((r for r in rows if r["output_tps_mean"]), key=lambda r: r["output_tps_mean"], default=None)
    if best:
        lines.append(f"\n**Best H100 result**: {best['exp_id']} — {best['output_tps_mean']:.1f} output tok/s")
        if h100_base:
            lines.append(f"  Overall H100 speedup vs BF16 baseline: {best['output_tps_mean']/h100_base:.3f}×")
        if a100_base and best["output_tps_mean"]:
            lines.append(f"  H100 best vs A100 best (E011={A100_RESULTS['E011']['output_tps']}): "
                         f"{best['output_tps_mean']/A100_RESULTS['E011']['output_tps']:.2f}×")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="ablation_results/all_runs.csv")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run run_ablation.sh first.")
        raise SystemExit(1)

    groups = load_csv(csv_path)
    if not groups:
        print("ERROR: CSV is empty or malformed.")
        raise SystemExit(1)

    summaries = [summarize_group(rows) for rows in groups.values()]
    scenarios = sorted({s["scenario"] for s in summaries})

    md_lines: list[str] = [
        "# Gemma 4 MoE FP8 — Ablation Benchmark Summary",
        "",
        "Generated by `analyze_ablation.py` from `ablation_results/all_runs.csv`.",
        "",
        "## A100 reference (from examples/EXPERIMENT_PLAN_ABLATION_STUDY.md)",
        "",
        "| Exp | A100 out tok/s | label |",
        "|-----|---:|-------|",
    ]
    for eid in sorted(A100_RESULTS.keys()):
        a = A100_RESULTS[eid]
        tps = f"{a['output_tps']:.1f}" if a["output_tps"] else "FAIL"
        md_lines.append(f"| {eid} | {tps} | {a['label']} |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## H100 Results")
    md_lines.append("")

    for sc in scenarios:
        md_lines.append(render_table(summaries, sc))

    # Key deltas summary
    md_lines.append("---")
    md_lines.append("## Key configuration contribution (H100, mean across reps)")
    md_lines.append("")
    md_lines.append("Estimated contribution of each optimization layer, from ablation pairs:")
    md_lines.append("")

    def delta_str(sc: str, eid_on: str, eid_off: str, label: str):
        def get_tps(eid: str) -> float | None:
            key = f"{eid}_{sc}"
            s = next((x for x in summaries if f"{x['exp_id']}_{x['scenario']}" == key), None)
            return s["output_tps_mean"] if s else None

        on_tps = get_tps(eid_on)
        off_tps = get_tps(eid_off)
        if on_tps and off_tps:
            delta = on_tps - off_tps
            pct = (delta / off_tps) * 100
            return f"  {label:<40} : {delta:+.1f} tok/s ({pct:+.1f}%)"
        return f"  {label:<40} : data missing ({eid_on} or {eid_off})"

    for sc in scenarios:
        md_lines.append(f"**{sc}:**")
        md_lines.append(delta_str(sc, "E002", "E001", "FP8 weights vs BF16 (E002-E001)"))
        md_lines.append(delta_str(sc, "E003", "E002", "FlashInfer attn vs FA2 (E003-E002)"))
        md_lines.append(delta_str(sc, "E004", "E003", "batch 128 vs 64 (E004-E003)"))
        md_lines.append(delta_str(sc, "E005", "E004", "CUDA graphs vs eager (E005-E004)"))
        md_lines.append(delta_str(sc, "E006", "E005", "MTP k=5 (E006-E005)"))
        md_lines.append(delta_str(sc, "E007", "E006", "text-only model (E007-E006)"))
        md_lines.append(delta_str(sc, "E011", "E007", "gpu_mem 0.80 vs 0.75 (E011-E007)"))
        md_lines.append(delta_str(sc, "E012", "E011", "FA2 vs FlashInfer at optimal (E012-E011)"))
        md_lines.append(delta_str(sc, "E013", "E011", "disable MTP at optimal (E013-E011) — should be negative"))
        md_lines.append("")

    md_content = "\n".join(md_lines)
    print(md_content)

    out_md = Path("ablation_results/summary.md")
    out_md.parent.mkdir(exist_ok=True)
    out_md.write_text(md_content, encoding="utf-8")
    print(f"\nWrote {out_md}")


if __name__ == "__main__":
    main()
