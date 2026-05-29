#!/usr/bin/env python3
"""bench_sweep.py — multi-axis sweep around the E011 A100 optimum.

Sweeps four engine knobs around the Gemma 4 26B-A4B MoE FP8 A100 anchor
(E011 from bench_ablation.py):

    max_num_seqs (mns)
    mtp_k                (== num_speculative_tokens for MTP)
    max_num_batched_tokens (mnbt) — chunked-prefill token budget
    enable_chunked_prefill (cp)   — on / off

Anchor (E011):
    FP8 weights + CUDA graphs + MTP k=5 + text-only model
    + gpu_memory_utilization=0.95 + mns=128 + mnbt=16384 + cp=default(on)

Sweep cells are generated programmatically. Each cell gets a stable ID
of the form

    SW_mns{NNN}_mk{K}_mnbt{N}_cp{0|1|D}

where the four fields encode the axis values (D = default, 1 = on, 0 = off).

Results are appended to the same all_runs.csv as bench_ablation.py so the
analyzer can pick them up.

Usage:
    python3 bench_sweep.py --phase 0          # anchor confirmation, 1 cell
    python3 bench_sweep.py --phase 1          # 1-D sweeps (~18 unique cells)
    python3 bench_sweep.py --phase 2          # 1-D + 2-D interaction slices
    python3 bench_sweep.py --phase 3          # full 3-D nms x mtp_k x mnbt grid
    python3 bench_sweep.py --list             # print the cells without running
    python3 bench_sweep.py --only SW_mns128_mk5_mnbt16384_cpD  # one cell

By default --reps 2 and --scenario sc1 (same as the original ablation).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

# Reuse the engine builder + CSV writer from bench_ablation.py.
import bench_ablation as ba

# ---------------------------------------------------------------------------
# Anchor: matches E011 exactly
# ---------------------------------------------------------------------------
ANCHOR = dict(
    label="anchor (E011 settings)",
    dtype="bfloat16",
    quantization="fp8",
    kv_cache_dtype="auto",
    enforce_eager=False,
    mtp=True,
    mtp_k=5,
    max_num_seqs=128,
    gpu_memory_utilization=0.95,
    model_variant="text_only",
    max_num_batched_tokens=16384,
    enable_chunked_prefill=None,  # None = vLLM default (V1: on)
)

# ---------------------------------------------------------------------------
# Sweep axis values
# ---------------------------------------------------------------------------
AXES_1D = {
    "mns":  [32, 64, 96, 128, 192, 256, 384],
    "mk":   [0, 3, 4, 5, 6, 7, 8],          # 0 == MTP off
    "mnbt": [4096, 8192, 12288, 16384, 20480, 24576],
    "cp":   [True, False],                  # chunked prefill on/off
}

# 2-D interaction slices: each is a list of (axis_a, axis_b) pairs and the
# coarse value lists to cross.
AXES_2D = [
    ("mns",  [64, 128, 256],       "mk",   [0, 3, 5, 7]),
    ("mns",  [64, 128, 256],       "mnbt", [8192, 16384, 24576]),
    ("mk",   [0, 3, 5, 7],         "mnbt", [8192, 16384, 24576]),
]

# 3-D full grid (Phase 3)
AXES_3D = (
    [64, 128, 192, 256],   # mns
    [0, 3, 5, 7],          # mk
    [8192, 16384, 24576],  # mnbt
)


# ---------------------------------------------------------------------------
# Cell construction
# ---------------------------------------------------------------------------
def _cp_tag(cp: bool | None) -> str:
    if cp is None:
        return "D"
    return "1" if cp else "0"


def _cell_id(mns: int, mk: int, mnbt: int, cp: bool | None) -> str:
    return f"SW_mns{mns:03d}_mk{mk}_mnbt{mnbt}_cp{_cp_tag(cp)}"


def _make_cell(*, mns: int, mk: int, mnbt: int, cp: bool | None) -> tuple[str, dict]:
    cfg = dict(ANCHOR)
    cfg["max_num_seqs"] = mns
    cfg["mtp_k"] = mk
    cfg["mtp"] = mk > 0
    cfg["max_num_batched_tokens"] = mnbt
    cfg["enable_chunked_prefill"] = cp
    eid = _cell_id(mns, mk, mnbt, cp)
    cfg["label"] = (
        f"sweep mns={mns} mk={mk} mnbt={mnbt} cp={_cp_tag(cp)}"
    )
    return eid, cfg


def anchor_cell() -> tuple[str, dict]:
    return _make_cell(
        mns=ANCHOR["max_num_seqs"],
        mk=ANCHOR["mtp_k"],
        mnbt=ANCHOR["max_num_batched_tokens"],
        cp=ANCHOR["enable_chunked_prefill"],
    )


def cells_1d() -> list[tuple[str, dict]]:
    base = dict(
        mns=ANCHOR["max_num_seqs"],
        mk=ANCHOR["mtp_k"],
        mnbt=ANCHOR["max_num_batched_tokens"],
        cp=ANCHOR["enable_chunked_prefill"],
    )
    out: list[tuple[str, dict]] = [anchor_cell()]
    for axis, values in AXES_1D.items():
        for v in values:
            params = dict(base)
            params[axis] = v
            out.append(_make_cell(**params))
    return _dedup(out)


def cells_2d() -> list[tuple[str, dict]]:
    base = dict(
        mns=ANCHOR["max_num_seqs"],
        mk=ANCHOR["mtp_k"],
        mnbt=ANCHOR["max_num_batched_tokens"],
        cp=ANCHOR["enable_chunked_prefill"],
    )
    out: list[tuple[str, dict]] = []
    for ax_a, vals_a, ax_b, vals_b in AXES_2D:
        for va in vals_a:
            for vb in vals_b:
                params = dict(base)
                params[ax_a] = va
                params[ax_b] = vb
                out.append(_make_cell(**params))
    return _dedup(out)


def cells_3d() -> list[tuple[str, dict]]:
    mns_vals, mk_vals, mnbt_vals = AXES_3D
    out: list[tuple[str, dict]] = []
    for mns in mns_vals:
        for mk in mk_vals:
            for mnbt in mnbt_vals:
                out.append(
                    _make_cell(
                        mns=mns, mk=mk, mnbt=mnbt,
                        cp=ANCHOR["enable_chunked_prefill"],
                    )
                )
    return _dedup(out)


def _dedup(cells: Iterable[tuple[str, dict]]) -> list[tuple[str, dict]]:
    seen: dict[str, dict] = {}
    for eid, cfg in cells:
        if eid not in seen:
            seen[eid] = cfg
    return list(seen.items())


def cells_for_phase(phase: int) -> list[tuple[str, dict]]:
    if phase == 0:
        return [anchor_cell()]
    if phase == 1:
        return cells_1d()
    if phase == 2:
        return _dedup(cells_1d() + cells_2d())
    if phase == 3:
        return _dedup(cells_1d() + cells_2d() + cells_3d())
    raise ValueError(f"phase must be 0..3, got {phase}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", type=int, default=1,
                    help="0=anchor only, 1=1D, 2=1D+2D, 3=full")
    ap.add_argument("--scenario", default="sc1",
                    choices=list(ba.SCENARIOS.keys()))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--list", action="store_true",
                    help="Print cells and exit (no runs)")
    ap.add_argument("--only",
                    help="Comma-separated cell IDs to run (overrides --phase)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip cells already present in all_runs.csv with "
                         "rep count >= --reps")
    args = ap.parse_args()

    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        cells = [c for c in cells_for_phase(3) if c[0] in wanted]
        missing = wanted - {c[0] for c in cells}
        if missing:
            print(f"WARN: unknown cell IDs: {sorted(missing)}", file=sys.stderr)
    else:
        cells = cells_for_phase(args.phase)

    if args.skip_existing:
        cells = _filter_existing(cells, scenario=args.scenario, reps=args.reps)

    print(f"\nSweep: phase={args.phase} scenario={args.scenario} reps={args.reps}")
    print(f"Cells to run: {len(cells)}")
    for eid, cfg in cells:
        print(f"  {eid}  {cfg['label']}")
    if args.list:
        return 0

    ba.ensure_csv_header()
    sc_cfg = ba.SCENARIOS[args.scenario]

    for i, (eid, cfg) in enumerate(cells, 1):
        env_backend = os.environ.get("VLLM_ATTENTION_BACKEND", "unset")
        print(
            f"\n### Cell {i}/{len(cells)}: {eid}\n"
            f"    VLLM_ATTENTION_BACKEND={env_backend} "
            f"(Gemma 4 forces TRITON_ATTN — env var is no-op)",
            flush=True,
        )
        try:
            ba.run_experiment(
                exp_id=eid,
                exp_cfg=cfg,
                scenario=args.scenario,
                sc_cfg=sc_cfg,
                reps=args.reps,
            )
        except Exception as e:
            print(f"!!! cell {eid} FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("\nSweep done.", flush=True)
    return 0


def _filter_existing(
    cells: list[tuple[str, dict]], *, scenario: str, reps: int,
) -> list[tuple[str, dict]]:
    """Drop cells whose all_runs.csv rows already cover the requested reps."""
    import csv
    from collections import Counter
    if not ba.CSV_PATH.exists():
        return cells
    counts: Counter = Counter()
    with ba.CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            if row.get("scenario") == scenario:
                counts[row["exp_id"]] += 1
    keep: list[tuple[str, dict]] = []
    for eid, cfg in cells:
        if counts[eid] >= reps:
            print(f"  skip {eid} (already has {counts[eid]} reps)")
            continue
        keep.append((eid, cfg))
    return keep


if __name__ == "__main__":
    sys.exit(main())
