#!/usr/bin/env python3
"""Summarize the paper validation sweeps from the TensorBoard logs of finished runs.

For every run ``logs/WholeBodyTracking/<timestamp>-<method>_seed<N>-locomotion`` the eval
curve of each metric is read from ``events.out.tfevents*`` and reduced to

* ``final10``       mean of the final K evaluations (K=10 -> steps 91k..100k), J_s
* ``best`` / ``best_step``  best evaluation over training (max, or min for failure rates)
* ``last``          value at the final evaluation (100k)
* ``last_over_best``  last / best (retention of the peak at the end of the budget)

and then aggregated over seeds as mean +- std (sample std, ddof=1). Success rate is
``Eval/stop_reason_percent/motion_ends``; evaluations where no episode reached the end
of the motion are logged without that tag and are counted as 0 %.

Outputs (default ``validation/results/``):
  runs.csv            one row per (run, metric)
  summary.csv         one row per (method, metric) with mean/std over seeds
  summary.md          markdown tables per paper group
  curves/<method>.csv full eval curves (step x seed) per metric, for figures

Usage::

    python scripts/summarize_validation_results.py
    python scripts/summarize_validation_results.py --log-root logs/WholeBodyTracking --final-k 10
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN_RE = re.compile(r"^(?P<ts>\d{8}_\d{6})-(?P<method>.+)_seed(?P<seed>\d+)-locomotion$")
EVAL_STEP_TAG = "Eval/num_episodes"

# metric key -> (tensorboard tag, higher_is_better, fill missing eval points with 0)
METRICS = {
    "success": ("Eval/stop_reason_percent/motion_ends", True, True),
    "return": ("Eval/episode_return_mean", True, False),
    "length": ("Eval/episode_length_mean", True, False),
    "bad_tracking": ("Eval/stop_reason_percent/bad_tracking", False, True),
    "timeout": ("Eval/stop_reason_percent/timeout", False, True),
}

# run-name prefix -> (group, label, order)
METHOD_INFO = {
    "g1_29dof_wbt_bc": ("main", "BC", 0),
    "g1_29dof_wbt_iql": ("main", "IQL", 1),
    "g1_29dof_wbt_cql": ("main", "CQL (alpha=5)", 2),
    "g1_29dof_wbt_acl_ql": ("main", "ACL-QL", 3),
    "g1_29dof_wbt_aw_cql_H50": ("main", "AW-CQL (H50)", 4),
    "g1_29dof_wbt_w_bc": ("ablation", "wBC", 0),
    "g1_29dof_wbt_b_arm": ("ablation", "B-arm (LSE - wQ_D)", 1),
    "g1_29dof_wbt_c_arm": ("ablation", "C-arm (wLSE - Q_D)", 2),
    "g1_29dof_wbt_asym_cql": ("ablation", "Asym-CQL", 3),
    "g1_29dof_wbt_aw_cql_H50_global": ("ablation", "AW-CQL global baseline (no phase in w)", 4),
    "g1_29dof_wbt_aw_cql_H50_K10": ("bin_sweep", "AW-CQL (H50, K=10 bins)", 1),
    "g1_29dof_wbt_aw_cql_H50_K40": ("bin_sweep", "AW-CQL (H50, K=40 bins)", 3),
    "g1_29dof_wbt_cql_alpha1p0": ("alpha_sweep", "CQL alpha=1", 0),
    "g1_29dof_wbt_cql_alpha10p0": ("alpha_sweep", "CQL alpha=10", 2),
    "g1_29dof_wbt_aw_cql_H50_alpha1p0": ("alpha_sweep", "AW-CQL alpha=1", 3),
    "g1_29dof_wbt_aw_cql_H50_alpha10p0": ("alpha_sweep", "AW-CQL alpha=10", 5),
    "g1_29dof_wbt_aw_cql_H25": ("h_sweep", "AW-CQL (H25)", 0),
    "g1_29dof_wbt_aw_cql_H100": ("h_sweep", "AW-CQL (H100)", 2),
    "g1_29dof_wbt_lafan_dance1_cql": ("lafan", "LAFAN CQL", 0),
    "g1_29dof_wbt_lafan_dance1_iql": ("lafan", "LAFAN IQL", 1),
    "g1_29dof_wbt_lafan_dance1_acl_ql": ("lafan", "LAFAN ACL-QL", 2),
    "g1_29dof_wbt_lafan_dance1_aw_cql": ("lafan", "LAFAN AW-CQL", 3),
    "g1_29dof_wbt_lafan_dance1_aw_cql_H50_global": ("lafan", "LAFAN AW-CQL global baseline (no phase in w)", 4),
    "g1_29dof_wbt_lafan_kick_bc": ("kick", "KICK BC", 0),
    "g1_29dof_wbt_lafan_kick_cql": ("kick", "KICK CQL", 1),
    "g1_29dof_wbt_lafan_kick_iql": ("kick", "KICK IQL", 2),
    "g1_29dof_wbt_lafan_kick_acl_ql": ("kick", "KICK ACL-QL", 3),
    "g1_29dof_wbt_lafan_kick_aw_cql": ("kick", "KICK AW-CQL", 4),
    "g1_29dof_wbt_lafan_kick_aw_cql_H50_global": ("kick", "KICK AW-CQL global baseline (no phase in w)", 5),
}
# (method, group, label, order, seeds) rows re-used from another table, restricted to `seeds`
SHARED_ROWS = [
    ("g1_29dof_wbt_aw_cql_H50", "h_sweep", "AW-CQL (H50, main arm)", 1, None),
    ("g1_29dof_wbt_cql", "alpha_sweep", "CQL alpha=5 (main)", 1, None),
    ("g1_29dof_wbt_aw_cql_H50", "alpha_sweep", "AW-CQL alpha=5 (main)", 4, None),
    ("g1_29dof_wbt_aw_cql_H50_global", "bin_sweep", "AW-CQL (H50, K=1 = global baseline)", 0, None),
    ("g1_29dof_wbt_aw_cql_H50", "bin_sweep", "AW-CQL (H50, K=20 = main arm)", 2, None),
]

GROUP_TITLES = {
    "main": "Main G1-WBT table (seeds 1-5)",
    "ablation": "Structural ablation (seeds 1-5)",
    "h_sweep": "H robustness (seeds 1-5; H50 is the main AW-CQL arm)",
    "alpha_sweep": "Conservative weight alpha (largebox; alpha=5 is the main table)",
    "bin_sweep": "Progress-bin sensitivity of b(kappa) (seeds 1-5; K=1 is the global baseline, K=20 the main arm)",
    "lafan": "LAFAN dance1 cross-motion (seeds 1-5)",
    "kick": "LAFAN single-kick cross-motion (fightAndSports1_subject4 f1728-1858)",
    "other": "Other runs",
}


@dataclass
class RunResult:
    run_dir: str
    method: str
    seed: int
    steps: list[int]
    curves: dict[str, list[float]]  # metric -> value per eval step


def discover_runs(log_root: Path, final_step: int) -> list[Path]:
    """Return one run dir per (method, seed): the latest one holding the final checkpoint."""
    best: dict[tuple[str, int], tuple[str, Path]] = {}
    for d in sorted(log_root.glob("*-locomotion")):
        m = RUN_RE.match(d.name)
        if not m:
            continue
        if final_step and not (d / f"model_{final_step:07d}.pt").exists():
            print(f"skip (no final checkpoint): {d.name}")
            continue
        key = (m["method"], int(m["seed"]))
        if key not in best or m["ts"] > best[key][0]:
            best[key] = (m["ts"], d)
    return [d for _, d in sorted(best.values())]


def load_run(run_dir: Path) -> RunResult | None:
    events = sorted(run_dir.glob("events.out.tfevents*"))
    if not events:
        print(f"skip (no tfevents): {run_dir.name}")
        return None
    ea = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    if EVAL_STEP_TAG not in tags:
        print(f"skip (no eval scalars): {run_dir.name}")
        return None
    steps = sorted({e.step for e in ea.Scalars(EVAL_STEP_TAG)})
    curves: dict[str, list[float]] = {}
    for key, (tag, _, fill_zero) in METRICS.items():
        values = {e.step: e.value for e in ea.Scalars(tag)} if tag in tags else {}
        if not values and not fill_zero:
            continue
        curves[key] = [values.get(s, 0.0 if fill_zero else math.nan) for s in steps]
    m = RUN_RE.match(run_dir.name)
    assert m
    return RunResult(run_dir.name, m["method"], int(m["seed"]), steps, curves)


def reduce_curve(steps: list[int], values: list[float], higher_is_better: bool, final_k: int) -> dict[str, float]:
    tail = values[-final_k:]
    final = sum(tail) / len(tail)
    pick = max if higher_is_better else min
    best_idx = pick(range(len(values)), key=lambda i: values[i])
    best = values[best_idx]
    last = values[-1]
    ratio = last / best if best != 0 else math.nan
    return {
        "final10": final,
        "best": best,
        "best_step": steps[best_idx],
        "last": last,
        "last_over_best": ratio,
    }


def mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return math.nan, math.nan
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def fmt(m: float, s: float, digits: int = 2) -> str:
    if math.isnan(m):
        return "-"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-root", default="logs/WholeBodyTracking")
    ap.add_argument("--out-dir", default="validation/results")
    ap.add_argument("--final-k", type=int, default=10, help="number of final evaluations averaged for J_s")
    ap.add_argument("--final-step", type=int, default=100000, help="required final checkpoint step (0 = any)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "curves").mkdir(parents=True, exist_ok=True)
    runs = [r for r in (load_run(d) for d in discover_runs(Path(args.log_root), args.final_step)) if r]
    print(f"{len(runs)} runs loaded")

    # ---------------- per-run rows
    run_rows = []
    per_method: dict[str, list[RunResult]] = defaultdict(list)
    for r in runs:
        per_method[r.method].append(r)
        group, label, _ = METHOD_INFO.get(r.method, ("other", r.method, 99))
        for metric, values in r.curves.items():
            red = reduce_curve(r.steps, values, METRICS[metric][1], args.final_k)
            run_rows.append(
                {
                    "run": r.run_dir,
                    "method": r.method,
                    "group": group,
                    "label": label,
                    "seed": r.seed,
                    "metric": metric,
                    **red,
                }
            )
    with (out / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
        w.writeheader()
        w.writerows(run_rows)

    # ---------------- curves per method (step x seed) for figures
    for method, method_runs in per_method.items():
        rs = sorted(method_runs, key=lambda r: r.seed)
        for metric in METRICS:
            if not all(metric in r.curves for r in rs):
                continue
            steps = rs[0].steps
            with (out / "curves" / f"{method}.{metric}.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step"] + [f"seed{r.seed}" for r in rs])
                for i, s in enumerate(steps):
                    w.writerow([s] + [r.curves[metric][i] if i < len(r.curves[metric]) else "" for r in rs])

    # ---------------- aggregate over seeds
    summary_rows = []
    targets = [(method, *METHOD_INFO.get(method, ("other", method, 99)), None) for method in per_method]
    targets += [(m, g, lbl, o, seeds) for (m, g, lbl, o, seeds) in SHARED_ROWS if m in per_method]
    for method, group, label, order, seed_filter in targets:
        rs = [r for r in per_method[method] if seed_filter is None or r.seed in seed_filter]
        for metric in METRICS:
            reds = [
                reduce_curve(r.steps, r.curves[metric], METRICS[metric][1], args.final_k)
                for r in rs
                if metric in r.curves
            ]
            if not reds:
                continue
            row = {
                "method": method,
                "group": group,
                "label": label,
                "order": order,
                "metric": metric,
                "n_seeds": len(reds),
                "seeds": " ".join(str(r.seed) for r in sorted(rs, key=lambda r: r.seed)),
            }
            for key in ("final10", "best", "last", "last_over_best"):
                m, s = mean_std([d[key] for d in reds])
                row[f"{key}_mean"], row[f"{key}_std"] = m, s
            row["best_step_mean"], _ = mean_std([d["best_step"] for d in reds])
            summary_rows.append(row)
    summary_rows.sort(key=lambda r: (list(GROUP_TITLES).index(r["group"]), r["order"], r["metric"]))
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # ---------------- markdown
    lines = [
        f"# Validation summary (final-{args.final_k} mean over the last {args.final_k} evals; mean ± std over seeds)",
        "",
    ]
    for group, title in GROUP_TITLES.items():
        rows = [r for r in summary_rows if r["group"] == group]
        if not rows:
            continue
        lines += [f"## {title}", ""]
        for metric, unit in (("success", "%"), ("return", ""), ("length", "steps")):
            mrows = [r for r in rows if r["metric"] == metric]
            if not mrows:
                continue
            lines += [
                f"**{metric}** ({unit})" if unit else f"**{metric}**",
                "",
                "| Method | seeds | final-10 J_s | best (peak) | best step | last (100k) | last / best |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
            for r in mrows:
                lines.append(
                    f"| {r['label']} | {r['n_seeds']} | {fmt(r['final10_mean'], r['final10_std'])} | "
                    f"{fmt(r['best_mean'], r['best_std'])} | {r['best_step_mean']:.0f} | "
                    f"{fmt(r['last_mean'], r['last_std'])} | {fmt(r['last_over_best_mean'], r['last_over_best_std'])} |"
                )
            lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwritten: {out / 'runs.csv'}, {out / 'summary.csv'}, {out / 'summary.md'}, {out / 'curves'}/")


if __name__ == "__main__":
    main()
