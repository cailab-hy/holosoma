#!/usr/bin/env python3
"""Replay failure hazard vs. evaluation failure-phase distribution ("wall alignment").

(a) Replay hazard from the offline dataset (scripts/analyze_h5_phase_failure_hazard.py)::

        h(k) = P(bad-tracking termination | episode reached phase bin k)

(b) Evaluation failure-phase distribution of a trained policy, read from the TensorBoard
    logs written during training (``Bad_tracking/phase_<lo>_<hi>``: share of the 4096
    evaluation episodes' bad-tracking terminations that happened in each phase bin)::

        p_fail(k) = N_{termination at k} / N_{failures}

    For every method and seed two checkpoints are compared: ``peak`` (evaluation step with
    the highest success rate) and ``final`` (last evaluation, 100k).

Alignment metrics per (method, checkpoint, seed): mass of p_fail on the hazard wall bins,
Spearman rank correlation between h and p_fail, and the Jensen-Shannon distance between
p_fail and the normalized hazard. Seeds are aggregated as mean +- std.

Outputs (``--out-dir``): hazard.csv, p_fail_runs.csv, summary.csv, summary.md and
phase_failure_alignment.png (+ one PNG per method).

Usage::

    python scripts/phase_failure_alignment.py                      # default methods, main dataset
    python scripts/phase_failure_alignment.py --methods g1_29dof_wbt_cql g1_29dof_wbt_aw_cql_H50
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from analyze_h5_phase_failure_hazard import analyze_phase_failure_hazard  # noqa: E402

RUN_RE = re.compile(r"^(?P<ts>\d{8}_\d{6})-(?P<method>.+)_seed(?P<seed>\d+)-locomotion$")
PHASE_TAG_RE = re.compile(r"^Bad_tracking/phase_(?P<lo>[0-9.]+)_(?P<hi>[0-9.]+)$")
SUCCESS_TAG = "Eval/stop_reason_percent/motion_ends"
BAD_TAG = "Eval/stop_reason_percent/bad_tracking"
EPISODES_TAG = "Eval/num_episodes"

LABELS = {
    "g1_29dof_wbt_cql": "CQL (alpha=5)",
    "g1_29dof_wbt_iql": "IQL",
    "g1_29dof_wbt_bc": "BC",
    "g1_29dof_wbt_acl_ql": "ACL-QL",
    "g1_29dof_wbt_aw_cql_H50": "AW-CQL (H50)",
    "g1_29dof_wbt_aw_cql_H25": "AW-CQL (H25)",
    "g1_29dof_wbt_aw_cql_H100": "AW-CQL (H100)",
    "g1_29dof_wbt_w_bc": "wBC",
    "g1_29dof_wbt_b_arm": "B-arm",
    "g1_29dof_wbt_c_arm": "C-arm",
    "g1_29dof_wbt_asym_cql": "Asym-CQL",
    "g1_29dof_wbt_aw_cql_H50_global": "AW-CQL global baseline",
    "g1_29dof_wbt_lafan_dance1_cql": "LAFAN CQL",
    "g1_29dof_wbt_lafan_dance1_iql": "LAFAN IQL",
    "g1_29dof_wbt_lafan_dance1_aw_cql": "LAFAN AW-CQL",
    "g1_29dof_wbt_lafan_dance1_aw_cql_H50_global": "LAFAN AW-CQL global baseline",
    "g1_29dof_wbt_lafan_dance1_acl_ql": "LAFAN ACL-QL",
}
DEFAULT_METHODS = [
    "g1_29dof_wbt_cql",
    "g1_29dof_wbt_aw_cql_H50",
    "g1_29dof_wbt_iql",
    "g1_29dof_wbt_b_arm",
]


# ------------------------------------------------------------------ hazard (dataset side)
def load_hazard(h5: Path, num_bins: int, cache: Path | None) -> dict:
    if cache is not None and cache.exists():
        data = json.loads(cache.read_text())
        if data.get("h5") == str(h5) and data.get("num_bins") == num_bins:
            return data
    result = analyze_phase_failure_hazard(h5, num_bins=num_bins)
    data = {
        "h5": str(h5),
        "num_bins": num_bins,
        "edges": [float(x) for x in result.phase_edges],
        "entered": [int(x) for x in result.entered_episodes],
        "failures": [int(x) for x in result.bad_tracking_failures],
        "hazard": [None if math.isnan(float(x)) else float(x) for x in result.hazard],
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, indent=2))
    return data


# ------------------------------------------------------------------ evaluation (TB side)
@dataclass
class EvalDistribution:
    method: str
    seed: int
    checkpoint: str  # "peak" | "final"
    step: int
    success: float  # %
    bad_tracking: float  # %
    num_episodes: int
    fractions: np.ndarray  # [num_bins], sums to 1 (0 if no failures)

    @property
    def num_failures(self) -> int:
        return round(self.num_episodes * self.bad_tracking / 100.0)

    @property
    def counts(self) -> np.ndarray:
        return np.round(self.fractions * self.num_failures)

    @property
    def entered(self) -> np.ndarray:
        """Episodes that entered each bin. Evaluation episodes all start at phase 0 (MotionCommand
        zeroes the start phase when is_evaluating) and motion-end terminations only happen in the
        last bin, so entered(k) = N - sum_{j<k} bad-tracking terminations(j)."""
        return self.num_episodes - np.concatenate([[0.0], np.cumsum(self.counts)[:-1]])

    @property
    def policy_hazard(self) -> np.ndarray:
        """h_pi(k) = bad-tracking terminations at k / episodes that entered k."""
        entered = self.entered
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(entered > 0, self.counts / entered, np.nan)


def peak_and_final_steps(steps: list[int], success: dict[int, float]) -> tuple[int, int]:
    """Peak = eval step with the highest success (earliest on ties); final = last eval step."""
    return max(steps, key=lambda s: (success.get(s, 0.0), -s)), steps[-1]


def discover_runs(log_root: Path, methods: list[str], final_step: int) -> dict[str, list[Path]]:
    best: dict[tuple[str, int], tuple[str, Path]] = {}
    for d in sorted(log_root.glob("*-locomotion")):
        m = RUN_RE.match(d.name)
        if not m or m["method"] not in methods:
            continue
        if final_step and not (d / f"model_{final_step:07d}.pt").exists():
            continue
        key = (m["method"], int(m["seed"]))
        if key not in best or m["ts"] > best[key][0]:
            best[key] = (m["ts"], d)
    runs: dict[str, list[Path]] = defaultdict(list)
    for (method, _), (_, d) in sorted(best.items()):
        runs[method].append(d)
    return runs


def read_run(run_dir: Path, num_bins: int, edges: np.ndarray) -> list[EvalDistribution]:
    events = sorted(run_dir.glob("events.out.tfevents*"))
    ea = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    m = RUN_RE.match(run_dir.name)
    assert m
    steps = sorted({e.step for e in ea.Scalars(EPISODES_TAG)})
    episodes = {e.step: int(e.value) for e in ea.Scalars(EPISODES_TAG)}
    success = {e.step: e.value for e in ea.Scalars(SUCCESS_TAG)} if SUCCESS_TAG in tags else {}
    bad = {e.step: e.value for e in ea.Scalars(BAD_TAG)} if BAD_TAG in tags else {}

    # phase tags -> bin index by matching the lower edge
    bin_series: dict[int, dict[int, float]] = {}
    for tag in tags:
        pm = PHASE_TAG_RE.match(tag)
        if not pm:
            continue
        lo = float(pm["lo"])
        idx = np.argmin(np.abs(edges[:-1] - lo)).item()
        bin_series[idx] = {e.step: e.value for e in ea.Scalars(tag)}
    if len(bin_series) != num_bins:
        print(f"warning: {run_dir.name} has {len(bin_series)} phase bins, expected {num_bins}")

    peak_step, final_step = peak_and_final_steps(steps, success)
    out = []
    for name, step in (("peak", peak_step), ("final", final_step)):
        frac = np.array([bin_series.get(k, {}).get(step, 0.0) / 100.0 for k in range(num_bins)])
        total = frac.sum()
        if total > 0:
            frac = frac / total
        out.append(
            EvalDistribution(
                method=m["method"],
                seed=int(m["seed"]),
                checkpoint=name,
                step=step,
                success=success.get(step, 0.0),
                bad_tracking=bad.get(step, 0.0),
                num_episodes=episodes.get(step, 0),
                fractions=frac,
            )
        )
    return out


# ------------------------------------------------------------------ metrics
def alignment_metrics(frac: np.ndarray, hazard: np.ndarray, wall_bins: list[int]) -> dict[str, float]:
    h = np.nan_to_num(hazard, nan=0.0)
    h_norm = h / h.sum() if h.sum() > 0 else h
    valid = frac.sum() > 0
    rho = float(spearmanr(h, frac).correlation) if valid and np.std(frac) > 0 else math.nan
    js = float(jensenshannon(frac, h_norm, base=2)) if valid else math.nan
    return {
        "wall_mass": float(frac[wall_bins].sum()) if valid else math.nan,
        "spearman": rho,
        "js_distance": js,
        "top_bin": int(np.argmax(frac)) if valid else -1,
        "top_bin_mass": float(frac.max()) if valid else math.nan,
    }


def mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return math.nan, math.nan
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def fmt(m: float, s: float, digits: int = 2) -> str:
    return "-" if math.isnan(m) else f"{m:.{digits}f} ± {s:.{digits}f}"


# ------------------------------------------------------------------ plotting
def plot(hazard: dict, dists: list[EvalDistribution], methods: list[str], wall_bins: list[int], out: Path) -> None:
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    num_bins = hazard["num_bins"]
    edges = np.array(hazard["edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    width = (edges[1] - edges[0]) * 0.8
    h = np.array([np.nan if v is None else v for v in hazard["hazard"]])

    def shade_walls(ax):
        for k in wall_bins:
            ax.axvspan(edges[k], edges[k + 1], color="tab:red", alpha=0.08, lw=0)

    def panel_pfail(ax, method):
        for chk, color, offset in (("peak", "tab:blue", -0.5), ("final", "tab:orange", 0.5)):
            rows = [d for d in dists if d.method == method and d.checkpoint == chk]
            if not rows:
                continue
            stack = np.stack([d.fractions for d in rows]) * 100.0
            mean, std = stack.mean(0), stack.std(0, ddof=1) if len(rows) > 1 else np.zeros(num_bins)
            succ_m, succ_s = mean_std([d.success for d in rows])
            step_m, _ = mean_std([float(d.step) for d in rows])
            ax.bar(
                centers + offset * width / 2,
                mean,
                width / 2,
                yerr=std,
                color=color,
                alpha=0.85,
                capsize=2,
                label=f"{chk} (step ~{step_m / 1000:.0f}k, success {succ_m:.1f}±{succ_s:.1f}%, n={len(rows)})",
            )
        shade_walls(ax)
        ax.set_ylabel("p_fail(k) [%]")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_title(LABELS.get(method, method), loc="left", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    fig, axes = plt.subplots(1 + len(methods), 1, figsize=(10, 2.6 * (1 + len(methods))), sharex=True)
    ax0 = axes[0]
    ax0.bar(centers, 100.0 * np.nan_to_num(h), width, color="tab:gray")
    shade_walls(ax0)
    ax0.set_ylabel("hazard h(k) [%]")
    ax0.set_title(f"Replay failure hazard ({Path(hazard['h5']).name}); wall bins {wall_bins}", loc="left", fontsize=10)
    for ax, method in zip(axes[1:], methods):
        panel_pfail(ax, method)
    axes[-1].set_xlabel("motion phase")
    axes[-1].set_xticks(edges[:-1])
    axes[-1].set_xticklabels([str(k) for k in range(num_bins)])
    axes[-1].set_xlabel("phase bin k (20 bins)")
    fig.tight_layout()
    fig.savefig(out / "phase_failure_alignment.png", dpi=150)
    plt.close(fig)

    for method in methods:
        fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
        axes[0].bar(centers, 100.0 * np.nan_to_num(h), width, color="tab:gray")
        shade_walls(axes[0])
        axes[0].set_ylabel("hazard h(k) [%]")
        axes[0].set_title("Replay failure hazard", loc="left", fontsize=10)
        panel_pfail(axes[1], method)
        axes[1].set_xticks(edges[:-1])
        axes[1].set_xticklabels([str(k) for k in range(num_bins)])
        axes[1].set_xlabel("phase bin k")
        fig.tight_layout()
        fig.savefig(out / f"phase_failure_alignment_{method}.png", dpi=150)
        plt.close(fig)


def plot_policy_hazard(
    hazard: dict, dists: list[EvalDistribution], methods: list[str], wall_bins: list[int], out: Path
) -> None:
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    num_bins = hazard["num_bins"]
    edges = np.array(hazard["edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    width = (edges[1] - edges[0]) * 0.8
    h = 100 * np.nan_to_num(np.array([np.nan if v is None else v for v in hazard["hazard"]]))
    fig, axes = plt.subplots(len(methods), 1, figsize=(10, 2.8 * len(methods)), sharex=True, squeeze=False)
    for ax, method in zip(axes[:, 0], methods):
        for k in wall_bins:
            ax.axvspan(edges[k], edges[k + 1], color="tab:red", alpha=0.08, lw=0)
        ax.bar(centers - width / 3, h, width / 3, color="tab:gray", label="replay hazard h(k)")
        for chk, color, offset in (("peak", "tab:blue", 0.0), ("final", "tab:orange", width / 3)):
            rows = [d for d in dists if d.method == method and d.checkpoint == chk]
            if not rows:
                continue
            stack = np.stack([np.nan_to_num(d.policy_hazard) for d in rows]) * 100.0
            mean = stack.mean(0)
            std = stack.std(0, ddof=1) if len(rows) > 1 else np.zeros(num_bins)
            succ_m, succ_s = mean_std([d.success for d in rows])
            ax.bar(
                centers + offset,
                mean,
                width / 3,
                yerr=std,
                capsize=2,
                color=color,
                alpha=0.85,
                label=f"policy hazard h_pi(k), {chk} (success {succ_m:.1f}±{succ_s:.1f}%, n={len(rows)})",
            )
        ax.set_ylabel("%")
        ax.set_title(LABELS.get(method, method), loc="left", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    axes[-1, 0].set_xticks(edges[:-1])
    axes[-1, 0].set_xticklabels([str(k) for k in range(num_bins)])
    axes[-1, 0].set_xlabel("phase bin k")
    fig.tight_layout()
    fig.savefig(out / "policy_hazard.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default="offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5")
    ap.add_argument("--hazard-cache", default=None, help="json cache for the hazard (default: <out-dir>/hazard.json)")
    ap.add_argument("--log-root", default="logs/WholeBodyTracking")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--num-bins", type=int, default=20)
    ap.add_argument("--wall-bins", nargs="+", type=int, default=[4, 5, 12, 13])
    ap.add_argument("--final-step", type=int, default=100000)
    ap.add_argument("--out-dir", default="validation/results/phase_alignment")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.hazard_cache) if args.hazard_cache else out / "hazard.json"

    print(f"[hazard] {args.h5}")
    hazard = load_hazard(Path(args.h5), args.num_bins, cache)
    edges = np.array(hazard["edges"])
    h = np.array([np.nan if v is None else v for v in hazard["hazard"]])
    with (out / "hazard.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "phase_lo", "phase_hi", "entered", "bad_failures", "hazard", "is_wall"])
        for k in range(args.num_bins):
            w.writerow(
                [
                    k,
                    edges[k],
                    edges[k + 1],
                    hazard["entered"][k],
                    hazard["failures"][k],
                    hazard["hazard"][k],
                    int(k in args.wall_bins),
                ]
            )
    top = np.argsort(-np.nan_to_num(h))[:6]
    print("[hazard] top bins:", ", ".join(f"k={k}: {100 * h[k]:.1f}%" for k in top))

    runs = discover_runs(Path(args.log_root), args.methods, args.final_step)
    methods = [m for m in args.methods if runs.get(m)]
    for m in args.methods:
        if m not in methods:
            print(f"warning: no finished runs for {m}")
    dists: list[EvalDistribution] = []
    for method in methods:
        for d in runs[method]:
            dists += read_run(d, args.num_bins, edges)
        print(f"[eval] {method}: {len(runs[method])} seeds")

    # per-run rows
    with (out / "p_fail_runs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "checkpoint",
                "seed",
                "step",
                "success_pct",
                "n_failures",
                "bin",
                "phase_lo",
                "phase_hi",
                "fraction",
                "count",
                "entered",
                "policy_hazard",
            ]
        )
        for d in dists:
            hp = d.policy_hazard
            for k in range(args.num_bins):
                w.writerow(
                    [
                        d.method,
                        d.checkpoint,
                        d.seed,
                        d.step,
                        d.success,
                        d.num_failures,
                        k,
                        edges[k],
                        edges[k + 1],
                        d.fractions[k],
                        int(d.counts[k]),
                        int(d.entered[k]),
                        hp[k],
                    ]
                )

    # summary
    summary = []
    for method in methods:
        for chk in ("peak", "final"):
            rows = [d for d in dists if d.method == method and d.checkpoint == chk]
            mets = [alignment_metrics(d.fractions, h, args.wall_bins) for d in rows]
            row = {
                "method": method,
                "label": LABELS.get(method, method),
                "checkpoint": chk,
                "n_seeds": len(rows),
                "seeds": " ".join(str(d.seed) for d in rows),
                "step_mean": statistics.fmean(d.step for d in rows),
            }
            for key, vals in (
                ("success", [d.success for d in rows]),
                ("n_failures", [float(d.num_failures) for d in rows]),
                ("wall_mass", [x["wall_mass"] for x in mets]),
                ("spearman", [x["spearman"] for x in mets]),
                ("js_distance", [x["js_distance"] for x in mets]),
                ("top_bin_mass", [x["top_bin_mass"] for x in mets]),
            ):
                row[f"{key}_mean"], row[f"{key}_std"] = mean_std(vals)
            top_bins = [x["top_bin"] for x in mets]
            row["top_bin_mode"] = max(set(top_bins), key=top_bins.count) if top_bins else -1
            for k in range(args.num_bins):
                row[f"pfail_bin{k:02d}_mean"], row[f"pfail_bin{k:02d}_std"] = mean_std(
                    [float(d.fractions[k]) for d in rows]
                )
                row[f"hpi_bin{k:02d}_mean"], row[f"hpi_bin{k:02d}_std"] = mean_std(
                    [float(d.policy_hazard[k]) for d in rows]
                )
            hpi_means = np.array([row[f"hpi_bin{k:02d}_mean"] for k in range(args.num_bins)], dtype=float)
            row["hpi_max_bin"] = int(np.nanargmax(hpi_means)) if np.isfinite(hpi_means).any() else -1
            row["hpi_max"] = float(np.nanmax(hpi_means)) if np.isfinite(hpi_means).any() else math.nan
            summary.append(row)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    lines = [
        f"# Replay hazard wall vs. evaluation failure wall (wall bins {args.wall_bins}, {args.num_bins} bins)",
        "",
        "hazard h(k) top bins: " + ", ".join(f"k={k}: {100 * h[k]:.1f}%" for k in top),
        "",
        "| Method | ckpt | seeds | step | success % | failures | wall mass (p_fail on wall bins) | Spearman(h, p_fail) | JS dist | mode bin |",  # noqa: E501
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary:
        lines.append(  # noqa: PERF401
            f"| {r['label']} | {r['checkpoint']} | {r['n_seeds']} | {r['step_mean'] / 1000:.0f}k | {fmt(r['success_mean'], r['success_std'], 1)} | "  # noqa: E501
            f"{r['n_failures_mean']:.0f} | {fmt(100 * r['wall_mass_mean'], 100 * r['wall_mass_std'], 1)} % | "
            f"{fmt(r['spearman_mean'], r['spearman_std'])} | {fmt(r['js_distance_mean'], r['js_distance_std'])} | {r['top_bin_mode']} |"  # noqa: E501
        )
    lines += [
        "",
        "Per-bin p_fail (%), mean over seeds:",
        "",
        "| Method | ckpt | " + " | ".join(str(k) for k in range(args.num_bins)) + " |",
        "| --- | --- | " + " | ".join("---" for _ in range(args.num_bins)) + " |",
    ]
    lines.append(
        "| hazard h(k) | dataset | " + " | ".join(f"{100 * v:.0f}" if not math.isnan(v) else "-" for v in h) + " |"
    )
    for r in summary:
        lines.append(
            f"| {r['label']} | {r['checkpoint']} | "
            + " | ".join(f"{100 * r[f'pfail_bin{k:02d}_mean']:.0f}" for k in range(args.num_bins))
            + " |"
        )
    lines += [
        "",
        "Policy hazard h_pi(k) = bad-tracking terminations at k / episodes entering k (%), mean over seeds:",
        "",
        "| Method | ckpt | " + " | ".join(str(k) for k in range(args.num_bins)) + " | max bin |",
        "| --- | --- | " + " | ".join("---" for _ in range(args.num_bins)) + " | --- |",
    ]
    h_cells = " | ".join(f"{100 * v:.0f}" if not math.isnan(v) else "-" for v in h)
    lines.append(f"| hazard h(k) | dataset | {h_cells} | {int(np.nanargmax(np.nan_to_num(h, nan=-1.0)))} |")
    for r in summary:
        cells = []
        for k in range(args.num_bins):
            v = r[f"hpi_bin{k:02d}_mean"]
            cells.append("-" if math.isnan(v) else f"{100 * v:.0f}")
        lines.append(
            f"| {r['label']} | {r['checkpoint']} | "
            + " | ".join(cells)
            + f" | {r['hpi_max_bin']} ({100 * r['hpi_max']:.0f}%) |"
        )
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    plot(hazard, dists, methods, args.wall_bins, out)
    plot_policy_hazard(hazard, dists, methods, args.wall_bins, out)
    print(f"\nwritten: {out}/ (hazard.csv, p_fail_runs.csv, summary.csv, summary.md, phase_failure_alignment*.png)")


if __name__ == "__main__":
    main()
