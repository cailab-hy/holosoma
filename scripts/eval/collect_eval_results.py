#!/usr/bin/env python3
"""
scripts/eval/collect_eval_results.py
====================================

Aggregate ``eval_summary.json`` files produced by the eval pipeline
into a single CSV / JSON / Markdown report, and compare against the
goldens declared in the eval manifest.

Inputs
------
* ``--eval-manifest`` : the eval manifest YAML (Step-1).
* ``--golden-manifest``: optional Step-0 golden manifest (only used
  for hyper-parameter context in the Markdown header).
* ``--out-root``      : matches the ``--out-root`` of the runner;
  the per-run ``eval_output_dir`` paths are read from the manifest.

Outputs (under ``<out-root>/_pipeline/``)
-----------------------------------------
* ``eval_results.csv``           – flat table over (experiment, step).
* ``eval_results.json``          – same data, structured.
* ``eval_results.md``            – human-readable Markdown.
* ``golden_eval_comparison.csv`` – PASS / WARN / FAIL per golden.
* ``golden_eval_comparison.json``
* ``golden_eval_comparison.md``

Tolerance semantics
-------------------
Each metric tolerance is either ``{kind: absolute, value: x}`` or
``{kind: relative, value: x}``.
* PASS  → ``|actual − expected| ≤ tolerance``
* WARN  → ``≤ 2 × tolerance``
* FAIL  → otherwise
For ``mean_min_hand_obj_dist`` ``actual ≤ expected`` is also PASS
(lower-is-better metric).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LOWER_IS_BETTER = {"mean_min_hand_obj_dist", "mean_action_norm",
                   "bad_tracking_after_lift_rate"}

PRIMARY_METRICS = [
    "success_rate",
    "mean_reward",
    "mean_length",
    "lift_rate_v1",
    "first_contact_rate",
    "grasp_rate_v1",
    "grasp_rate_v2",
    "lift_after_contact_rate",
    "mean_min_hand_obj_dist",
    "mean_max_obj_height",
    "mean_goal_progress_frac",
    "mean_action_norm",
    "bad_tracking_after_lift_rate",
]

# Best-checkpoint scoring rank (used to mark "best" within a sweep).
BEST_RANK = [
    ("success_rate",            "higher"),
    ("mean_reward",             "higher"),
    ("lift_rate_v1",            "higher"),
    ("mean_min_hand_obj_dist",  "lower"),
    ("mean_goal_progress_frac", "higher"),
]


@dataclass
class EvalRow:
    experiment_name: str
    algorithm_name: str
    algorithm_group: str
    step: int
    checkpoint: str
    eval_dir: str
    metrics: dict[str, float]
    status: str          # "ok" | "missing"

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "experiment_name":  self.experiment_name,
            "algorithm_name":   self.algorithm_name,
            "algorithm_group":  self.algorithm_group,
            "step":             self.step,
            "checkpoint":       self.checkpoint,
            "eval_dir":         self.eval_dir,
            "status":           self.status,
        }
        for k in PRIMARY_METRICS:
            d[k] = self.metrics.get(k, None)
        return d


# ─────────────────────────────────────────────────────────────────────
def load_eval_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def gather_rows(manifest: dict[str, Any]) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for run in manifest.get("runs", []) or []:
        eval_root = Path(run["eval_output_dir"])
        pattern   = run.get("checkpoint_pattern", "model_{step:07d}.pt")
        run_dir   = Path(run["run_dir"])
        for step in run.get("checkpoint_steps", []) or []:
            ck_path = run_dir / pattern.format(step=step)
            out_dir = eval_root / f"step_{step}"
            summary = load_eval_summary(out_dir / "eval_summary.json")
            metrics = summary if summary else {}
            rows.append(EvalRow(
                experiment_name=run["experiment_name"],
                algorithm_name=run["algorithm_name"],
                algorithm_group=run["algorithm_group"],
                step=int(step),
                checkpoint=str(ck_path),
                eval_dir=str(out_dir),
                metrics={k: metrics.get(k) for k in PRIMARY_METRICS
                         if metrics.get(k) is not None},
                status="ok" if summary else "missing",
            ))
    return rows


def classify(actual: float, expected: float, tol: dict[str, Any],
             metric_name: str) -> str:
    if actual is None or expected is None:
        return "MISSING"
    kind = tol.get("kind", "absolute")
    value = float(tol.get("value", 0.0))
    if kind == "absolute":
        delta_pass = value
    else:
        delta_pass = abs(expected) * value
    diff = actual - expected
    # Lower-is-better: actual below expected is fine regardless of tol.
    if metric_name in LOWER_IS_BETTER and diff <= 0:
        return "PASS"
    if abs(diff) <= delta_pass:
        return "PASS"
    if abs(diff) <= 2.0 * delta_pass:
        return "WARN"
    return "FAIL"


def build_golden_compare(manifest: dict[str, Any],
                         rows: list[EvalRow]) -> list[dict[str, Any]]:
    defaults_tol = (manifest.get("defaults") or {}).get("tolerance") or {}
    out: list[dict[str, Any]] = []
    by_key = {(r.experiment_name, r.step): r for r in rows}
    for run in manifest.get("runs", []) or []:
        expected = run.get("expected_eval_metrics") or {}
        if not expected:
            continue
        run_tol = {**defaults_tol, **(run.get("tolerance") or {})}
        for step in run.get("checkpoint_steps", []) or []:
            row = by_key.get((run["experiment_name"], int(step)))
            if row is None or row.status != "ok":
                out.append({
                    "experiment_name": run["experiment_name"],
                    "step": int(step),
                    "overall": "MISSING",
                    "details": {m: {"expected": v, "actual": None,
                                    "verdict": "MISSING"}
                                for m, v in expected.items()},
                })
                continue
            details: dict[str, Any] = {}
            worst = "PASS"
            order = {"PASS": 0, "WARN": 1, "FAIL": 2, "MISSING": 2}
            for m, exp_val in expected.items():
                actual = row.metrics.get(m)
                tol = run_tol.get(m, {"kind": "absolute", "value": 0.05})
                verdict = classify(actual, exp_val, tol, m)
                details[m] = {"expected": exp_val, "actual": actual,
                              "tolerance": tol, "verdict": verdict}
                if order[verdict] > order[worst]:
                    worst = verdict
            out.append({
                "experiment_name": run["experiment_name"],
                "algorithm_name":  run["algorithm_name"],
                "step":            int(step),
                "overall":         worst,
                "details":         details,
            })
    return out


def mark_best(rows: list[EvalRow]) -> dict[str, tuple[int, dict]]:
    """Return per-experiment {experiment_name: (best_step, metrics)}."""
    grouped: dict[str, list[EvalRow]] = {}
    for r in rows:
        if r.status != "ok":
            continue
        grouped.setdefault(r.experiment_name, []).append(r)
    best: dict[str, tuple[int, dict]] = {}
    for name, group in grouped.items():
        def score(r: EvalRow):
            t = []
            for metric, direction in BEST_RANK:
                v = r.metrics.get(metric)
                if v is None:
                    v = -math.inf if direction == "higher" else math.inf
                t.append(-v if direction == "higher" else v)
            return tuple(t)
        winner = min(group, key=score)
        best[name] = (winner.step, winner.metrics)
    return best


# ─────────────────────────────────────────────────────────────────────
def write_csv(rows: list[EvalRow], path: Path) -> None:
    import csv
    cols = ["experiment_name", "algorithm_name", "algorithm_group", "step",
            "status", *PRIMARY_METRICS, "checkpoint", "eval_dir"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r.as_dict())


def write_json(rows: list[EvalRow], path: Path) -> None:
    with path.open("w") as f:
        json.dump([r.as_dict() for r in rows], f, indent=2)


def write_md(rows: list[EvalRow], best: dict[str, tuple[int, dict]],
             path: Path) -> None:
    lines: list[str] = []
    lines.append("# Eval Pipeline Results\n")
    lines.append(f"Total entries: **{len(rows)}**  "
                 f"(ok={sum(1 for r in rows if r.status=='ok')}, "
                 f"missing={sum(1 for r in rows if r.status=='missing')})\n")
    by_exp: dict[str, list[EvalRow]] = {}
    for r in rows:
        by_exp.setdefault(r.experiment_name, []).append(r)
    for exp, group in by_exp.items():
        lines.append(f"\n## {exp}\n")
        best_step = best.get(exp, (None, None))[0]
        header = ["step", "status", "success_rate", "mean_reward",
                  "lift_rate_v1", "mean_min_hand_obj_dist",
                  "mean_goal_progress_frac"]
        lines.append("| " + " | ".join(header + ["best?"]) + " |")
        lines.append("|" + "|".join(["---"] * (len(header) + 1)) + "|")
        for r in sorted(group, key=lambda x: x.step):
            cells = [str(r.step), r.status]
            for k in header[2:]:
                v = r.metrics.get(k)
                cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
            cells.append("★" if r.step == best_step else "")
            lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_golden_md(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Golden Comparison\n"]
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "MISSING": 0}
    for r in rows:
        counts[r["overall"]] = counts.get(r["overall"], 0) + 1
    lines.append("Overall: " + ", ".join(f"**{k}**={v}" for k, v in counts.items()) + "\n")
    for r in rows:
        lines.append(f"\n## {r['experiment_name']} @ step {r['step']} — **{r['overall']}**\n")
        lines.append("| metric | expected | actual | tolerance | verdict |")
        lines.append("|---|---|---|---|---|")
        for m, d in r["details"].items():
            exp_v = d["expected"]
            act_v = d.get("actual")
            tol   = d.get("tolerance", {})
            exp_s = f"{exp_v:.4f}" if isinstance(exp_v, (int, float)) else str(exp_v)
            act_s = f"{act_v:.4f}" if isinstance(act_v, (int, float)) else "—"
            tol_s = (f"{tol.get('kind','?')[:3]}={tol.get('value','?')}"
                     if tol else "—")
            lines.append(f"| {m} | {exp_s} | {act_s} | {tol_s} | {d['verdict']} |")
    path.write_text("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-manifest", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path,
                    help="matches --out-root of the runner")
    ap.add_argument("--golden-manifest", type=Path, default=None,
                    help="optional Step-0 golden manifest (context only)")
    args = ap.parse_args()

    with args.eval_manifest.open() as f:
        manifest = yaml.safe_load(f)

    rows = gather_rows(manifest)
    best = mark_best(rows)
    golden = build_golden_compare(manifest, rows)

    pipe = args.out_root / "_pipeline"
    pipe.mkdir(parents=True, exist_ok=True)

    write_csv (rows, pipe / "eval_results.csv")
    write_json(rows, pipe / "eval_results.json")
    write_md  (rows, best, pipe / "eval_results.md")

    with (pipe / "golden_eval_comparison.json").open("w") as f:
        json.dump(golden, f, indent=2)
    import csv as _csv
    with (pipe / "golden_eval_comparison.csv").open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["experiment_name", "step", "metric", "expected",
                    "actual", "verdict", "overall"])
        for r in golden:
            for m, d in r["details"].items():
                w.writerow([r["experiment_name"], r["step"], m,
                            d["expected"], d.get("actual"),
                            d["verdict"], r["overall"]])
    write_golden_md(golden, pipe / "golden_eval_comparison.md")

    # Console summary
    n_ok = sum(1 for r in rows if r.status == "ok")
    print(f"[OK] {n_ok}/{len(rows)} eval entries collected → {pipe}")
    if golden:
        overall = [r["overall"] for r in golden]
        print(f"     golden verdicts: "
              f"PASS={overall.count('PASS')}  "
              f"WARN={overall.count('WARN')}  "
              f"FAIL={overall.count('FAIL')}  "
              f"MISSING={overall.count('MISSING')}")
    for exp, (step, _) in best.items():
        print(f"     best({exp}) = step {step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
