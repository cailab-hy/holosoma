#!/usr/bin/env python3
"""
scripts/eval/extract_train_scalars.py
=====================================

Pull a curated set of TensorBoard scalars out of one or more training
runs and emit a CSV + YAML sidecar.  Designed to make Step-1 eval
reports self-contained: each (experiment, step) eval row can be cross-
referenced against the training-time metrics at the same step.

Tag aliasing
------------
The spec referenced ``near_tau_gradient_amplification_ratio`` but the
production training code actually logs ``Loss/smqr/sg/near_tau_grad_mass``.
The alias map below resolves spec-name → actual tag.  Unknown / missing
tags are recorded as ``status: missing`` rather than failing the run.

Usage
-----
    python scripts/eval/extract_train_scalars.py \\
        --run-dir logs/hv-g1-manager/exp_S6_1m_fixed05_cqls05_bc30_seed1_bs4096 \\
        --steps 50000 100000 300000 1000000 \\
        --out-dir reports/train_scalars/exp_S6_1m \\
        --window 5000

Outputs
-------
* ``train_scalars.csv``   one row per (run, step)
* ``train_scalars.yaml``  same data, keyed by run name
* ``train_scalars.log``   resolution trace (tag → resolved / missing)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
except ImportError as e:  # pragma: no cover
    print(f"[FATAL] tensorboard not importable: {e}", file=sys.stderr)
    sys.exit(2)


# Canonical tag set we want to surface.  Keys are stable "report names";
# values are lists of candidate TB tags tried in order (first hit wins).
ALIAS_MAP: dict[str, list[str]] = {
    # ── Critic-side ──
    "td_loss":                       ["Loss/td_loss"],
    "critic_loss":                   ["Loss/critic_loss"],
    "critic_grad_norm":              ["Loss/critic_grad_norm"],
    "cql_penalty":                   ["Loss/cql_penalty"],
    "q_data_mean":                   ["Loss/q_data_mean"],
    "cql_q_rand_mean":               ["Loss/cql_q_rand_mean"],
    "cql_q_pi_mean":                 ["Loss/cql_q_pi_mean"],
    "q_overestimation_gap":          ["Loss/q_overestimation_gap"],
    "q_data_q_pi_gap":               ["Loss/q_data_q_pi_gap"],
    # ── Actor-side ──
    "actor_loss":                    ["Loss/actor_loss"],
    "bc_loss":                       ["Loss/bc_loss"],
    "rl_actor_term":                 ["Loss/rl_actor_term"],
    "action_l2_vs_data":             ["Loss/action_l2_vs_data"],
    "action_std":                    ["Loss/action_std"],
    "policy_entropy":                ["Loss/policy_entropy"],
    "actor_grad_norm":               ["Loss/actor_grad_norm"],
    # ── SMQR-SG diagnostics ──
    # Spec name → actual production tag.
    "near_tau_gradient_amplification_ratio":
        ["Loss/smqr/sg/near_tau_grad_mass"],   # alias fallback
    "near_tau_grad_mass":            ["Loss/smqr/sg/near_tau_grad_mass"],
    "rank_corr_q_vs_core_input":     ["Loss/smqr/sg/rank_corr_q_vs_core_input"],
    "rank_corr_q_vs_final_logits":   ["Loss/smqr/sg/rank_corr_q_vs_final_logits"],
    "smqr_blend_lambda_active":      ["Loss/smqr_blend_lambda_active"],
}


def load_accumulator(run_dir: Path) -> EventAccumulator:
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def resolve_tag(ea: EventAccumulator, candidates: list[str]) -> str | None:
    available = set(ea.Tags().get("scalars", []))
    for c in candidates:
        if c in available:
            return c
    return None


def value_at_step(ea: EventAccumulator, tag: str, step: int,
                  window: int) -> dict[str, Any]:
    """
    Return:
       * ``final``  – the scalar closest to ``step`` (within window).
       * ``window_mean`` – mean of values within [step-window, step+window].
       * ``window_n``    – count of samples averaged.
    """
    events = ea.Scalars(tag)
    if not events:
        return {"final": None, "window_mean": None, "window_n": 0,
                "closest_step": None}
    # closest hit
    closest = min(events, key=lambda e: abs(e.step - step))
    if abs(closest.step - step) > window:
        # not within window — still record but flag
        return {"final": None, "window_mean": None, "window_n": 0,
                "closest_step": int(closest.step)}
    bucket = [e.value for e in events
              if step - window <= e.step <= step + window]
    return {
        "final":        float(closest.value),
        "window_mean":  float(sum(bucket) / len(bucket)) if bucket else None,
        "window_n":     len(bucket),
        "closest_step": int(closest.step),
    }


def extract_run(run_dir: Path, steps: list[int],
                window: int, trace: list[str]) -> dict[str, Any]:
    ea = load_accumulator(run_dir)
    available = ea.Tags().get("scalars", [])
    trace.append(f"[INFO] {run_dir}: {len(available)} scalar tags discovered")

    # Resolve aliases once.
    resolved: dict[str, str | None] = {}
    for name, cands in ALIAS_MAP.items():
        hit = resolve_tag(ea, cands)
        resolved[name] = hit
        if hit:
            trace.append(f"  [OK]      {name:48s} ← {hit}")
        else:
            trace.append(f"  [MISSING] {name:48s} (tried {cands})")

    out: dict[str, Any] = {
        "run_dir":        str(run_dir),
        "n_scalar_tags":  len(available),
        "resolved_tags":  {k: v for k, v in resolved.items() if v},
        "missing_tags":   [k for k, v in resolved.items() if v is None],
        "by_step":        {},
    }
    for step in steps:
        row: dict[str, Any] = {}
        for name, tag in resolved.items():
            if tag is None:
                row[name] = {"status": "missing"}
                continue
            v = value_at_step(ea, tag, step, window)
            v["status"] = "ok" if v["final"] is not None else "out_of_window"
            row[name] = v
        out["by_step"][int(step)] = row
    return out


def write_outputs(results: dict[str, dict[str, Any]],
                  out_dir: Path, trace: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # YAML — full structured
    with (out_dir / "train_scalars.yaml").open("w") as f:
        yaml.safe_dump(results, f, sort_keys=False)

    # CSV — flat (one row per run × step × metric)
    csv_path = out_dir / "train_scalars.csv"
    cols = ["run_name", "step", "metric", "status",
            "final", "window_mean", "window_n", "closest_step"]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for run_name, payload in results.items():
            for step, row in payload["by_step"].items():
                for metric, v in row.items():
                    w.writerow([
                        run_name, step, metric, v.get("status"),
                        v.get("final"), v.get("window_mean"),
                        v.get("window_n"), v.get("closest_step"),
                    ])

    # log
    (out_dir / "train_scalars.log").write_text("\n".join(trace) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", type=Path, required=True,
                    help="repeatable; one per training run to extract")
    ap.add_argument("--steps", nargs="+", type=int, required=True,
                    help="training step values to sample")
    ap.add_argument("--window", type=int, default=5000,
                    help="±step window for window_mean")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    trace: list[str] = []
    results: dict[str, dict[str, Any]] = {}
    for rd in args.run_dir:
        if not rd.is_dir():
            trace.append(f"[ERROR] not a directory: {rd}")
            print(f"[ERROR] not a directory: {rd}", file=sys.stderr)
            continue
        results[rd.name] = extract_run(rd, args.steps, args.window, trace)

    write_outputs(results, args.out_dir, trace)

    # Console summary
    print(f"[OK] extracted {len(results)} run(s) → {args.out_dir}")
    for name, payload in results.items():
        print(f"  {name}: "
              f"resolved={len(payload['resolved_tags'])}  "
              f"missing={len(payload['missing_tags'])}")
        if payload["missing_tags"]:
            print(f"    missing: {payload['missing_tags']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
