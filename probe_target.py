#!/usr/bin/env python3
"""Paired actor-target probe for wall-bin survivor/failure rows.

The original probe compared ``pi(s)`` with action centers averaged across every
state in a phase bin. In WBT that destroys the state-conditioned action signal:
the center is dominated by between-state variation and may not resemble a valid
action for any individual state.

This paired version compares the actor and dataset action at the SAME state:

    d_SURV = E[||pi(s_i) - a_D,i|| | i is a SURV probe row]
    d_FAIL = E[||pi(s_i) - a_D,i|| | i is a FAIL probe row]
    paired_gap = d_SURV - d_FAIL

A negative gap means the actor is closer to survivor actions; a positive gap
means it is closer to failure actions. The main CSV records this trajectory per
checkpoint/bin. A second CSV decomposes the absolute and signed error by action
dimension so a small subset of failure-critical joints cannot be hidden by the
29-dimensional L2 norm.

Probe cells reuse the SURV/FAIL indices in ``probe_rows.npz`` and retain its H5
reward-hash guard. Missing cells are selected with the same wall-probe labeling:
FAIL means the episode has a bad-tracking terminal in [bin, bin + wall_span].
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

from aw_wall_probe import batched, build_scorer, label_rows, load_arrays


def _load_cache(path: str, reward_hash: str | None) -> tuple[dict, np.ndarray | None]:
    if not path or not os.path.exists(path):
        return {}, None
    with np.load(path, allow_pickle=True) as cache:
        cached_hash = str(np.asarray(cache["rhash"]).item())
        if reward_hash is not None and cached_hash != reward_hash:
            print(f"[rows] cache rhash mismatch ({cached_hash} != {reward_hash}); ignoring {path}")
            return {}, None
        cells = {
            (int(key[0]), str(key[1])): np.asarray(indices, dtype=np.int64).reshape(-1)
            for key, indices in zip(cache["cell_keys"], cache["cell_idx"])
        }
        span_indices = (
            np.asarray(cache["span_idx"], dtype=np.int64).reshape(-1)
            if "span_idx" in cache
            else None
        )
    print(f"[rows] loaded {len(cells)} cached cells from {path}")
    return cells, span_indices


def _save_cache(
    path: str,
    cells: dict[tuple[int, str], np.ndarray],
    span_indices: np.ndarray,
    reward_hash: str | None,
) -> None:
    ordered_keys = sorted(cells)
    object_indices = np.empty(len(ordered_keys), dtype=object)
    for index, key in enumerate(ordered_keys):
        object_indices[index] = np.asarray(cells[key], dtype=np.int64)
    np.savez_compressed(
        path,
        cell_keys=np.asarray(ordered_keys),
        cell_idx=object_indices,
        span_idx=np.asarray(span_indices, dtype=np.int64),
        rhash=str(reward_hash),
    )


def _load_or_select_probe_cells(
    *,
    cache_path: str,
    reward_hash: str | None,
    bins: np.ndarray,
    episode_ids: np.ndarray,
    terminal_bins: np.ndarray,
    terminal_bad: np.ndarray,
    target_bins: list[int],
    per_cell: int,
    wall_span: int,
    seed: int,
) -> dict[tuple[int, str], np.ndarray]:
    """Reuse cached probe cells and merge only cells missing for this probe."""
    cells, span_indices = _load_cache(cache_path, reward_hash)
    rng = np.random.default_rng(seed)
    changed = False

    for phase_bin in target_bins:
        bin_rows = np.flatnonzero(bins == phase_bin)
        if len(bin_rows) == 0:
            raise ValueError(f"phase bin {phase_bin} contains no dataset rows")
        fail_episodes = (
            terminal_bad
            & (terminal_bins >= phase_bin)
            & (terminal_bins <= phase_bin + wall_span)
        )
        is_failure = fail_episodes[episode_ids[bin_rows]]
        for label, label_mask in (("SURV", ~is_failure), ("FAIL", is_failure)):
            key = (phase_bin, label)
            if key not in cells:
                pool = bin_rows[label_mask]
                selected = (
                    pool
                    if len(pool) <= per_cell
                    else rng.choice(pool, per_cell, replace=False)
                )
                cells[key] = np.sort(selected.astype(np.int64, copy=False))
                changed = True
                print(f"[rows] selected bin {phase_bin} {label}: {len(selected):,}/{len(pool):,}")

            if len(cells[key]) == 0:
                raise ValueError(
                    f"phase bin {phase_bin} has no {label} probe rows; paired comparison is undefined"
                )
            print(f"[rows] bin {phase_bin} {label}: {len(cells[key]):,} probe rows")

    if span_indices is None:
        span_indices = np.empty(0, dtype=np.int64)
    if cache_path and changed:
        _save_cache(cache_path, cells, span_indices, reward_hash)
        print(f"[rows] merged missing cells into {cache_path}")
    return cells


def _parse_checkpoint(spec: str) -> tuple[str, str, int, str]:
    try:
        run, algo, step, path = spec.split(",", 3)
        return run, algo, int(step), path
    except ValueError as error:
        raise ValueError(
            f"invalid --ckpt '{spec}'; expected RUNLABEL,ALGO,STEP,PATH"
        ) from error


def _action_names(action_dim: int) -> list[str]:
    """Use the configured G1 joint order when dimensions match; otherwise use indices."""
    repo_root = Path(__file__).resolve().parent
    source_root = repo_root / "src" / "holosoma"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        from holosoma.config_values.robot import g1_29dof

        if len(g1_29dof.dof_names) == action_dim:
            return list(g1_29dof.dof_names)
    except (ImportError, AttributeError):
        pass
    return [f"action_{index}" for index in range(action_dim)]


def _paired_errors(
    policy_fn,
    observations: np.ndarray,
    dataset_actions: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actor_actions = batched(policy_fn, observations[indices], bs=batch_size)
    expected_shape = (len(indices), dataset_actions.shape[-1])
    if actor_actions.shape != expected_shape:
        raise ValueError(
            f"actor action shape mismatch: got {actor_actions.shape}, expected {expected_shape}"
        )
    signed_error = actor_actions.astype(np.float64) - dataset_actions[indices].astype(np.float64)
    absolute_error = np.abs(signed_error)
    l2_error = np.linalg.norm(signed_error, axis=-1)
    return l2_error, absolute_error, signed_error


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def _default_per_dim_path(output_path: str) -> str:
    root, extension = os.path.splitext(output_path)
    return f"{root}_per_dim{extension or '.csv'}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5")
    parser.add_argument(
        "--ckpt",
        action="append",
        required=True,
        help="RUNLABEL,ALGO,STEP,PATH (repeatable; ALGO is cql or iql)",
    )
    parser.add_argument("--bins", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--wall-span", type=int, default=1)
    parser.add_argument("--per-cell", type=int, default=2000)
    parser.add_argument("--index-cache", default="probe_rows.npz")
    parser.add_argument("--out", default="probe_target_paired.csv")
    parser.add_argument(
        "--per-dim-out",
        default=None,
        help="per-action output; default: <out stem>_per_dim.csv",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--weights",
        default=None,
        help="deprecated compatibility option; paired v2 does not use AW weights",
    )
    args = parser.parse_args(argv)

    if args.weights is not None:
        print("[note] --weights is ignored: paired v2 compares pi(s) directly with a_D at the same state")

    rewards, phase, dones, truncations, bad, observations, _, actions, reward_hash = load_arrays(args.h5)
    print(f"[load] N={len(rewards):,} action_dim={actions.shape[-1]} rhash={reward_hash}")
    bins, episode_ids, terminal_bins, terminal_bad = label_rows(
        phase, dones, truncations, bad, n_bins=args.n_bins
    )
    probe_cells = _load_or_select_probe_cells(
        cache_path=args.index_cache,
        reward_hash=reward_hash,
        bins=bins,
        episode_ids=episode_ids,
        terminal_bins=terminal_bins,
        terminal_bad=terminal_bad,
        target_bins=args.bins,
        per_cell=args.per_cell,
        wall_span=args.wall_span,
        seed=args.seed,
    )

    output_rows = []
    per_dim_rows = []
    action_names = _action_names(int(actions.shape[-1]))
    dry_rng = np.random.default_rng(args.seed + 1)

    for checkpoint_spec in args.ckpt:
        run, algo, step, checkpoint_path = _parse_checkpoint(checkpoint_spec)
        if args.dry_run:
            action_dim = int(actions.shape[-1])

            def policy_fn(values, *, _action_dim=action_dim):
                return dry_rng.normal(size=(len(values), _action_dim)).astype(np.float32)

        else:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
            _, policy_fn = build_scorer(algo, checkpoint_path, args.device)

        for phase_bin in args.bins:
            errors = {}
            for label in ("SURV", "FAIL"):
                indices = probe_cells[(phase_bin, label)]
                errors[label] = _paired_errors(
                    policy_fn,
                    observations,
                    actions,
                    indices,
                    args.batch_size,
                )

            surv_l2, surv_abs, surv_signed = errors["SURV"]
            fail_l2, fail_abs, fail_signed = errors["FAIL"]
            surv_stats = _summary(surv_l2)
            fail_stats = _summary(fail_l2)
            paired_gap = surv_stats["mean"] - fail_stats["mean"]
            output_rows.append(
                {
                    "run": run,
                    "algo": algo,
                    "step": step,
                    "bin": phase_bin,
                    "phase_start": phase_bin / args.n_bins,
                    "phase_end": (phase_bin + 1) / args.n_bins,
                    "n_surv": len(surv_l2),
                    "n_fail": len(fail_l2),
                    "d_surv_mean": surv_stats["mean"],
                    "d_surv_std": surv_stats["std"],
                    "d_surv_p50": surv_stats["p50"],
                    "d_surv_p90": surv_stats["p90"],
                    "d_fail_mean": fail_stats["mean"],
                    "d_fail_std": fail_stats["std"],
                    "d_fail_p50": fail_stats["p50"],
                    "d_fail_p90": fail_stats["p90"],
                    "paired_gap_surv_minus_fail": paired_gap,
                    "d_fail_over_surv": fail_stats["mean"] / max(surv_stats["mean"], 1e-12),
                }
            )

            surv_abs_mean = surv_abs.mean(axis=0)
            fail_abs_mean = fail_abs.mean(axis=0)
            surv_signed_mean = surv_signed.mean(axis=0)
            fail_signed_mean = fail_signed.mean(axis=0)
            for action_index, action_name in enumerate(action_names):
                per_dim_rows.append(
                    {
                        "run": run,
                        "algo": algo,
                        "step": step,
                        "bin": phase_bin,
                        "action_index": action_index,
                        "action_name": action_name,
                        "abs_error_surv": surv_abs_mean[action_index],
                        "abs_error_fail": fail_abs_mean[action_index],
                        "abs_gap_surv_minus_fail": (
                            surv_abs_mean[action_index] - fail_abs_mean[action_index]
                        ),
                        "signed_error_surv": surv_signed_mean[action_index],
                        "signed_error_fail": fail_signed_mean[action_index],
                    }
                )

            side = "SURV" if paired_gap < 0.0 else "FAIL"
            print(
                f"[{run} @{step} bin{phase_bin}] "
                f"d_SURV={surv_stats['mean']:.4f} d_FAIL={fail_stats['mean']:.4f} "
                f"gap(S-F)={paired_gap:+.4f} closer={side}"
            )

    if not output_rows:
        raise RuntimeError("no probe rows were produced")
    with open(args.out, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    per_dim_path = args.per_dim_out or _default_per_dim_path(args.out)
    with open(per_dim_path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(per_dim_rows[0]))
        writer.writeheader()
        writer.writerows(per_dim_rows)

    print(f"[saved] paired summary: {args.out} ({len(output_rows)} rows)")
    print(f"[saved] per-dim detail: {per_dim_path} ({len(per_dim_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
