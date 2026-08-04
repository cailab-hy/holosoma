#!/usr/bin/env python3
"""Probe actor convergence toward survivor, mixture, and AW-weighted action centers.

For fixed probe rows in each requested motion-phase bin, this script restores each
checkpoint, evaluates its deterministic actor action, and reports the trajectory of

    ||pi(s) - mu_SURV||, ||pi(s) - mu_mix||, ||pi(s) - mu_w||.

The three action centers are computed once from the dataset and then shared by every
run/checkpoint. ``mu_SURV`` uses rows whose episode does not terminate with bad
tracking in the current/next wall bin, ``mu_mix`` uses every row in the bin, and
``mu_w`` uses the exact AW-CQL sidecar weights. Probe rows reuse ``probe_rows.npz``
when available, including its reward-hash guard and SURV/FAIL cell split.

Example:
  python probe_target.py offline_data/cell3.h5 \
      --weights offline_data/cell3.h5.aw_weights.npz \
      --ckpt aw,cql,50000,/path/to/aw/model_0050000.pt \
      --ckpt os_aw,cql,50000,/path/to/os_aw/model_0050000.pt \
      --ckpt cql,cql,50000,/path/to/cql/model_0050000.pt \
      --bins 0 3 --index-cache probe_rows.npz --out probe_target_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass

import numpy as np

from aw_wall_probe import batched, build_scorer, label_rows, load_arrays


@dataclass(frozen=True)
class ActionCenters:
    survivor: np.ndarray
    mixture: np.ndarray
    weighted: np.ndarray
    num_survivor: int
    num_mixture: int
    weight_sum: float


def _load_aw_weights(path: str, num_rows: int, reward_hash: str | None) -> np.ndarray:
    """Load the exact AW sidecar after enforcing its H5 pairing guards."""
    with np.load(path, allow_pickle=False) as sidecar:
        required = ("weight", "n", "rhash")
        missing = [key for key in required if key not in sidecar]
        if missing:
            raise KeyError(f"AW sidecar '{path}' is missing keys: {missing}")
        weights = np.asarray(sidecar["weight"], dtype=np.float64).reshape(-1)
        stored_rows = int(np.asarray(sidecar["n"]).item())
        stored_hash = str(np.asarray(sidecar["rhash"]).item())

    if stored_rows != num_rows or len(weights) != num_rows:
        raise ValueError(
            f"sidecar/H5 row mismatch: sidecar_n={stored_rows:,}, "
            f"weights={len(weights):,}, H5={num_rows:,}"
        )
    if reward_hash is not None and stored_hash != reward_hash:
        raise ValueError(
            f"sidecar/H5 rhash mismatch: sidecar={stored_hash}, H5={reward_hash}. "
            "Use the same .aw_weights.npz that the AW run trained with."
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("AW sidecar contains non-finite or negative weights")
    print(f"[pairing] rhash OK  N={num_rows:,}  weights={path}")
    return weights


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


def _load_or_select_probe_rows(
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
) -> dict[int, np.ndarray]:
    """Reuse wall-probe cells, selecting and merging only missing cells."""
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
            if key in cells:
                continue
            pool = bin_rows[label_mask]
            if len(pool) <= per_cell:
                selected = pool
            else:
                selected = rng.choice(pool, per_cell, replace=False)
            cells[key] = np.sort(selected.astype(np.int64, copy=False))
            changed = True
            print(f"[rows] selected bin {phase_bin} {label}: {len(selected):,}/{len(pool):,}")

    if span_indices is None:
        span_indices = np.empty(0, dtype=np.int64)
    if cache_path and changed:
        _save_cache(cache_path, cells, span_indices, reward_hash)
        print(f"[rows] merged missing cells into {cache_path}")

    probe_rows = {}
    for phase_bin in target_bins:
        selected = np.unique(
            np.concatenate(
                [cells[(phase_bin, "SURV")], cells[(phase_bin, "FAIL")]]
            )
        )
        if len(selected) == 0:
            raise ValueError(f"phase bin {phase_bin} has no reusable probe rows")
        probe_rows[phase_bin] = selected
        print(f"[rows] bin {phase_bin} probe union: {len(selected):,}")
    return probe_rows


def _compute_action_centers(
    actions: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
    episode_ids: np.ndarray,
    terminal_bins: np.ndarray,
    terminal_bad: np.ndarray,
    target_bins: list[int],
    wall_span: int,
) -> dict[int, ActionCenters]:
    """Compute bin-conditioned centers once; checkpoints never affect them."""
    centers = {}
    action_values = np.asarray(actions, dtype=np.float64)
    for phase_bin in target_bins:
        bin_rows = np.flatnonzero(bins == phase_bin)
        fail_episodes = (
            terminal_bad
            & (terminal_bins >= phase_bin)
            & (terminal_bins <= phase_bin + wall_span)
        )
        survivor_rows = bin_rows[~fail_episodes[episode_ids[bin_rows]]]
        if len(survivor_rows) == 0:
            raise ValueError(f"phase bin {phase_bin} has no SURV rows for mu_SURV")

        bin_weights = weights[bin_rows]
        weight_sum = float(bin_weights.sum())
        if weight_sum <= 0.0:
            raise ValueError(f"phase bin {phase_bin} has non-positive AW weight mass")

        centers[phase_bin] = ActionCenters(
            survivor=action_values[survivor_rows].mean(axis=0),
            mixture=action_values[bin_rows].mean(axis=0),
            weighted=(action_values[bin_rows] * bin_weights[:, None]).sum(axis=0) / weight_sum,
            num_survivor=len(survivor_rows),
            num_mixture=len(bin_rows),
            weight_sum=weight_sum,
        )
        print(
            f"[centers] bin {phase_bin}: SURV={len(survivor_rows):,} "
            f"mix={len(bin_rows):,} weight_mass={weight_sum:.2f}"
        )
    return centers


def _distance_stats(actor_actions: np.ndarray, center: np.ndarray) -> dict[str, float]:
    distances = np.linalg.norm(actor_actions - center[None, :], axis=-1)
    return {
        "mean": float(distances.mean()),
        "std": float(distances.std()),
        "p50": float(np.percentile(distances, 50)),
        "p90": float(np.percentile(distances, 90)),
    }


def _parse_checkpoint(spec: str) -> tuple[str, str, int, str]:
    try:
        run, algo, step, path = spec.split(",", 3)
        return run, algo, int(step), path
    except ValueError as error:
        raise ValueError(
            f"invalid --ckpt '{spec}'; expected RUNLABEL,ALGO,STEP,PATH"
        ) from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5")
    parser.add_argument(
        "--ckpt",
        action="append",
        required=True,
        help="RUNLABEL,ALGO,STEP,PATH (repeatable; ALGO is cql or iql)",
    )
    parser.add_argument("--weights", default=None, help="AW sidecar; default: <h5>.aw_weights.npz")
    parser.add_argument("--bins", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--wall-span", type=int, default=1)
    parser.add_argument("--per-cell", type=int, default=2000)
    parser.add_argument("--index-cache", default="probe_rows.npz")
    parser.add_argument("--out", default="probe_target_results.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    rewards, phase, dones, truncations, bad, observations, _, actions, reward_hash = load_arrays(args.h5)
    print(f"[load] N={len(rewards):,} action_dim={actions.shape[-1]} rhash={reward_hash}")
    bins, episode_ids, terminal_bins, terminal_bad = label_rows(
        phase, dones, truncations, bad, n_bins=args.n_bins
    )
    sidecar_path = args.weights or f"{args.h5}.aw_weights.npz"
    weights = _load_aw_weights(sidecar_path, len(rewards), reward_hash)
    probe_rows = _load_or_select_probe_rows(
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
    centers = _compute_action_centers(
        actions,
        weights,
        bins,
        episode_ids,
        terminal_bins,
        terminal_bad,
        args.bins,
        args.wall_span,
    )

    output_rows = []
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
            indices = probe_rows[phase_bin]
            actor_actions = batched(policy_fn, observations[indices], bs=args.batch_size)
            if actor_actions.shape != (len(indices), actions.shape[-1]):
                raise ValueError(
                    f"actor action shape mismatch for {run}@{step}: got {actor_actions.shape}, "
                    f"expected {(len(indices), actions.shape[-1])}"
                )

            center = centers[phase_bin]
            survivor_stats = _distance_stats(actor_actions, center.survivor)
            mixture_stats = _distance_stats(actor_actions, center.mixture)
            weighted_stats = _distance_stats(actor_actions, center.weighted)
            means = {
                "SURV": survivor_stats["mean"],
                "mix": mixture_stats["mean"],
                "w": weighted_stats["mean"],
            }
            closest = min(means, key=means.get)
            output_rows.append(
                {
                    "run": run,
                    "algo": algo,
                    "step": step,
                    "bin": phase_bin,
                    "phase_start": phase_bin / args.n_bins,
                    "phase_end": (phase_bin + 1) / args.n_bins,
                    "n_probe": len(indices),
                    "n_surv_reference": center.num_survivor,
                    "n_mix_reference": center.num_mixture,
                    "aw_weight_mass": center.weight_sum,
                    "d_surv_mean": survivor_stats["mean"],
                    "d_surv_std": survivor_stats["std"],
                    "d_surv_p50": survivor_stats["p50"],
                    "d_surv_p90": survivor_stats["p90"],
                    "d_mix_mean": mixture_stats["mean"],
                    "d_mix_std": mixture_stats["std"],
                    "d_mix_p50": mixture_stats["p50"],
                    "d_mix_p90": mixture_stats["p90"],
                    "d_w_mean": weighted_stats["mean"],
                    "d_w_std": weighted_stats["std"],
                    "d_w_p50": weighted_stats["p50"],
                    "d_w_p90": weighted_stats["p90"],
                    "closest_target": closest,
                }
            )
            print(
                f"[{run} @{step} bin{phase_bin}] "
                f"||pi-mu_SURV||={survivor_stats['mean']:.4f} "
                f"||pi-mu_mix||={mixture_stats['mean']:.4f} "
                f"||pi-mu_w||={weighted_stats['mean']:.4f} closest={closest}"
            )

    if not output_rows:
        raise RuntimeError("no probe rows were produced")
    with open(args.out, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[saved] {args.out} ({len(output_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
