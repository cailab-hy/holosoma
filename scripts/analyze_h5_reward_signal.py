#!/usr/bin/env python3
"""Analyze whether an HDF5 offline dataset has discriminative reward signal."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


TRACKING_ERROR_KEYS = (
    "next_err_root_pos",
    "next_err_root_ori",
    "next_err_body_pos_max",
    "next_err_body_pos_mean",
    "next_err_object_pos",
    "next_err_object_ori",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize reward distribution, terminal balance, and reward/error correlation in HDF5 data."
    )
    parser.add_argument("path", type=Path, help="Input HDF5 dataset path.")
    parser.add_argument("--reward-key", default="rewards", help="HDF5 key containing rewards.")
    parser.add_argument("--done-key", default="dones", help="HDF5 key containing done flags.")
    parser.add_argument("--trunc-key", default="truncations", help="HDF5 key containing truncation flags.")
    parser.add_argument("--bad-key", default="next_done_bad_tracking", help="HDF5 key for bad-tracking done flags.")
    parser.add_argument("--motion-key", default="next_done_motion_ends", help="HDF5 key for motion-end done flags.")
    parser.add_argument("--timeout-key", default="next_done_timeout", help="HDF5 key for timeout flags.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount used for scale diagnostics.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    return parser.parse_args()


def _read_scalar(h5_file: h5py.File, key: str, num_samples: int, *, dtype=np.float64) -> np.ndarray:
    return np.asarray(h5_file[key][:num_samples]).reshape(num_samples, -1).mean(axis=1).astype(dtype, copy=False)


def _read_bool(h5_file: h5py.File, key: str, num_samples: int, *, required: bool = False) -> np.ndarray:
    if key not in h5_file:
        if required:
            raise KeyError(f"Required key missing: {key}")
        return np.zeros(num_samples, dtype=bool)
    return np.asarray(h5_file[key][:num_samples]).astype(bool).reshape(num_samples)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    qs = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    names = ["min", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]
    return {name: float(value) for name, value in zip(names, np.quantile(values, qs))}


def _print_distribution(name: str, values: np.ndarray) -> None:
    print(f"{name}")
    if values.size == 0:
        print("  n: 0")
        return
    q = _quantiles(values)
    mean = float(values.mean())
    std = float(values.std())
    iqr = q["p75"] - q["p25"]
    near_zero = float(np.mean(np.abs(values) < 1e-3) * 100.0)
    positive = float(np.mean(values > 0.0) * 100.0)
    negative = float(np.mean(values < 0.0) * 100.0)
    print(f"  n:      {values.size:,}")
    print(f"  mean:   {mean:.6f}")
    print(f"  std:    {std:.6f}")
    print(f"  iqr:    {iqr:.6f}")
    print(f"  min/p01/p05/p25/p50/p75/p95/p99/max:")
    print(
        "          "
        f"{q['min']:.6f} / {q['p01']:.6f} / {q['p05']:.6f} / {q['p25']:.6f} / "
        f"{q['p50']:.6f} / {q['p75']:.6f} / {q['p95']:.6f} / {q['p99']:.6f} / {q['max']:.6f}"
    )
    print(f"  sign:   positive={positive:.2f}% negative={negative:.2f}% near_zero(|r|<1e-3)={near_zero:.2f}%")


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    x = x[mask]
    y = y[mask]
    if float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _print_error_correlation(h5_file: h5py.File, rewards: np.ndarray, nonterminal: np.ndarray, num_samples: int) -> None:
    available_keys = [key for key in TRACKING_ERROR_KEYS if key in h5_file]
    if not available_keys:
        print("Tracking error correlation")
        print("  no next_err_* keys found")
        return

    print("Tracking error correlation")
    print("  corr(reward, -error) should be positive if reward clearly prefers lower tracking error.")
    for key in available_keys:
        error = _read_scalar(h5_file, key, num_samples)
        corr_all = _safe_corr(rewards, -error)
        corr_nonterminal = _safe_corr(rewards[nonterminal], -error[nonterminal])
        print(
            f"  {key:<24} corr_all={corr_all if corr_all is not None else 'n/a'} "
            f"corr_nonterminal={corr_nonterminal if corr_nonterminal is not None else 'n/a'}"
        )

        if int(nonterminal.sum()) >= 10:
            nt_error = error[nonterminal]
            nt_reward = rewards[nonterminal]
            low_cut, high_cut = np.quantile(nt_error, [0.25, 0.75])
            low_mask = nt_error <= low_cut
            high_mask = nt_error >= high_cut
            low_reward = float(nt_reward[low_mask].mean()) if low_mask.any() else float("nan")
            high_reward = float(nt_reward[high_mask].mean()) if high_mask.any() else float("nan")
            print(
                f"    nonterminal reward low-error p25={low_reward:.6f}, "
                f"high-error p75={high_reward:.6f}, delta={low_reward - high_reward:.6f}"
            )


def _print_file_fraction_stats(
    rewards: np.ndarray,
    done_or_trunc: np.ndarray,
    bad: np.ndarray,
    motion: np.ndarray,
    timeout: np.ndarray,
) -> None:
    print("File-order thirds")
    n = rewards.shape[0]
    bounds = [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]
    for idx, (start, end) in enumerate(bounds, start=1):
        sl = slice(start, end)
        denom = max(end - start, 1)
        print(
            f"  third{idx}: rows={denom:,} reward_mean={float(rewards[sl].mean()):.6f} "
            f"done/trunc={100.0 * float(done_or_trunc[sl].mean()):.3f}% "
            f"bad={100.0 * float(bad[sl].mean()):.3f}% "
            f"motion={100.0 * float(motion[sl].mean()):.3f}% "
            f"timeout={100.0 * float(timeout[sl].mean()):.3f}%"
        )


def main() -> None:
    args = _parse_args()
    if not args.path.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {args.path}")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError(f"--gamma must be in (0, 1], got {args.gamma}")

    with h5py.File(args.path, "r") as h5_file:
        if args.reward_key not in h5_file:
            raise KeyError(f"Required reward key missing: {args.reward_key}")
        num_samples = int(h5_file.attrs.get("num_samples", h5_file[args.reward_key].shape[0]))

        rewards = _read_scalar(h5_file, args.reward_key, num_samples)
        done = _read_bool(h5_file, args.done_key, num_samples, required=True)
        trunc = _read_bool(h5_file, args.trunc_key, num_samples)
        bad = _read_bool(h5_file, args.bad_key, num_samples)
        motion = _read_bool(h5_file, args.motion_key, num_samples)
        timeout = _read_bool(h5_file, args.timeout_key, num_samples)
        done_or_trunc = done | trunc
        nonterminal = ~done_or_trunc

        print(f"path: {args.path}")
        print(f"num_samples: {num_samples:,}")
        print()
        _print_distribution("Reward distribution: all", rewards)
        _print_distribution("Reward distribution: nonterminal", rewards[nonterminal])
        _print_distribution("Reward distribution: done_or_truncated", rewards[done_or_trunc])
        _print_distribution("Reward distribution: bad_tracking terminal", rewards[bad & done_or_trunc])
        _print_distribution("Reward distribution: motion_ends terminal", rewards[motion & done_or_trunc])
        _print_distribution("Reward distribution: timeout terminal", rewards[timeout & done_or_trunc])
        print()

        print("Terminal sampling expectation")
        bad_rate = float(bad.mean())
        timeout_rate = float(timeout.mean())
        motion_rate = float(motion.mean())
        done_rate = float(done_or_trunc.mean())
        for batch_size in args.batch_sizes:
            print(
                f"  batch={batch_size:<5} done/trunc={batch_size * done_rate:.2f} "
                f"bad={batch_size * bad_rate:.2f} motion={batch_size * motion_rate:.2f} "
                f"timeout={batch_size * timeout_rate:.2f}"
            )
        if args.gamma < 1.0:
            print()
            print("Discount scale")
            nonterminal_mean = float(rewards[nonterminal].mean()) if nonterminal.any() else 0.0
            print(f"  nonterminal_mean / (1-gamma): {nonterminal_mean / (1.0 - args.gamma):.6f}")
            for horizon in (50, 100, 250, 500):
                print(f"  gamma^{horizon:<3}: {args.gamma ** horizon:.6f}")
        print()

        _print_error_correlation(h5_file, rewards, nonterminal, num_samples)
        print()
        _print_file_fraction_stats(rewards, done_or_trunc, bad, motion, timeout)


if __name__ == "__main__":
    main()
