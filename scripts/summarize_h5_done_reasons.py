#!/usr/bin/env python3
"""Summarize done-reason flags in a Holosoma HDF5 offline dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize HDF5 done reason ratios.")
    parser.add_argument("path", type=Path, help="Input HDF5 dataset path.")
    parser.add_argument("--done-key", default="dones", help="HDF5 key containing done flags.")
    parser.add_argument("--trunc-key", default="truncations", help="HDF5 key containing truncation flags.")
    parser.add_argument("--bad-key", default="next_done_bad_tracking", help="HDF5 key for bad-tracking done flags.")
    parser.add_argument("--motion-key", default="next_done_motion_ends", help="HDF5 key for motion-end done flags.")
    parser.add_argument("--timeout-key", default="next_done_timeout", help="HDF5 key for timeout flags.")
    parser.add_argument("--reward-key", default="rewards", help="HDF5 key containing rewards.")
    return parser.parse_args()


def _read_bool(h5_file: h5py.File, key: str, num_samples: int, *, required: bool = False) -> np.ndarray:
    if key not in h5_file:
        if required:
            raise KeyError(f"Required key missing: {key}")
        return np.zeros(num_samples, dtype=bool)
    return np.asarray(h5_file[key][:num_samples]).astype(bool).reshape(-1)


def _fmt_ratio(count: int, denom: int) -> str:
    ratio = 0.0 if denom == 0 else 100.0 * count / denom
    return f"{count:,} / {denom:,} ({ratio:.3f}%)"


def _fmt_reward_mean(rewards: np.ndarray, mask: np.ndarray) -> str:
    count = int(mask.sum())
    if count == 0:
        return "n/a"
    return f"{float(rewards[mask].mean()):.6f} over {count:,} rows"


def main() -> None:
    args = _parse_args()
    if not args.path.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {args.path}")

    with h5py.File(args.path, "r") as h5_file:
        if args.done_key in h5_file:
            num_samples = int(h5_file.attrs.get("num_samples", h5_file[args.done_key].shape[0]))
        elif "observations" in h5_file:
            num_samples = int(h5_file.attrs.get("num_samples", h5_file["observations"].shape[0]))
        else:
            raise KeyError(f"Cannot infer num_samples; missing '{args.done_key}' and 'observations'.")

        done = _read_bool(h5_file, args.done_key, num_samples, required=True)
        trunc = _read_bool(h5_file, args.trunc_key, num_samples)
        bad = _read_bool(h5_file, args.bad_key, num_samples)
        motion = _read_bool(h5_file, args.motion_key, num_samples)
        timeout = _read_bool(h5_file, args.timeout_key, num_samples)
        if args.reward_key not in h5_file:
            raise KeyError(f"Required reward key missing: {args.reward_key}")
        rewards = np.asarray(h5_file[args.reward_key][:num_samples]).reshape(num_samples, -1).mean(axis=1)

    done_count = int(done.sum())
    trunc_count = int(trunc.sum())
    terminal_or_trunc = done | trunc
    terminal_or_trunc_count = int(terminal_or_trunc.sum())

    masks = {
        "bad_tracking": bad,
        "motion_ends": motion,
        "timeout": timeout,
    }
    reason_union = bad | motion | timeout
    overlap = sum(mask.astype(np.int16) for mask in masks.values()) > 1
    done_without_reason = done & ~reason_union
    reason_without_done_or_trunc = reason_union & ~terminal_or_trunc

    print(f"path: {args.path}")
    print(f"num_samples: {num_samples:,}")
    print()
    print("Overall")
    print(f"  dones:             {_fmt_ratio(done_count, num_samples)}")
    print(f"  truncations:       {_fmt_ratio(trunc_count, num_samples)}")
    print(f"  done_or_truncated: {_fmt_ratio(terminal_or_trunc_count, num_samples)}")
    print()
    print("Reason ratios among dones")
    for name, mask in masks.items():
        count = int((mask & done).sum())
        print(f"  {name:<14} {_fmt_ratio(count, done_count)}")
    print(f"  no_reason      {_fmt_ratio(int(done_without_reason.sum()), done_count)}")
    print()
    print("Reason ratios among done_or_truncated")
    for name, mask in masks.items():
        count = int((mask & terminal_or_trunc).sum())
        print(f"  {name:<14} {_fmt_ratio(count, terminal_or_trunc_count)}")
    print()
    print("Reason ratios among all transitions")
    for name, mask in masks.items():
        print(f"  {name:<14} {_fmt_ratio(int(mask.sum()), num_samples)}")
    print()
    print("Reward means")
    print(f"  all_transitions      {float(rewards.mean()):.6f} over {num_samples:,} rows")
    print(f"  done_or_truncated    {_fmt_reward_mean(rewards, terminal_or_trunc)}")
    for name, mask in masks.items():
        print(f"  {name:<18} {_fmt_reward_mean(rewards, mask & terminal_or_trunc)}")
    print(f"  no_reason_terminal   {_fmt_reward_mean(rewards, done_without_reason)}")
    print()
    print("Sanity")
    print(f"  overlapping_reasons:        {_fmt_ratio(int(overlap.sum()), num_samples)}")
    print(f"  reason_without_done/trunc:  {_fmt_ratio(int(reason_without_done_or_trunc.sum()), num_samples)}")


if __name__ == "__main__":
    main()
