#!/usr/bin/env python3
"""Add a reward penalty to HDF5 transitions ending by WBT bad tracking.

This is intended for offline-RL dataset ablations. By default it copies the
input HDF5 file to a new output path and modifies only the copied reward field.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a terminal bad-tracking reward penalty to a Holosoma HDF5 offline dataset."
    )
    parser.add_argument("input", type=Path, help="Input HDF5 dataset path.")
    parser.add_argument("output", type=Path, help="Output HDF5 dataset path.")
    parser.add_argument(
        "--penalty",
        type=float,
        default=-10.0,
        help="Reward delta applied to bad-tracking transitions. Use a negative value.",
    )
    parser.add_argument(
        "--mode",
        choices=("terminal", "episode"),
        default="terminal",
        help=(
            "terminal: apply only to transitions where next_done_bad_tracking == 1; "
            "episode: apply to every transition in episodes containing bad tracking."
        ),
    )
    parser.add_argument(
        "--bad-key",
        default="next_done_bad_tracking",
        help="HDF5 key containing bad-tracking flags.",
    )
    parser.add_argument(
        "--reward-key",
        default="rewards",
        help="HDF5 key containing rewards.",
    )
    parser.add_argument(
        "--episode-key",
        default="episode_id",
        help="HDF5 key containing episode ids, required for --mode episode.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def _mask_for_terminal_bad_tracking(h5_file: h5py.File, bad_key: str, num_samples: int) -> np.ndarray:
    if bad_key not in h5_file:
        raise KeyError(f"Dataset is missing bad-tracking key '{bad_key}'.")
    bad_flags = np.asarray(h5_file[bad_key][:num_samples]).astype(bool).reshape(-1)
    return bad_flags


def _mask_for_bad_tracking_episodes(
    h5_file: h5py.File,
    *,
    bad_key: str,
    episode_key: str,
    num_samples: int,
) -> np.ndarray:
    if episode_key not in h5_file:
        raise KeyError(f"--mode episode requires episode key '{episode_key}'.")

    terminal_mask = _mask_for_terminal_bad_tracking(h5_file, bad_key, num_samples)
    if not np.any(terminal_mask):
        return terminal_mask

    episode_ids = np.asarray(h5_file[episode_key][:num_samples]).reshape(-1)
    bad_episode_ids = np.unique(episode_ids[terminal_mask])
    return np.isin(episode_ids, bad_episode_ids)


def _copy_input(input_path: Path, output_path: Path, *, force: bool) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different paths. Refusing in-place edit.")
    if output_path.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {output_path}. Use --force to overwrite.")
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)


def main() -> None:
    args = _parse_args()
    _copy_input(args.input, args.output, force=args.force)

    with h5py.File(args.output, "r+") as h5_file:
        if args.reward_key not in h5_file:
            raise KeyError(f"Dataset is missing reward key '{args.reward_key}'.")

        rewards = h5_file[args.reward_key]
        num_samples = int(h5_file.attrs.get("num_samples", rewards.shape[0]))
        num_samples = min(num_samples, rewards.shape[0])

        if args.mode == "terminal":
            penalty_mask = _mask_for_terminal_bad_tracking(h5_file, args.bad_key, num_samples)
        else:
            penalty_mask = _mask_for_bad_tracking_episodes(
                h5_file,
                bad_key=args.bad_key,
                episode_key=args.episode_key,
                num_samples=num_samples,
            )

        reward_values = np.asarray(rewards[:num_samples])
        original_shape = reward_values.shape
        reward_flat = reward_values.reshape(num_samples, -1)
        reward_flat[penalty_mask] += np.asarray(args.penalty, dtype=reward_flat.dtype)
        rewards[:num_samples] = reward_flat.reshape(original_shape)

        h5_file.attrs["bad_tracking_penalty"] = float(args.penalty)
        h5_file.attrs["bad_tracking_penalty_mode"] = args.mode
        h5_file.attrs["bad_tracking_penalty_key"] = args.bad_key
        h5_file.attrs["bad_tracking_penalty_num_rows"] = int(np.count_nonzero(penalty_mask))

    print(
        f"wrote {args.output} | mode={args.mode} | penalty={args.penalty} | "
        f"modified_rows={int(np.count_nonzero(penalty_mask))}/{num_samples}"
    )


if __name__ == "__main__":
    main()
