#!/usr/bin/env python3
"""Add a reward bonus to HDF5 transitions ending by timeout.

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
        description="Apply a terminal timeout success bonus to a Holosoma HDF5 offline dataset."
    )
    parser.add_argument("input", type=Path, help="Input HDF5 dataset path.")
    parser.add_argument("output", type=Path, help="Output HDF5 dataset path.")
    parser.add_argument(
        "--bonus",
        type=float,
        default=1.0,
        help="Reward delta applied to timeout transitions. Use a positive value.",
    )
    parser.add_argument(
        "--mode",
        choices=("terminal", "episode"),
        default="terminal",
        help=(
            "terminal: apply only to transitions where next_done_timeout == 1; "
            "episode: apply to every transition in episodes containing timeout."
        ),
    )
    parser.add_argument(
        "--timeout-key",
        default="next_done_timeout",
        help="HDF5 key containing timeout flags.",
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
        "--exclude-key",
        default="next_done_bad_tracking",
        help="Optional HDF5 flag key to exclude from bonus rows. Use '' to disable.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def _flag_mask(h5_file: h5py.File, key: str, num_samples: int, *, required: bool) -> np.ndarray:
    if not key:
        return np.zeros(num_samples, dtype=bool)
    if key not in h5_file:
        if required:
            raise KeyError(f"Dataset is missing flag key '{key}'.")
        return np.zeros(num_samples, dtype=bool)
    return np.asarray(h5_file[key][:num_samples]).astype(bool).reshape(-1)


def _terminal_timeout_mask(
    h5_file: h5py.File,
    *,
    timeout_key: str,
    exclude_key: str,
    num_samples: int,
) -> np.ndarray:
    timeout_mask = _flag_mask(h5_file, timeout_key, num_samples, required=True)
    exclude_mask = _flag_mask(h5_file, exclude_key, num_samples, required=False)
    return timeout_mask & ~exclude_mask


def _timeout_episode_mask(
    h5_file: h5py.File,
    *,
    timeout_key: str,
    episode_key: str,
    exclude_key: str,
    num_samples: int,
) -> np.ndarray:
    if episode_key not in h5_file:
        raise KeyError(f"--mode episode requires episode key '{episode_key}'.")

    terminal_mask = _terminal_timeout_mask(
        h5_file,
        timeout_key=timeout_key,
        exclude_key=exclude_key,
        num_samples=num_samples,
    )
    if not np.any(terminal_mask):
        return terminal_mask

    episode_ids = np.asarray(h5_file[episode_key][:num_samples]).reshape(-1)
    timeout_episode_ids = np.unique(episode_ids[terminal_mask])
    return np.isin(episode_ids, timeout_episode_ids)


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
            bonus_mask = _terminal_timeout_mask(
                h5_file,
                timeout_key=args.timeout_key,
                exclude_key=args.exclude_key,
                num_samples=num_samples,
            )
        else:
            bonus_mask = _timeout_episode_mask(
                h5_file,
                timeout_key=args.timeout_key,
                episode_key=args.episode_key,
                exclude_key=args.exclude_key,
                num_samples=num_samples,
            )

        reward_values = np.asarray(rewards[:num_samples])
        original_shape = reward_values.shape
        reward_flat = reward_values.reshape(num_samples, -1)
        reward_flat[bonus_mask] += np.asarray(args.bonus, dtype=reward_flat.dtype)
        rewards[:num_samples] = reward_flat.reshape(original_shape)

        h5_file.attrs["done_timeout_bonus"] = float(args.bonus)
        h5_file.attrs["done_timeout_bonus_mode"] = args.mode
        h5_file.attrs["done_timeout_bonus_key"] = args.timeout_key
        h5_file.attrs["done_timeout_bonus_exclude_key"] = args.exclude_key
        h5_file.attrs["done_timeout_bonus_num_rows"] = int(np.count_nonzero(bonus_mask))

    print(
        f"wrote {args.output} | mode={args.mode} | bonus={args.bonus} | "
        f"modified_rows={int(np.count_nonzero(bonus_mask))}/{num_samples}"
    )


if __name__ == "__main__":
    main()
