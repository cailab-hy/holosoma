#!/usr/bin/env python3
"""Scale rewards in a copied Holosoma HDF5 offline dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale reward values in a Holosoma HDF5 offline dataset copy.")
    parser.add_argument("input", type=Path, help="Input HDF5 dataset path.")
    parser.add_argument("output", type=Path, help="Output HDF5 dataset path.")
    parser.add_argument("--scale", type=float, required=True, help="Multiplicative reward scale.")
    parser.add_argument("--reward-key", default="rewards", help="HDF5 key containing rewards.")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it already exists.")
    return parser.parse_args()


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

        reward_values = np.asarray(rewards[:num_samples])
        rewards[:num_samples] = reward_values * np.asarray(args.scale, dtype=reward_values.dtype)

        h5_file.attrs["reward_scale"] = float(args.scale)
        h5_file.attrs["reward_scale_key"] = args.reward_key
        h5_file.attrs["reward_scale_num_rows"] = int(num_samples)

    print(f"wrote {args.output} | scale={args.scale} | modified_rows={num_samples}")


if __name__ == "__main__":
    main()
