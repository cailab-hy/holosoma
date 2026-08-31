#!/usr/bin/env python3
"""Crop an end-exclusive frame range from a LAFAN joint-position array."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def crop_lafan_motion(source: Path, output: Path, start: int, end: int) -> tuple[int, int]:
    """Save ``source[start:end]`` and return the source and cropped frame counts."""
    motion = np.load(source, allow_pickle=False)
    if motion.ndim != 3 or motion.shape[-1] != 3:
        raise ValueError(f"Expected a [frames, joints, 3] LAFAN array, got {motion.shape}")
    if start < 0 or end <= start or end > motion.shape[0]:
        raise ValueError(f"Invalid frame range [{start}, {end}) for {motion.shape[0]} source frames")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, motion[start:end])
    return motion.shape[0], end - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source LAFAN .npy file")
    parser.add_argument("output", type=Path, help="Destination cropped .npy file")
    parser.add_argument("--start", type=int, required=True, help="First source frame, inclusive")
    parser.add_argument("--end", type=int, required=True, help="Last source frame, exclusive")
    parser.add_argument("--fps", type=float, default=30.0, help="Source FPS used only for reporting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError(f"FPS must be positive, got {args.fps}")
    source_frames, cropped_frames = crop_lafan_motion(args.source, args.output, args.start, args.end)
    print(f"source={args.source} frames={source_frames}")
    print(
        f"crop=[{args.start}, {args.end}) frames={cropped_frames} "
        f"duration={cropped_frames / args.fps:.3f}s fps={args.fps:g}"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
