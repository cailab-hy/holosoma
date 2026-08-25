#!/usr/bin/env python3
"""Compare two checkpoints through the exact aw_wall_probe critic loader."""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aw_wall_probe import build_scorer, load_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("checkpoint", nargs=2, metavar=("CKPT_A", "CKPT_B"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=2048)
    args = parser.parse_args()

    _, _, _, _, _, _, critic_obs, actions, _ = load_arrays(args.dataset)
    row_count = min(args.rows, len(actions))
    results = []
    for checkpoint_path in args.checkpoint:
        q_fn, _ = build_scorer("cql", checkpoint_path, args.device)
        q_values = q_fn(critic_obs[:row_count], actions[:row_count])
        metadata = q_fn.probe_metadata
        span = np.percentile(q_values, 99) - np.percentile(q_values, 1)
        results.append((metadata, q_values))
        print(
            f"checkpoint={checkpoint_path} loaded_step={metadata['checkpoint_global_step']} "
            f"checksum={metadata['critic_checksum']} "
            f"first_layer_sum={metadata['first_layer_sum']:+.9f} "
            f"mean={q_values.mean():+.9f} std={q_values.std():.9f} span={span:.9f}"
        )
    (metadata_a, q_a), (metadata_b, q_b) = results
    print(
        f"compare checksum_equal="
        f"{metadata_a['critic_checksum'] == metadata_b['critic_checksum']} "
        f"q_max_abs_diff={np.max(np.abs(q_a - q_b)):.9f}"
    )


if __name__ == "__main__":
    main()
