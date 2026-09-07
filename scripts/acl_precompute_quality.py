#!/usr/bin/env python3
"""Precompute ACL-QL relative transition quality m(s,a) for an HDF5 dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.aw_precompute_weights import episode_bounds, h5_reward_fingerprint, reward_fingerprint


def discounted_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    truncations: np.ndarray,
    gamma: float,
) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float64)
    starts, ends = episode_bounds(dones.astype(bool), truncations.astype(bool))
    for start, end in zip(starts, ends):
        running = 0.0
        for idx in range(int(end), int(start) - 1, -1):
            running = float(rewards[idx]) + gamma * running
            out[idx] = running
    return out


def normalize01(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    return (values - lo) / max(hi - lo, eps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-quality", type=float, default=0.5)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    out = args.out or Path(f"{args.h5}.acl_quality.npz")
    if args.verify:
        actual_hash, h5_rows, _ = h5_reward_fingerprint(args.h5)
        with np.load(out, allow_pickle=False) as sidecar:
            stored_hash = str(np.asarray(sidecar["rhash"]).item())
            stored_rows = int(np.asarray(sidecar["n"]).item())
            quality_rows = int(np.asarray(sidecar["quality"]).reshape(-1).shape[0])
        ok = stored_hash == actual_hash and stored_rows == h5_rows and quality_rows == h5_rows
        print(
            f"[acl-quality verify] {'PASS' if ok else 'FAIL'} "
            f"h5_rows={h5_rows} sidecar_n={stored_rows} quality_rows={quality_rows} "
            f"rhash stored={stored_hash} actual={actual_hash}"
        )
        return 0 if ok else 1

    with h5py.File(args.h5, "r") as h5_file:
        rewards = np.asarray(h5_file["rewards"], dtype=np.float64)
        dones = np.asarray(h5_file.get("dones", np.zeros_like(rewards)), dtype=bool)
        truncations = np.asarray(h5_file.get("truncations", np.zeros_like(rewards)), dtype=bool)
        num_rows = int(h5_file.attrs.get("num_samples", rewards.shape[0]))
    rewards = rewards[:num_rows]
    dones = dones[:num_rows]
    truncations = truncations[:num_rows]

    returns = discounted_returns(rewards, dones, truncations, args.gamma)
    g_norm = normalize01(returns)
    r_norm = normalize01(rewards)
    lam = float(args.lambda_quality)
    quality = np.clip(lam * g_norm + (1.0 - lam) * r_norm, 1e-6, 1.0 - 1e-6).astype(np.float32)

    np.savez_compressed(
        out,
        quality=quality,
        returns=returns.astype(np.float32),
        rewards=rewards.astype(np.float32),
        gamma=np.float32(args.gamma),
        lambda_quality=np.float32(args.lambda_quality),
        n=np.int64(num_rows),
        num_rows=np.int64(num_rows),
        h5=args.h5.name,
        source_h5=str(args.h5),
        rhash=reward_fingerprint(rewards),
        r_min=np.float32(rewards.min()),
        r_max=np.float32(rewards.max()),
        g_min=np.float32(returns.min()),
        g_max=np.float32(returns.max()),
    )
    print(f"[acl-quality] wrote {out} n={quality.shape[0]} mean={quality.mean():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
