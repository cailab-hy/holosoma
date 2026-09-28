#!/usr/bin/env python3
"""Re-anchor a whole-body-tracking motion clip near its env origin.

MotionCommand adds ``env_origins`` to the clip's raw world positions, so a clip whose
root sits far from (0, 0) makes every robot walk that far away from its own env cell.
With ``env_spacing`` 2.5 m and ``xy_offset_range`` 1.0 m, a clip that drifts past ~1.2 m
reaches into the neighbouring cell. This shifts the whole clip in XY so the root path is
centred on the origin, which minimises the worst-case distance.

Translation-invariant fields (velocities, quaternions, joint angles) are untouched, and
so is the ``world`` body row, which is a fixed frame marker at (0, 0, 0) that the loader
never selects. Z is never shifted -- ground contact must not move.

Usage:
  python scripts/recenter_wbt_motion.py MOTION.npz              # in place
  python scripts/recenter_wbt_motion.py MOTION.npz -o OUT.npz
  python scripts/recenter_wbt_motion.py MOTION.npz --anchor start --dry-run
"""
from __future__ import annotations

import argparse
import numpy as np

ROOT_QPOS_XY = slice(0, 2)
WORLD_BODY = "world"


def root_xy_path(data):
    """Pelvis XY over time, taken from qpos so it is independent of body ordering."""
    return np.asarray(data["joint_pos"])[:, ROOT_QPOS_XY]


def compute_shift(path, anchor):
    """XY translation to apply. 'bbox' minimises the max distance from the origin."""
    if anchor == "start":
        return path[0].copy()
    if anchor == "mean":
        return path.mean(axis=0)
    if anchor == "bbox":
        return 0.5 * (path.min(axis=0) + path.max(axis=0))
    raise ValueError(f"unknown anchor: {anchor}")


def recenter(data, shift):
    out = {k: np.array(data[k]) for k in data.files}
    body_names = [str(n) for n in out["body_names"]]

    out["joint_pos"][:, ROOT_QPOS_XY] -= shift

    keep = np.array([n != WORLD_BODY for n in body_names])
    out["body_pos_w"][:, keep, 0:2] -= shift
    if "object_pos_w" in out:
        out["object_pos_w"][:, 0:2] -= shift
    return out


def describe(path, label):
    dist = np.hypot(path[:, 0], path[:, 1])
    return (f"  {label:9s} start={dist[0]:.3f}m  end={dist[-1]:.3f}m  max={dist.max():.3f}m  "
            f"x[{path[:, 0].min():+.2f},{path[:, 0].max():+.2f}] "
            f"y[{path[:, 1].min():+.2f},{path[:, 1].max():+.2f}]")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("-o", "--out", default=None, help="default: overwrite the input")
    ap.add_argument("--anchor", choices=("bbox", "mean", "start"), default="bbox")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    with np.load(a.motion, allow_pickle=False) as data:
        before = root_xy_path(data)
        shift = compute_shift(before, a.anchor)
        out = recenter(data, shift)

    after = out["joint_pos"][:, ROOT_QPOS_XY]
    print(f"[recenter] {a.motion}  anchor={a.anchor}  shift=({shift[0]:+.4f}, {shift[1]:+.4f})")
    print(describe(before, "before"))
    print(describe(after, "after"))

    # Z and the world marker must survive untouched.
    with np.load(a.motion, allow_pickle=False) as data:
        assert np.array_equal(out["body_pos_w"][:, :, 2], data["body_pos_w"][:, :, 2]), "Z moved"
        assert np.array_equal(out["joint_pos"][:, 2:], data["joint_pos"][:, 2:]), "root Z / joints moved"
        for key in ("joint_vel", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            assert np.array_equal(out[key], data[key]), f"{key} moved"
        names = [str(n) for n in data["body_names"]]
        if WORLD_BODY in names:
            i = names.index(WORLD_BODY)
            assert np.array_equal(out["body_pos_w"][:, i], data["body_pos_w"][:, i]), "world marker moved"
    print("[recenter] invariants OK (Z, joints, velocities, quaternions, world marker unchanged)")

    if a.dry_run:
        print("[recenter] dry-run: nothing written")
        return 0
    dest = a.out or a.motion
    np.savez(dest, **out)
    print(f"[recenter] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
