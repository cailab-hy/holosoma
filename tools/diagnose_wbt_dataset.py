"""Diagnose a WBT offline-collection HDF5: where do episodes die, and is the reward
telling the policy to survive?

Answers, from the dataset a *_data run is writing:
  1. Death-frame histogram — do bad_tracking deaths cluster at specific motion
     frames (physically hard sub-region) or spread uniformly (incentive problem)?
  2. Reward sign profile — mean per-step reward by motion-frame bucket; if a
     region is net-negative, early termination is return-optimal there.
  3. Training-time trend — same stats split by global_step quartile: does the
     policy's death location / reward improve or degrade over training?

Usage (training machine; stop the run or copy the file first — the writer holds
the file open without SWMR):
    python tools/diagnose_wbt_dataset.py offline_data/g1_29dof_wbt_d3_seg_a_dataset.h5
"""

from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np

BUCKET = 10  # motion frames per histogram bucket


def _load(f: h5py.File, key: str, n: int) -> np.ndarray | None:
    if key not in f:
        return None
    return np.asarray(f[key][:n]).reshape(n, -1).squeeze(-1)


def _hist_line(frames: np.ndarray, max_frame: int, label: str) -> None:
    if frames.size == 0:
        print(f"  {label}: (none)")
        return
    edges = np.arange(0, max_frame + BUCKET, BUCKET)
    counts, _ = np.histogram(frames, bins=edges)
    peak = counts.max()
    print(f"  {label}: n={frames.size}, frame mean={frames.mean():.1f}, p50={np.median(frames):.0f}")
    for lo, c in zip(edges[:-1], counts):
        if c == 0:
            continue
        bar = "#" * max(1, int(40 * c / peak))
        print(f"    [{lo:4d}-{lo + BUCKET:4d}) {c:8d} {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5", help="collected dataset (*.h5)")
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as f:
        n = int(f.attrs.get("num_samples", f["observations"].shape[0]))
        print(f"{args.h5}: {n} transitions")
        print(f"keys: {sorted(f.keys())}\n")

        frame = _load(f, "motion_time_step", n)
        rewards = _load(f, "rewards", n)
        dones = _load(f, "dones", n)
        truncs = _load(f, "truncations", n)
        bad = _load(f, "next_done_bad_tracking", n)
        seg_end = _load(f, "next_done_segment_ends", n)
        mot_end = _load(f, "next_done_motion_ends", n)
        ep_step = _load(f, "next_episode_step", n)
        gstep = _load(f, "next_global_step", n)
        err_body = _load(f, "next_err_body_pos_max", n)
        err_root = _load(f, "next_err_root_pos", n)

        if frame is None or rewards is None or dones is None:
            sys.exit("dataset is missing motion_time_step/rewards/dones — cannot diagnose")
        max_frame = int(frame.max()) + 1

        done_mask = dones > 0
        print(f"episodes ended in-dataset: {int(done_mask.sum())}")
        for name, mask in (
            ("bad_tracking", bad),
            ("segment_ends", seg_end),
            ("motion_ends", mot_end),
            ("truncations(all)", truncs),
        ):
            if mask is not None:
                print(f"  {name:18s}: {int((mask > 0).sum())}")
        if ep_step is not None and done_mask.any():
            print(f"episode length at done: mean={ep_step[done_mask].mean():.1f}, p50={np.median(ep_step[done_mask]):.0f}")

        print("\n--- 1. death / success frame histograms ---")
        if bad is not None:
            _hist_line(frame[bad > 0], max_frame, "bad_tracking deaths by motion frame")
        if seg_end is not None and (seg_end > 0).any():
            _hist_line(frame[seg_end > 0], max_frame, "segment_ends by motion frame")

        print("\n--- 2. per-frame-bucket profiles (all transitions) ---")
        edges = np.arange(0, max_frame + BUCKET, BUCKET)
        bins = np.clip(np.digitize(frame, edges) - 1, 0, len(edges) - 2)
        print(f"  {'frames':>12s} {'count':>9s} {'mean_rew':>9s} {'neg_rew%':>8s} "
              f"{'err_body':>9s} {'err_root':>9s} {'death%':>7s}")
        for b in range(len(edges) - 1):
            sel = bins == b
            cnt = int(sel.sum())
            if cnt == 0:
                continue
            row = [f"[{edges[b]:4d}-{edges[b + 1]:4d})", f"{cnt:9d}",
                   f"{rewards[sel].mean():9.4f}", f"{100.0 * (rewards[sel] < 0).mean():7.1f}%"]
            row.append(f"{err_body[sel].mean():9.3f}" if err_body is not None else f"{'n/a':>9s}")
            row.append(f"{err_root[sel].mean():9.3f}" if err_root is not None else f"{'n/a':>9s}")
            row.append(f"{100.0 * (bad[sel] > 0).mean():6.2f}%" if bad is not None else f"{'n/a':>7s}")
            print("  " + " ".join(row))

        if gstep is not None:
            print("\n--- 3. training-time trend (global_step quartiles) ---")
            qs = np.quantile(gstep, [0.25, 0.5, 0.75])
            qbin = np.digitize(gstep, qs)
            for q in range(4):
                sel = qbin == q
                if not sel.any():
                    continue
                line = (f"  Q{q + 1}: rew mean={rewards[sel].mean():+.4f}, neg%={100.0 * (rewards[sel] < 0).mean():.1f}%")
                if bad is not None and (sel & (dones > 0)).any():
                    denom = max(int((sel & (dones > 0)).sum()), 1)
                    line += f", bad_tracking share of dones={100.0 * (bad[sel & (dones > 0)] > 0).mean():.1f}%"
                if bad is not None and (bad > 0)[sel].any():
                    line += f", death frame p50={np.median(frame[sel & (bad > 0)]):.0f}"
                print(line)

        print("\nInterpretation:")
        print("  * deaths clustered in one frame band + err_body exploding there -> that motion")
        print("    sub-region is the physical bottleneck (consider w_object env / different split).")
        print("  * mean_rew < 0 over wide frame ranges -> termination is return-optimal there;")
        print("    verify the run uses the patched reward (limits_dof_pos=-10, alive=+2).")
        print("  * death frame p50 drifting EARLIER over quartiles -> learned early-termination.")


if __name__ == "__main__":
    main()
