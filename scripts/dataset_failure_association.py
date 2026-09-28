#!/usr/bin/env python3
"""Failure-associated transition ratios of an offline episode dataset, per motion phase bin.

Two dataset-composition indicators complement the phase hazard h(k):

* Episode-level ratio::

      eta_F(k) = #{transitions in phase bin k whose episode ended in bad tracking}
                 / #{transitions in phase bin k}

  "Of the replay transitions in phase k, this share belongs to trajectories that eventually
  failed." Includes well-executed transitions that failed much later.

* H-step ratio (H = AW-CQL return horizon, default 50)::

      z_i^H      = 1[bad tracking occurs within the next H steps after transition i]
      eta_F^H(k) = sum_{i: k_i = k} z_i^H / N_k

  where N_k only counts transitions whose H-step future is *observable*: z_i^H = 1, or at
  least H further steps are recorded, or the episode reaches a genuine terminal (motion end)
  within H steps. Transitions cut short by a timeout / truncation / incomplete episode with
  no failure inside the window are excluded from N_k (they are not "no failure").

Failure-associated mass can precede the termination bin: an action at k=4 that causes the
failure at k=5 is counted at k=4 by eta_F^H but at k=5 by the hazard / policy termination.

Outputs (``--out-dir``): dataset_phase_profile.csv (+ hazard column if ``--hazard-csv``),
dataset_phase_profile.md, dataset_phase_profile.png.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import numpy as np


def load(h5: Path) -> dict[str, np.ndarray]:
    keys = [
        "episode_id",
        "motion_phase",
        "next_done_bad_tracking",
        "next_done_motion_ends",
        "next_done_timeout",
        "truncations",
        "episode_data_complete",
        "next_episode_step",
    ]
    with h5py.File(h5, "r") as f:
        n = int(f.attrs.get("num_samples", f["episode_id"].shape[0]))
        data = {k: f[k][:n] for k in keys if k in f}
        data["_phase_semantics"] = str(f.attrs.get("motion_phase_semantics", "unknown"))
    return data


def episode_blocks(episode_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start/end (inclusive) row indices of contiguous episode blocks."""
    change = np.flatnonzero(np.diff(episode_id) != 0) + 1
    starts = np.r_[0, change]
    ends = np.r_[change - 1, episode_id.size - 1]
    return starts, ends


def future_failure_flags(data: dict[str, np.ndarray], horizon: int) -> dict[str, np.ndarray]:
    """Per-transition failure association (the definitions behind eta_F and eta_F^H).

    Returns ``end_bad`` (episode ended in bad tracking), ``z`` (bad tracking within the next
    ``horizon`` steps) and ``determinable`` (the H-step future is observable; see module doc).
    """
    ep = data["episode_id"]
    n = ep.size
    starts, ends = episode_blocks(ep)
    # sanity: bad tracking only terminates episodes
    bad = data["next_done_bad_tracking"].astype(bool)
    if bad.sum() != bad[ends].sum():
        raise ValueError("bad-tracking flags found inside episodes; expected only at episode ends")

    block_len = ends - starts + 1
    row_block = np.repeat(np.arange(starts.size), block_len)
    end_idx = ends[row_block]
    remaining = end_idx - np.arange(n) + 1  # rows observed from i to the episode end (inclusive)

    end_bad = bad[ends][row_block]
    end_motion = data["next_done_motion_ends"].astype(bool)[ends][row_block]
    complete = (
        data["episode_data_complete"].astype(bool)[ends][row_block]
        if "episode_data_complete" in data
        else np.ones(n, bool)
    )
    genuine_terminal = end_motion & complete  # a true terminal without failure

    z = end_bad & (remaining <= horizon)
    determinable = z | (remaining >= horizon) | ((remaining < horizon) & genuine_terminal)
    return {"end_bad": end_bad, "z": z, "determinable": determinable, "starts": starts, "ends": ends, "bad": bad}


def compute(data: dict[str, np.ndarray], num_bins: int, horizon: int) -> dict[str, np.ndarray]:
    flags = future_failure_flags(data, horizon)
    end_bad, z, determinable = flags["end_bad"], flags["z"], flags["determinable"]
    starts, ends, bad = flags["starts"], flags["ends"], flags["bad"]

    phase = np.clip(data["motion_phase"], 0.0, 1.0 - 1e-9)
    bins = np.minimum((phase * num_bins).astype(np.int64), num_bins - 1)

    n_bin = np.bincount(bins, minlength=num_bins).astype(np.int64)
    n_fail_ep = np.bincount(bins, weights=end_bad.astype(np.int64), minlength=num_bins)
    n_det = np.bincount(bins, weights=determinable.astype(np.int64), minlength=num_bins)
    n_z = np.bincount(bins, weights=z.astype(np.int64), minlength=num_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta_f = np.where(n_bin > 0, n_fail_ep / n_bin, np.nan)
        eta_fh = np.where(n_det > 0, n_z / n_det, np.nan)
        undet = np.where(n_bin > 0, 1.0 - n_det / n_bin, np.nan)
    return {
        "n_transitions": n_bin,
        "n_from_failed_episodes": n_fail_ep.astype(np.int64),
        "eta_F": eta_f,
        "n_determinable": n_det.astype(np.int64),
        "n_fail_within_H": n_z.astype(np.int64),
        "eta_F_H": eta_fh,
        "undeterminable_frac": undet,
        "_episodes": starts.size,
        "_episodes_bad": int(bad[ends].sum()),
        "_episodes_motion_end": int(data["next_done_motion_ends"][ends].sum()),
    }


def read_hazard_csv(path: Path) -> list[float]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return [float(r["hazard"]) if r["hazard"] not in ("", "None", "nan") else math.nan for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default="offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5")
    ap.add_argument("--num-bins", type=int, default=20)
    ap.add_argument("--H", type=int, default=50, help="AW-CQL return horizon used for eta_F^H")
    ap.add_argument(
        "--hazard-csv", default=None, help="hazard.csv from phase_failure_alignment.py (merged into the table)"
    )
    ap.add_argument("--wall-bins", nargs="+", type=int, default=[4, 5, 12, 13])
    ap.add_argument("--out-dir", default="validation/results/phase_alignment")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load(Path(args.h5))
    res = compute(data, args.num_bins, args.H)
    hazard = read_hazard_csv(Path(args.hazard_csv)) if args.hazard_csv else [math.nan] * args.num_bins
    if len(hazard) != args.num_bins:
        raise ValueError("hazard.csv bin count differs from --num-bins")

    print(
        f"{args.h5}: episodes={res['_episodes']} "
        f"(bad={res['_episodes_bad']}, motion_end={res['_episodes_motion_end']}), "
        f"phase semantics={data['_phase_semantics']}, H={args.H}"
    )

    rows = []
    for k in range(args.num_bins):
        rows.append(  # noqa: PERF401
            {
                "bin": k,
                "phase_lo": k / args.num_bins,
                "phase_hi": (k + 1) / args.num_bins,
                "hazard": hazard[k],
                "eta_F": res["eta_F"][k],
                "eta_F_H": res["eta_F_H"][k],
                "n_transitions": int(res["n_transitions"][k]),
                "n_from_failed_episodes": int(res["n_from_failed_episodes"][k]),
                "n_determinable": int(res["n_determinable"][k]),
                "n_fail_within_H": int(res["n_fail_within_H"][k]),
                "undeterminable_frac": res["undeterminable_frac"][k],
                "is_wall": int(k in args.wall_bins),
            }
        )
    with (out / "dataset_phase_profile.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def pct(v: float) -> str:
        return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.0f}"

    lines = [
        f"# Dataset phase profile ({Path(args.h5).name}; {args.num_bins} bins; H={args.H})",
        "",
        f"episodes={res['_episodes']} (bad tracking {res['_episodes_bad']}, "
        f"motion end {res['_episodes_motion_end']}); wall bins {args.wall_bins}",
        "",
        "| bin | " + " | ".join(str(k) for k in range(args.num_bins)) + " |",
        "| --- | " + " | ".join("---" for _ in range(args.num_bins)) + " |",
        "| hazard h(k) % | " + " | ".join(pct(hazard[k]) for k in range(args.num_bins)) + " |",
        "| eta_F(k) % (episode-level) | " + " | ".join(pct(res["eta_F"][k]) for k in range(args.num_bins)) + " |",
        f"| eta_F^{args.H}(k) % (within H steps) | "
        + " | ".join(pct(res["eta_F_H"][k]) for k in range(args.num_bins))
        + " |",
        "| transitions (k) | " + " | ".join(f"{int(v) // 1000}k" for v in res["n_transitions"]) + " |",
        "| undeterminable % | " + " | ".join(pct(res["undeterminable_frac"][k]) for k in range(args.num_bins)) + " |",
    ]
    (out / "dataset_phase_profile.md").write_text("\n".join(lines))
    print("\n".join(lines))

    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    centers = (np.arange(args.num_bins) + 0.5) / args.num_bins
    width = 0.8 / args.num_bins
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for k in args.wall_bins:
        ax.axvspan(k / args.num_bins, (k + 1) / args.num_bins, color="tab:red", alpha=0.08, lw=0)
    ax.bar(centers - width / 3, 100 * np.nan_to_num(hazard), width / 3, color="tab:gray", label="hazard h(k)")
    ax.bar(centers, 100 * np.nan_to_num(res["eta_F"]), width / 3, color="tab:purple", label="eta_F(k) episode-level")
    ax.bar(
        centers + width / 3,
        100 * np.nan_to_num(res["eta_F_H"]),
        width / 3,
        color="tab:green",
        label=f"eta_F^{args.H}(k)",
    )
    ax.set_xticks(np.arange(args.num_bins) / args.num_bins)
    ax.set_xticklabels([str(k) for k in range(args.num_bins)])
    ax.set_xlim(0, 1)
    ax.set_xlabel("phase bin k")
    ax.set_ylabel("%")
    ax.set_title(f"Dataset phase profile: {Path(args.h5).name}", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "dataset_phase_profile.png", dpi=150)
    print(f"written: {out}/dataset_phase_profile.{{csv,md,png}}")


if __name__ == "__main__":
    main()
