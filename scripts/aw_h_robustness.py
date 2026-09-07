#!/usr/bin/env python3
"""
AW-CQL : robustness of the precomputed weights to the return horizon H.

Recomputes w for H in {0.5*H0, H0, 2*H0} on the SAME h5 and asks whether the three
weightings prioritise the same transitions. Deterministic precomputed weights have no
sampling noise, so this is NOT a confidence-interval question: the right language is
rank consistency / set overlap / distributional stability.

Reported per H (vs the H0 reference):
  * Spearman rho(w^H, w^H0)              -- ordering agreement over all N transitions
  * top-k overlap                        -- |A_k ∩ B_k| / k and Jaccard, k = 1/5/10% of N
  * R_surv^{w,H}                         -- weighted survivor mass share in the wall bins
                                            (Measurement-C definition: FAIL = episode ends
                                            in bad_tracking within [b, b+span])
  * ESS/N, clip%, weight percentiles, KS distance, mean|dw|
  * horizon realisation                  -- fraction of rows whose full H-step window fits
                                            inside the episode (H is truncated at ep end)

beta is recalibrated per H (beta = beta_scale * std(A_hat^H)), which is what
aw_precompute_weights.py does; --beta-abs pins one absolute beta across all three
horizons instead. Note that w is a monotone function of A_hat below the clip, so with
clip%=0 the rank metrics are beta-invariant and measure A_hat^H alone -- only ESS,
clip%, R_surv and the KS distance move with beta.

Usage:
  python scripts/aw_h_robustness.py offline_data/xxx.h5
  python scripts/aw_h_robustness.py offline_data/xxx.h5 \
      --wall-bins 4 5 13 --csv h_robust.csv --wall-csv h_robust_wall.csv

Exit code: 0 = weight construction is H-robust by the configured gates, 2 = not.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

try:
    from scripts.aw_precompute_weights import (
        episode_bounds,
        ess_frac,
        h5_reward_fingerprint,
        load_arrays,
        make_weights,
        per_group_baseline,
        reward_fingerprint,
        truncated_returns,
    )
except ModuleNotFoundError:  # invoked as `python scripts/aw_h_robustness.py`
    from aw_precompute_weights import (
        episode_bounds,
        ess_frac,
        h5_reward_fingerprint,
        load_arrays,
        make_weights,
        per_group_baseline,
        reward_fingerprint,
        truncated_returns,
    )


# ---------------------------------------------------------------- rank metrics


def spearman_rho(x, y):
    """Spearman rank correlation with tie-averaged ranks (clipped weights tie a lot)."""
    from scipy.stats import rankdata

    rank_x = rankdata(x, method="average")
    rank_y = rankdata(y, method="average")
    rank_x -= rank_x.mean()
    rank_y -= rank_y.mean()
    denominator = np.sqrt(float(rank_x @ rank_x) * float(rank_y @ rank_y))
    if denominator <= 0.0:
        return float("nan")
    return float((rank_x @ rank_y) / denominator)


def kendall_tau_subsample(x, y, sample_size, seed):
    """Kendall tau-b on a random subsample (full-N Kendall is O(N log N) but heavy)."""
    from scipy.stats import kendalltau

    if sample_size <= 0 or sample_size >= len(x):
        idx = np.arange(len(x))
    else:
        idx = np.random.default_rng(seed).choice(len(x), size=sample_size, replace=False)
    return float(kendalltau(x[idx], y[idx]).statistic)


def top_k_overlap(x, y, k):
    """Overlap of the top-k sets. Returns (overlap = |A∩B|/k, jaccard = |A∩B|/|A∪B|)."""
    if k <= 0:
        return float("nan"), float("nan")
    top_x = np.argpartition(x, -k)[-k:]
    top_y = np.argpartition(y, -k)[-k:]
    mask = np.zeros(len(x), dtype=bool)
    mask[top_x] = True
    intersection = int(mask[top_y].sum())
    union = 2 * k - intersection
    return intersection / k, intersection / max(union, 1)


def ks_distance(x, y):
    """Two-sample KS statistic between the two weight distributions."""
    from scipy.stats import ks_2samp

    return float(ks_2samp(x, y, method="asymp").statistic)


# ------------------------------------------------------------- wall-bin R_surv


def episode_index(starts, ends, num_rows):
    ep_id = np.zeros(num_rows, np.int64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        ep_id[s : e + 1] = i
    return ep_id


def survivor_mass(weight, bins, ep_id, terminal_bad, terminal_bin, bin_index, wall_span):
    """R_surv^{w,H} for one wall bin: SURV share of the anchor w-mass in that bin.

    FAIL episodes are those ending in bad_tracking inside [b, b+span] (Measurement C).
    The count share is H-independent and is returned alongside as the reference point.
    """
    rows = bins == bin_index
    if not rows.any():
        return None
    fail_episode = (
        terminal_bad & (terminal_bin >= bin_index) & (terminal_bin <= bin_index + wall_span)
    )
    is_fail = fail_episode[ep_id[rows]]
    bin_weight = weight[rows]
    fail_mass = float(bin_weight[is_fail].sum())
    surv_mass = float(bin_weight[~is_fail].sum())
    return {
        "bin": bin_index,
        "rows": int(rows.sum()),
        "n_fail": int(is_fail.sum()),
        "n_surv": int((~is_fail).sum()),
        "surv_count_share": float((~is_fail).mean()),
        "R_surv": surv_mass / max(fail_mass + surv_mass, 1e-12),
        "mean_w_fail": float(bin_weight[is_fail].mean()) if is_fail.any() else float("nan"),
        "mean_w_surv": float(bin_weight[~is_fail].mean()) if (~is_fail).any() else float("nan"),
    }


# ------------------------------------------------------------------ per-H pass


def horizon_realised_frac(starts, ends, num_rows, horizon):
    """Fraction of rows with t+H still inside the episode (full window, no truncation)."""
    remaining = np.zeros(num_rows, np.int64)
    for s, e in zip(starts, ends):
        remaining[s : e + 1] = np.arange(e - s, -1, -1)
    return float((remaining >= horizon).mean())


def compute_arm(rewards, group, starts, ends, gamma, horizon, w_max, beta_scale, beta_abs):
    returns = truncated_returns(rewards, starts, ends, gamma, horizon)
    advantage = returns - per_group_baseline(returns, group)
    sigma = float(advantage.std())
    beta = beta_abs if beta_abs is not None else beta_scale * sigma
    weight, clip_frac = make_weights(advantage, beta, w_max)
    return {
        "H": horizon,
        "sigma": sigma,
        "beta": float(beta),
        "clip_frac": float(clip_frac),
        "ess_frac": float(ess_frac(weight)),
        "weight": weight.astype(np.float64),
        "advantage": advantage,
        "gH": returns,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5")
    ap.add_argument("--npz", default=None,
                    help="reference sidecar to read H0/gamma/beta defaults from and to "
                         "cross-check the H0 arm against (default: <h5>.aw_weights.npz)")
    ap.add_argument("--H0", type=int, default=None,
                    help="reference horizon; default: sidecar's H, else 50")
    ap.add_argument("--h-multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--gamma", type=float, default=None, help="default: sidecar's gamma, else 0.99")
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--w-max", type=float, default=None, help="default: sidecar's w_max, else 10.0")
    ap.add_argument("--beta-scale", type=float, default=1.0,
                    help="per-H beta = beta_scale * std(A_hat^H) (the precompute default)")
    ap.add_argument("--beta-abs", type=float, default=None,
                    help="pin one absolute beta across all horizons instead of recalibrating")
    ap.add_argument("--wall-bins", type=int, nargs="+", default=[4, 5, 13, 14])
    ap.add_argument("--wall-span", type=int, default=1)
    ap.add_argument("--topk-fracs", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    ap.add_argument("--primary-topk", type=float, default=0.10,
                    help="the top-k fraction the gate is evaluated on")
    ap.add_argument("--kendall-sample", type=int, default=200_000,
                    help="rows subsampled for Kendall tau-b (0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None, help="write the per-H summary table here")
    ap.add_argument("--wall-csv", default=None, help="write the per-H per-wall-bin table here")
    ap.add_argument("--json", default=None, help="write the full result dict here")
    ap.add_argument("--save-weights-dir", default=None,
                    help="also write a full sidecar npz per H into this directory")
    ap.add_argument("--min-spearman", type=float, default=0.90)
    ap.add_argument("--min-topk-overlap", type=float, default=0.70)
    ap.add_argument("--max-rsurv-shift", type=float, default=0.05,
                    help="max |R_surv^H - R_surv^H0| allowed in any wall bin")
    ap.add_argument("--max-ess-shift", type=float, default=0.10,
                    help="max |ESS/N^H - ESS/N^H0| allowed")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    sidecar_path = a.npz or (a.h5 + ".aw_weights.npz")

    reference = {}
    if os.path.exists(sidecar_path):
        with np.load(sidecar_path, allow_pickle=False) as sidecar:
            reference = {
                "H": int(np.asarray(sidecar["H"]).item()) if "H" in sidecar else None,
                "gamma": float(np.asarray(sidecar["gamma"]).item()) if "gamma" in sidecar else None,
                "w_max": float(np.asarray(sidecar["w_max"]).item()) if "w_max" in sidecar else None,
                "beta": float(np.asarray(sidecar["beta"]).item()) if "beta" in sidecar else None,
                "sigma": float(np.asarray(sidecar["sigma"]).item()) if "sigma" in sidecar else None,
                "ess_frac": float(np.asarray(sidecar["ess_frac"]).item()) if "ess_frac" in sidecar else None,
                "rhash": str(np.asarray(sidecar["rhash"]).item()) if "rhash" in sidecar else None,
                "weight": np.asarray(sidecar["weight"], dtype=np.float64) if "weight" in sidecar else None,
            }
        actual_hash, _, _ = h5_reward_fingerprint(a.h5)
        if reference.get("rhash") not in (None, actual_hash):
            raise ValueError(
                f"rhash mismatch: sidecar '{sidecar_path}' has {reference['rhash']}, "
                f"H5 '{a.h5}' has {actual_hash} -- the sidecar is not paired with this dataset"
            )
        print(f"[reference] {sidecar_path}  H={reference['H']} gamma={reference['gamma']} "
              f"w_max={reference['w_max']} beta={reference['beta']} ESS/N={reference['ess_frac']}")
    else:
        print(f"[reference] no sidecar at {sidecar_path}; falling back to CLI/default settings")

    H0 = a.H0 or reference.get("H") or 50
    gamma = a.gamma if a.gamma is not None else (reference.get("gamma") or 0.99)
    w_max = a.w_max if a.w_max is not None else (reference.get("w_max") or 10.0)

    rewards, phase, dones, truncs, motion_id, semantics, bad, motion_ends, _ = load_arrays(a.h5)
    num_rows = len(rewards)
    starts, ends = episode_bounds(dones, truncs)
    ep_len = ends - starts + 1
    bins = np.clip((phase * a.n_bins).astype(int), 0, a.n_bins - 1)
    group = bins if motion_id is None else (motion_id.astype(np.int64) * 10_000 + bins)
    print(f"[load] N={num_rows:,}  episodes={len(starts):,}  ep_len mean={ep_len.mean():.1f} "
          f"p50={np.median(ep_len):.0f} p90={np.percentile(ep_len, 90):.0f}  phase_semantics={semantics}")
    print(f"[setup] H0={H0} gamma={gamma} n_bins={a.n_bins} w_max={w_max} "
          f"beta={'abs %.6f' % a.beta_abs if a.beta_abs is not None else '%g*sigma(A^H) per H' % a.beta_scale}")

    horizons = []
    for multiplier in a.h_multipliers:
        horizon = max(int(round(multiplier * H0)), 1)
        if horizon not in [h for h, _ in horizons]:
            horizons.append((horizon, multiplier))
    if H0 not in [h for h, _ in horizons]:
        raise ValueError(f"H0={H0} is not among the recomputed horizons {[h for h, _ in horizons]}")

    ep_id = episode_index(starts, ends, num_rows)
    terminal_bin = bins[ends]
    terminal_bad = bad[ends] if bad is not None else np.zeros(len(ends), bool)
    if bad is None:
        print("[WARN] next_done_bad_tracking missing -> R_surv is undefined; wall table skipped")

    arms = {}
    for horizon, multiplier in horizons:
        print(f"\n[compute] H={horizon} ({multiplier:g}*H0) ...", flush=True)
        arm = compute_arm(rewards, group, starts, ends, gamma, horizon, w_max, a.beta_scale, a.beta_abs)
        arm["multiplier"] = multiplier
        arm["realised_frac"] = horizon_realised_frac(starts, ends, num_rows, horizon)
        arms[horizon] = arm
        print(f"[compute] H={horizon}  sigma={arm['sigma']:.5f} beta={arm['beta']:.5f} "
              f"ESS/N={arm['ess_frac']:.3f} clip%={100 * arm['clip_frac']:.2f} "
              f"full-window rows={100 * arm['realised_frac']:.1f}%")

    base = arms[H0]
    base_weight = base["weight"]

    # sanity: the H0 arm must reproduce the shipped sidecar it is being compared against
    if reference.get("weight") is not None and len(reference["weight"]) == num_rows:
        if abs(gamma - (reference.get("gamma") or gamma)) < 1e-12 and H0 == reference.get("H"):
            max_dev = float(np.abs(base_weight - reference["weight"]).max())
            rho_ref = spearman_rho(base_weight, reference["weight"])
            verdict = "PASS" if max_dev < 1e-3 and rho_ref > 0.9999 else "FAIL"
            print(f"\n[reproduction] H0 arm vs shipped sidecar: max|dw|={max_dev:.2e} "
                  f"spearman={rho_ref:.6f} -> {verdict}")
            if verdict == "FAIL":
                print("[WARN] the H0 arm does not reproduce the sidecar (different n_bins/beta/w_max?); "
                      "comparisons below are still internally consistent but are not about the shipped weights")

    percentile_points = [1, 10, 50, 90, 99]
    rows = []
    for horizon, _ in horizons:
        arm = arms[horizon]
        weight = arm["weight"]
        quantiles = np.percentile(weight, percentile_points)
        record = {
            "H": horizon,
            "H_over_H0": horizon / H0,
            "sigma_A": arm["sigma"],
            "beta": arm["beta"],
            "ess_frac": arm["ess_frac"],
            "clip_frac": arm["clip_frac"],
            "full_window_frac": arm["realised_frac"],
            "w_max_obs": float(weight.max()),
            **{f"w_p{p:02d}": float(q) for p, q in zip(percentile_points, quantiles)},
        }
        if horizon == H0:
            record.update(spearman=1.0, kendall_tau=1.0, ks=0.0, mean_abs_dw=0.0)
            for frac in a.topk_fracs:
                record[f"top{frac:.0%}_overlap".replace("%", "pct")] = 1.0
                record[f"top{frac:.0%}_jaccard".replace("%", "pct")] = 1.0
        else:
            record["spearman"] = spearman_rho(weight, base_weight)
            record["kendall_tau"] = (
                kendall_tau_subsample(weight, base_weight, a.kendall_sample, a.seed)
                if a.kendall_sample else float("nan")
            )
            record["ks"] = ks_distance(weight, base_weight)
            record["mean_abs_dw"] = float(np.abs(weight - base_weight).mean())
            for frac in a.topk_fracs:
                overlap, jaccard = top_k_overlap(weight, base_weight, int(round(frac * num_rows)))
                record[f"top{frac:.0%}_overlap".replace("%", "pct")] = overlap
                record[f"top{frac:.0%}_jaccard".replace("%", "pct")] = jaccard
        rows.append(record)

    print("\n== weight distribution per H ==")
    print(f"{'H':>6} {'H/H0':>6} {'sigma_A':>9} {'beta':>9} {'ESS/N':>7} {'clip%':>7} "
          f"{'fullwin%':>9} {'w_p01':>7} {'w_p50':>7} {'w_p99':>7} {'w_max':>8}")
    for record in rows:
        print(f"{record['H']:>6} {record['H_over_H0']:>6.2f} {record['sigma_A']:>9.5f} "
              f"{record['beta']:>9.5f} {record['ess_frac']:>7.3f} {100 * record['clip_frac']:>6.2f}% "
              f"{100 * record['full_window_frac']:>8.1f}% {record['w_p01']:>7.3f} "
              f"{record['w_p50']:>7.3f} {record['w_p99']:>7.3f} {record['w_max_obs']:>8.3f}")

    print(f"\n== ordering agreement vs H0={H0} ==")
    topk_cols = [f"top{frac:.0%}_overlap".replace("%", "pct") for frac in a.topk_fracs]
    header = f"{'H':>6} {'spearman':>9} {'kendall':>8} {'KS':>7} {'mean|dw|':>9}"
    header += "".join(f"{col.replace('_overlap',''):>10}" for col in topk_cols)
    print(header)
    for record in rows:
        line = (f"{record['H']:>6} {record['spearman']:>9.4f} {record['kendall_tau']:>8.4f} "
                f"{record['ks']:>7.4f} {record['mean_abs_dw']:>9.4f}")
        line += "".join(f"{record[col]:>10.4f}" for col in topk_cols)
        print(line)
    print("(top-k = overlap |A∩B|/k of the highest-weight k rows; Jaccard is in the CSV/JSON)")

    wall_rows = []
    if bad is not None:
        print(f"\n== wall-bin weighted survivor mass R_surv^(w,H)  (span={a.wall_span}) ==")
        print(f"{'bin':>4} {'rows':>10} {'cnt_share_SURV':>15}" +
              "".join(f"{'R_surv H=' + str(h):>16}" for h, _ in horizons) +
              f"{'max|dR|':>9}")
        for bin_index in a.wall_bins:
            per_h = {}
            for horizon, _ in horizons:
                stat = survivor_mass(arms[horizon]["weight"], bins, ep_id, terminal_bad,
                                     terminal_bin, bin_index, a.wall_span)
                if stat is None:
                    break
                per_h[horizon] = stat
            if not per_h:
                print(f"{bin_index:>4} {'no rows':>10}")
                continue
            base_stat = per_h[H0]
            max_shift = max(abs(per_h[h]["R_surv"] - base_stat["R_surv"]) for h, _ in horizons)
            print(f"{bin_index:>4} {base_stat['rows']:>10,} {base_stat['surv_count_share']:>15.3f}" +
                  "".join(f"{per_h[h]['R_surv']:>16.4f}" for h, _ in horizons) +
                  f"{max_shift:>9.4f}")
            for horizon, _ in horizons:
                stat = dict(per_h[horizon])
                stat["H"] = horizon
                stat["dR_surv_vs_H0"] = stat["R_surv"] - base_stat["R_surv"]
                stat["anchor"] = "SURV" if stat["R_surv"] > 0.5 else "FAIL"
                wall_rows.append(stat)
        flips = sorted({
            r["bin"] for r in wall_rows
            if r["anchor"] != next(x["anchor"] for x in wall_rows if x["bin"] == r["bin"] and x["H"] == H0)
        })
        print(f"[anchor direction] bins whose SURV/FAIL anchor flips across H: "
              f"{flips if flips else 'none'}")

    print("\n== H-ROBUSTNESS VERDICT ==")
    robust = True
    primary_col = f"top{a.primary_topk:.0%}_overlap".replace("%", "pct")
    for record in rows:
        if record["H"] == H0:
            continue
        checks = []
        if record["spearman"] < a.min_spearman:
            checks.append(f"spearman={record['spearman']:.4f} < {a.min_spearman}")
        overlap = record.get(primary_col, float("nan"))
        if np.isfinite(overlap) and overlap < a.min_topk_overlap:
            checks.append(f"{primary_col}={overlap:.4f} < {a.min_topk_overlap}")
        if abs(record["ess_frac"] - base["ess_frac"]) > a.max_ess_shift:
            checks.append(f"|dESS/N|={abs(record['ess_frac'] - base['ess_frac']):.3f} > {a.max_ess_shift}")
        bin_shifts = [r for r in wall_rows if r["H"] == record["H"]
                      and abs(r["dR_surv_vs_H0"]) > a.max_rsurv_shift]
        for shift in bin_shifts:
            checks.append(f"bin {shift['bin']} dR_surv={shift['dR_surv_vs_H0']:+.4f} "
                          f"> {a.max_rsurv_shift}")
        if checks:
            robust = False
            print(f"  H={record['H']}: FAIL -- " + "; ".join(checks))
        else:
            print(f"  H={record['H']}: PASS (spearman={record['spearman']:.4f}, "
                  f"{primary_col}={overlap:.4f}, |dESS/N|="
                  f"{abs(record['ess_frac'] - base['ess_frac']):.3f})")
    if robust:
        print("ROBUST: over 0.5x-2x H0 the AW weighting preserves its transition ordering, "
              "its top-decile set, and its wall-bin survivor mass.")
    else:
        print("NOT ROBUST: the transitions AW treats as important depend on H within 0.5x-2x H0; "
              "report H as a tuned hyperparameter, not an incidental choice.")

    result = {
        "h5": os.path.basename(a.h5),
        "rhash": reference.get("rhash") or reward_fingerprint(rewards),
        "n": int(num_rows),
        "n_episodes": int(len(starts)),
        "H0": H0,
        "gamma": gamma,
        "n_bins": a.n_bins,
        "w_max": w_max,
        "beta_scale": a.beta_scale,
        "beta_abs": a.beta_abs,
        "wall_span": a.wall_span,
        "gates": {
            "min_spearman": a.min_spearman,
            "min_topk_overlap": a.min_topk_overlap,
            "primary_topk": a.primary_topk,
            "max_rsurv_shift": a.max_rsurv_shift,
            "max_ess_shift": a.max_ess_shift,
        },
        "per_H": rows,
        "wall": wall_rows,
        "robust": robust,
    }

    if a.csv:
        with open(a.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[saved] {a.csv}")
    if a.wall_csv and wall_rows:
        with open(a.wall_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(wall_rows[0].keys()))
            writer.writeheader()
            writer.writerows(wall_rows)
        print(f"[saved] {a.wall_csv}")
    if a.json:
        with open(a.json, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"[saved] {a.json}")
    if a.save_weights_dir:
        os.makedirs(a.save_weights_dir, exist_ok=True)
        rhash = result["rhash"]
        for horizon, _ in horizons:
            arm = arms[horizon]
            out = os.path.join(a.save_weights_dir,
                               f"{os.path.basename(a.h5)}.aw_weights.H{horizon}.npz")
            np.savez_compressed(
                out, weight=arm["weight"].astype(np.float32),
                advantage=arm["advantage"].astype(np.float32),
                gH=arm["gH"].astype(np.float32),
                phase_bin=bins.astype(np.int16), beta=arm["beta"], sigma=arm["sigma"],
                gamma=gamma, H=horizon, w_max=w_max, n=num_rows,
                ess_frac=arm["ess_frac"], clip_frac=arm["clip_frac"],
                h5=os.path.basename(a.h5), rhash=rhash,
            )
            print(f"[saved] {out}")

    return 0 if robust else 2


if __name__ == "__main__":
    sys.exit(main())
