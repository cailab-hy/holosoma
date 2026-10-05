#!/usr/bin/env python3
# ruff: noqa: E501
"""Compare ODPR-A (OPER-A) priority weights with the AW-CQL progress-relative weights on one dataset.

Inputs: the H5 dataset, the AW phase-baseline sidecar (<h5>.aw_weights*.npz, keys weight/advantage/gH/phase_bin),
optionally the AW global-baseline sidecar, and one or more OPER-A npz files from scripts/oper_precompute_weights.py
(keys weight, weights_by_iter, advantage, value, td_target, phase_bin, rhash). Several OPER seeds are averaged
(mean of the mean-one weights, as the ODPR paper averages 3 seeds).

Reports (printed + CSV/markdown in --out-dir):
  * rhash pairing check for every file
  * weight distribution: ESS/N, std, percentiles, max, share of weight below 0.5 / above 2
  * agreement between weightings: Pearson and Spearman correlation of the weights, of the advantages, overlap of the
    top-10 % and bottom-10 % sets (Jaccard), and the same restricted to within-bin ranks
  * per-progress-bin: data mass, weight mass (AW phase, AW global, OPER), mean OPER advantage, mean AW advantage,
    mean OPER weight, and the low-return / high-return bins as in Panel A (bins whose mean G^H deviates from the
    dataset mean by more than --region-margin)
  * OPER weight std across its iterations and across seeds
  * optional: an OPER "sidecar" npz compatible with scripts/aw_post_h_validity.py (advantage = OPER advantage,
    gH = OPER TD target, sigma = std(advantage), H = --post-h-H) so the same post-H validity test can be run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def reward_fingerprint(rewards: np.ndarray) -> str:
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    return hashlib.sha256(
        np.ascontiguousarray(r[:1000]).tobytes() + np.ascontiguousarray(r[-1000:]).tobytes()
    ).hexdigest()[:16]


def ess_frac(w):
    w = np.asarray(w, np.float64)
    return float(w.sum() ** 2 / (len(w) * (w * w).sum()))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    return float(np.corrcoef(np.asarray(a, np.float64), np.asarray(b, np.float64))[0, 1])


def jaccard_top(a, b, frac, top=True):
    k = int(len(a) * frac)
    ia = np.argpartition(-a if top else a, k)[:k]
    ib = np.argpartition(-b if top else b, k)[:k]
    sa, sb = set(ia.tolist()), set(ib.tolist())
    return len(sa & sb) / len(sa | sb)


def odpr_replace_weights_linear(weight, std=2.0, eps=0.1, eps_max=None):
    """ODPR case-study load-time normalisation (utils.ReplayBuffer.replace_weights, weight_func='linear'):
    shift to min 0, convert to probabilities, rescale their std to std/N around 1/N, clip at eps/N, renormalise.
    Returned as mean-one weights (prob * N)."""
    n = len(weight)
    w = np.asarray(weight, np.float64) - weight.min()
    prob = w / w.sum()
    if std:
        scale = std / (prob.std() * n)
        prob = scale * (prob - 1 / n) + 1 / n
        if eps:
            prob = np.maximum(prob, eps / n)
        if eps_max:
            prob = np.minimum(prob, eps_max / n)
    prob = prob / prob.sum()
    return prob * n


def odpr_ess_matched(weight, target_ess, eps=0.1):
    """ODPR linear normalisation with the std parameter chosen (bisection) so that ESS/N matches target_ess.
    (Matching the std parameter itself does not match the resulting std/ESS because of the eps floor.)"""
    lo, hi = 1e-3, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ess_frac(odpr_replace_weights_linear(weight, std=mid, eps=eps)) > target_ess:
            lo = mid
        else:
            hi = mid
    return odpr_replace_weights_linear(weight, std=lo, eps=eps), lo


def within_bin_rank(x, bins, n_bins):
    """Rank of x within its phase bin, normalised to [0, 1]."""
    r = np.empty(len(x), np.float64)
    for k in range(n_bins):
        m = bins == k
        if m.any():
            order = np.argsort(np.argsort(x[m]))
            r[m] = order / max(m.sum() - 1, 1)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default="offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5")
    ap.add_argument("--aw", default="offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5.aw_weights.H50.npz")
    ap.add_argument(
        "--aw-global", default="offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5.aw_weights.H50.global.npz"
    )
    ap.add_argument("--oper", nargs="+", required=True, help="one or more OPER-A npz files (seeds are averaged)")
    ap.add_argument(
        "--oper-iter", type=int, default=0, help="use weights_by_iter[k-1] instead of the final weight (0 = final)"
    )
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--region-margin", type=float, default=0.5)
    ap.add_argument("--wall-bins", type=int, nargs="*", default=[4, 5, 13])
    ap.add_argument("--out-dir", default="validation/results/oper_vs_aw")
    ap.add_argument(
        "--post-h-H", type=int, default=50, help="H written into the OPER pseudo-sidecar for aw_post_h_validity.py"
    )
    ap.add_argument(
        "--odpr-std",
        type=float,
        default=2.0,
        help="ODPR load-time std scaling (case-study branches' argparse default and README commands: --std=2.0)",
    )
    ap.add_argument(
        "--odpr-eps", type=float, default=0.1, help="ODPR load-time clip (case-study argparse default --eps 0.1)"
    )
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.h5, "r") as f:
        n = int(f.attrs.get("num_samples", f["rewards"].shape[0]))
        rewards = f["rewards"][:n]
        phase = f["motion_phase"][:n].astype(np.float64)
    rh = reward_fingerprint(rewards)
    bins = np.clip((phase * a.n_bins).astype(int), 0, a.n_bins - 1)
    K = a.n_bins

    aw = np.load(a.aw)
    awg = np.load(a.aw_global) if a.aw_global and Path(a.aw_global).exists() else None
    opers = [np.load(p) for p in a.oper]
    lines = ["# OPER-A (ODPR-A) vs AW-CQL weights\n", f"dataset `{a.h5}` rhash {rh}, N={n:,}\n"]
    ok = True
    for name, z in [("AW phase", aw), ("AW global", awg)] + [(f"OPER seed file {i}", z) for i, z in enumerate(opers)]:
        if z is None:
            continue
        match = str(np.asarray(z["rhash"]).item()) == rh
        ok &= match
        lines.append(f"- rhash {name}: {'OK' if match else 'MISMATCH'}")
    if not ok:
        raise SystemExit("rhash mismatch; refusing to compare")

    w_aw = aw["weight"].astype(np.float64)
    A_aw = aw["advantage"].astype(np.float64)
    gH = aw["gH"].astype(np.float64)
    w_awg = awg["weight"].astype(np.float64) if awg is not None else None
    if a.oper_iter > 0:
        w_ops = [z["weights_by_iter"][a.oper_iter - 1].astype(np.float64) for z in opers]
    else:
        w_ops = [z["weight"].astype(np.float64) for z in opers]
    w_op = np.mean(w_ops, axis=0)
    w_op = w_op / w_op.mean()
    w_op_used = odpr_replace_weights_linear(w_op, std=a.odpr_std, eps=a.odpr_eps)  # what ODPR trains with
    A_op = np.mean([z["advantage"].astype(np.float64) for z in opers], axis=0)
    V_op = np.mean([z["value"].astype(np.float64) for z in opers], axis=0)
    Q_op = np.mean([z["td_target"].astype(np.float64) for z in opers], axis=0)
    cfg = json.loads(str(np.asarray(opers[0]["config_json"]).item())) if "config_json" in opers[0].files else {}
    iters = json.loads(str(np.asarray(opers[0]["iters_json"]).item())) if "iters_json" in opers[0].files else []

    # ---- distributions
    def dist(w):
        p = np.percentile(w, [1, 10, 50, 90, 99])
        return {
            "ess": ess_frac(w),
            "std": float(w.std()),
            "max": float(w.max()),
            "p1": p[0],
            "p10": p[1],
            "p50": p[2],
            "p90": p[3],
            "p99": p[4],
            "frac_below_0p5": float((w < 0.5).mean()),
            "frac_above_2": float((w > 2).mean()),
        }

    rows = {
        "AW phase (ours)": dist(w_aw),
        "OPER-A raw (iteration weights, mean one)": dist(w_op),
        f"OPER-A as used (linear std={a.odpr_std}, eps={a.odpr_eps})": dist(w_op_used),
    }
    w_op_ess, ess_std = odpr_ess_matched(w_op, ess_frac(w_aw), eps=a.odpr_eps)
    rows[f"OPER-A ESS-matched to AW (ODPR std param {ess_std:.3f})"] = dist(w_op_ess)
    if w_awg is not None:
        rows["AW global"] = dist(w_awg)
    lines.append("\n## Weight distributions (mean-one weights)\n")
    lines.append(
        "| weighting | ESS/N | std | max | p1 | p10 | p50 | p90 | p99 | share w<0.5 | share w>2 |\n|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for k, d in rows.items():
        lines.append(
            f"| {k} | {d['ess']:.3f} | {d['std']:.3f} | {d['max']:.2f} | {d['p1']:.3f} | {d['p10']:.3f} | {d['p50']:.3f} | {d['p90']:.3f} | {d['p99']:.3f} | {100 * d['frac_below_0p5']:.1f}% | {100 * d['frac_above_2']:.1f}% |"
        )
    if iters:
        lines.append(
            "\nOPER-A per iteration (seed file 0): "
            + "; ".join(
                f"iter {it['iter']}: ESS {it['ess_frac']:.3f}, std {it['weight_std']:.3f}, adv mean {it['adv_mean']:.4f}, |adv| {it['adv_abs_mean']:.4f}"
                for it in iters
            )
        )
    lines.append(f"\nOPER-A seed files averaged: {len(w_ops)} (ODPR averages 2-3 seeds before normalisation).")
    if len(w_ops) > 1:
        lines.append(
            "\nOPER-A seed agreement (Pearson of mean-one weights): "
            + ", ".join(
                f"{pearson(w_ops[i], w_ops[j]):.3f}" for i in range(len(w_ops)) for j in range(i + 1, len(w_ops))
            )
        )

    # ---- agreement
    lines.append("\n## Agreement between weightings\n")
    lines.append(
        "| pair | Pearson(w) | Spearman(w) | Jaccard top-10% | Jaccard bottom-10% | Spearman(within-bin rank) |\n|---|---|---|---|---|---|"
    )
    pairs = [("OPER-A raw vs AW phase", w_op, w_aw), ("OPER-A as used vs AW phase", w_op_used, w_aw)]
    if w_awg is not None:
        pairs += [("OPER-A vs AW global", w_op, w_awg), ("AW phase vs AW global", w_aw, w_awg)]
    for name, x, y in pairs:
        lines.append(
            f"| {name} | {pearson(x, y):.3f} | {spearman(x, y):.3f} | {jaccard_top(x, y, 0.1):.3f} | {jaccard_top(x, y, 0.1, top=False):.3f} | {spearman(within_bin_rank(x, bins, K), within_bin_rank(y, bins, K)):.3f} |"
        )
    lines.append(
        f"\nAdvantages: Pearson(OPER adv, AW A^H) = {pearson(A_op, A_aw):.3f}, Spearman = {spearman(A_op, A_aw):.3f}; Pearson(OPER adv, raw G^H) = {pearson(A_op, gH):.3f}; Pearson(OPER V(s), G^H) = {pearson(V_op, gH):.3f}; Pearson(OPER TD target, G^H) = {pearson(Q_op, gH):.3f}"
    )

    # ---- per bin
    gmean = gH.mean()
    bin_mean = np.array([gH[bins == k].mean() for k in range(K)])
    low = bin_mean < gmean - a.region_margin
    high = bin_mean > gmean + a.region_margin
    data_mass = np.array([(bins == k).mean() for k in range(K)])

    def mass(w):
        return np.array([w[bins == k].sum() for k in range(K)]) / w.sum()

    m_aw, m_op, m_opu = mass(w_aw), mass(w_op), mass(w_op_used)
    m_awg = mass(w_awg) if w_awg is not None else np.full(K, np.nan)
    lines.append("\n## Per progress bin\n")
    lines.append(
        "| bin | region | data mass % | AW phase w-mass % | AW global w-mass % | OPER raw w-mass % | OPER as-used w-mass % | mean OPER w (as used) | mean OPER adv | mean AW A^H | mean G^H |\n|---|---|---|---|---|---|---|---|---|---|---|"
    )
    csv = [
        "bin,region,data_mass,aw_phase_mass,aw_global_mass,oper_raw_mass,oper_used_mass,oper_used_w_mean,oper_adv_mean,aw_adv_mean,gH_mean"
    ]
    for k in range(K):
        m = bins == k
        reg = "low" if low[k] else ("high" if high[k] else "")
        lines.append(
            f"| {k} | {reg} | {100 * data_mass[k]:.2f} | {100 * m_aw[k]:.2f} | {100 * m_awg[k]:.2f} | {100 * m_op[k]:.2f} | {100 * m_opu[k]:.2f} | {w_op_used[m].mean():.3f} | {A_op[m].mean():+.4f} | {A_aw[m].mean():+.3f} | {bin_mean[k]:.3f} |"
        )
        csv.append(
            f"{k},{reg},{data_mass[k]:.6f},{m_aw[k]:.6f},{m_awg[k]:.6f},{m_op[k]:.6f},{m_opu[k]:.6f},{w_op_used[m].mean():.6f},{A_op[m].mean():.6f},{A_aw[m].mean():.6f},{bin_mean[k]:.6f}"
        )
    (out / "per_bin.csv").write_text("\n".join(csv) + "\n")
    wall = np.isin(bins, a.wall_bins)
    lowm = np.isin(bins, np.flatnonzero(low))
    highm = np.isin(bins, np.flatnonzero(high))
    lines.append("\n## Region weight mass (share of total weight; first column = data share)\n")
    lines.append("| region | data | AW phase | AW global | OPER-A raw | OPER-A as used |\n|---|---|---|---|---|---|")
    for name, msk in [
        (f"wall bins {a.wall_bins}", wall),
        (f"low-return bins {np.flatnonzero(low).tolist()}", lowm),
        (f"high-return bins {np.flatnonzero(high).tolist()}", highm),
    ]:
        lines.append(
            f"| {name} | {100 * msk.mean():.1f}% | {100 * w_aw[msk].sum() / w_aw.sum():.1f}% | "
            + (f"{100 * w_awg[msk].sum() / w_awg.sum():.1f}%" if w_awg is not None else "n/a")
            + f" | {100 * w_op[msk].sum() / w_op.sum():.1f}% | {100 * w_op_used[msk].sum() / w_op_used.sum():.1f}% |"
        )

    # how much of OPER's weight variance is explained by the bin (progress) alone
    def r2_bin(w):
        mu = np.array([w[bins == k].mean() for k in range(K)])[bins]
        return float(1 - ((w - mu) ** 2).sum() / ((w - w.mean()) ** 2).sum())

    lines.append(
        f"\nShare of weight variance explained by the progress bin alone (R^2 of per-bin means): AW phase {r2_bin(w_aw):.3f}, AW global {r2_bin(w_awg) if w_awg is not None else float('nan'):.3f}, OPER-A raw {r2_bin(w_op):.3f}, OPER-A as used {r2_bin(w_op_used):.3f}"
    )
    lines.append(
        f"Phase dependence of the OPER advantage: R^2 of per-bin means = {r2_bin(A_op):.3f} (AW A^H by construction 0: {r2_bin(A_aw):.3f})"
    )

    # ---- pseudo-sidecar for the post-H validity script
    sigma = float(A_op.std())
    pseudo = out / "oper_a_pseudo_sidecar.npz"
    np.savez(
        pseudo,
        weight=w_op.astype(np.float32),
        advantage=A_op.astype(np.float32),
        gH=Q_op.astype(np.float32),
        phase_bin=bins.astype(np.int16),
        beta=sigma,
        sigma=sigma,
        gamma=float(cfg.get("discount", 0.99)),
        H=a.post_h_H,
        w_max=float(w_op.max()),
        n=n,
        ess_frac=ess_frac(w_op),
        clip_frac=0.0,
        h5=a.h5,
        rhash=rh,
    )
    lines.append(
        f"\nWrote `{pseudo}` (advantage = OPER-A TD(0) advantage, gH = OPER TD target, H={a.post_h_H} for the validation window) for scripts/aw_post_h_validity.py --npz. Only its within-bin Panel A and the bin-FE logistic are meaningful for OPER-A: gH here is a 1-step TD target, not a 50-step return, so Panel B's raw-G^H curve does not apply."
    )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
