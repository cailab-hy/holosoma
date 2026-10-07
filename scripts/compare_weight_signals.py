#!/usr/bin/env python3
# ruff: noqa: E501
"""Compare per-transition weight tables (PRe / AW, frozen V_H, ODPR-A) on one dataset.

For each weighting: distribution statistics, weight mass and spread per progress bin, what the weight
correlates with, and whether it down-weights transitions that are about to terminate by bad tracking
(mean weight and AUC by time-to-failure, pooled and within progress bin).

  python scripts/compare_weight_signals.py --h5 <h5> --weights PRe=<npz> V_H=<npz> ODPR-A=<npz> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def ess(w):
    return float(w.sum() ** 2 / (len(w) * (w * w).sum()))


def auc(score, positive):
    """P(score of a random positive > score of a random negative), ties = 0.5 (Mann-Whitney)."""
    n_pos = int(positive.sum())
    n_neg = len(score) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), np.float64)
    s_sorted = score[order]
    # average ranks for ties
    i = 0
    r = np.arange(1, len(score) + 1, dtype=np.float64)
    boundaries = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1], True])
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        r[a:b] = 0.5 * (a + 1 + b)
    ranks[order] = r
    del i
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binned_auc(score, positive, bins, n_bins):
    vals, cnt = [], []
    for k in range(n_bins):
        m = bins == k
        a = auc(score[m], positive[m])
        if np.isfinite(a):
            vals.append(a)
            cnt.append(int(m.sum()))
    return float(np.average(vals, weights=cnt)), vals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--weights", nargs="+", required=True, help="NAME=path.npz ...")
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--H", type=int, default=50)
    ap.add_argument("--out-dir", default="validation/results/weight_comparison")
    a = ap.parse_args()

    with h5py.File(a.h5, "r") as f:
        n = int(f.attrs.get("num_samples", f["rewards"].shape[0]))
        g = lambda k: f[k][:n].reshape(-1)  # noqa: E731
        r = g("rewards").astype(np.float64)
        phase = g("motion_phase").astype(np.float64)
        eid = g("episode_id")
        bad = g("next_done_bad_tracking").astype(bool)
        mends = g("next_done_motion_ends").astype(bool)
    bins = np.clip((phase * a.n_bins).astype(int), 0, a.n_bins - 1)
    change = np.flatnonzero(np.r_[eid[1:] != eid[:-1], True])  # inclusive episode ends
    starts = np.r_[0, change[:-1] + 1]
    ep = np.repeat(np.arange(len(starts)), change - starts + 1)
    end_row = change[ep]
    steps_to_end = end_row - np.arange(n)          # 0 = terminal transition
    ep_bad = bad[change][ep]                        # episode ends by bad tracking
    ep_ok = mends[change][ep]                       # episode completes the motion

    W = {}
    extra = {}
    for item in a.weights:
        name, path = item.split("=", 1)
        z = np.load(path, allow_pickle=False)
        assert int(z["n"]) == n, (name, int(z["n"]), n)
        W[name] = z["weight"].astype(np.float64)
        if "gH" in z:
            extra["gH"] = z["gH"].astype(np.float64)
    names = list(W)
    gH = extra.get("gH")
    out = {"h5": Path(a.h5).name, "N": n, "episodes": int(len(starts)), "names": names}
    L = [f"# Weight signal comparison: {', '.join(names)}\n", f"dataset `{Path(a.h5).name}`, N={n:,}, episodes={len(starts):,} "
         f"(bad tracking {100 * bad[change].mean():.1f}%, motion completed {100 * mends[change].mean():.1f}%)\n"]

    # 1. distribution
    L += ["## 1. Distribution (all weights have mean 1)\n", "| weights | ESS/N | std | min | p1 | p10 | p50 | p90 | p99 | max | share w<0.5 | share w>2 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out["distribution"] = {}
    for k in names:
        w = W[k]; q = np.percentile(w, [1, 10, 50, 90, 99])
        out["distribution"][k] = dict(ess=ess(w), std=float(w.std()), min=float(w.min()), max=float(w.max()), q=q.tolist(), lt05=float((w < .5).mean()), gt2=float((w > 2).mean()))
        L.append(f"| {k} | {ess(w):.3f} | {w.std():.3f} | {w.min():.3f} | {q[0]:.2f} | {q[1]:.2f} | {q[2]:.2f} | {q[3]:.2f} | {q[4]:.2f} | {w.max():.1f} | {100 * (w < .5).mean():.1f}% | {100 * (w > 2).mean():.1f}% |")

    # 2. agreement
    L += ["\n## 2. Agreement between weightings\n", "| pair | Pearson | Spearman | within-bin Spearman | top-10% Jaccard | bottom-10% Jaccard |", "|---|---|---|---|---|---|"]
    rank = {k: np.argsort(np.argsort(W[k])).astype(np.float64) for k in names}
    def wb_sp(x, y):
        v, c = [], []
        for b in range(a.n_bins):
            m = bins == b
            rx = np.argsort(np.argsort(x[m])).astype(float); ry = np.argsort(np.argsort(y[m])).astype(float)
            v.append(np.corrcoef(rx, ry)[0, 1]); c.append(m.sum())
        return float(np.average(v, weights=c))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = W[names[i]], W[names[j]]
            tx, ty = x >= np.quantile(x, .9), y >= np.quantile(y, .9)
            bx, by = x <= np.quantile(x, .1), y <= np.quantile(y, .1)
            L.append(f"| {names[i]} vs {names[j]} | {np.corrcoef(x, y)[0, 1]:.3f} | {np.corrcoef(rank[names[i]], rank[names[j]])[0, 1]:.3f} | {wb_sp(x, y):.3f} | {(tx & ty).sum() / (tx | ty).sum():.3f} | {(bx & by).sum() / (bx | by).sum():.3f} |")
    L.append("\n(top/bottom-10% Jaccard of two independent weightings would be 0.053.)")

    # 3. per bin
    hazard = np.array([bad[bins == b].mean() for b in range(a.n_bins)])
    L += ["\n## 3. Per progress bin (K=20)\n", "| bin | data % | bad-tracking hazard %/step | " + " | ".join(f"{k} mass %" for k in names) + " | " + " | ".join(f"{k} mean w" for k in names) + " | " + " | ".join(f"{k} std w" for k in names) + " |", "|---|---|---|" + "---|" * (3 * len(names))]
    out["per_bin"] = {k: {"mass": [], "mean": [], "std": []} for k in names}
    for b in range(a.n_bins):
        m = bins == b
        row = f"| {b} | {100 * m.mean():.2f} | {100 * hazard[b]:.2f} | "
        row += " | ".join(f"{100 * W[k][m].sum() / n:.2f}" for k in names) + " | " + " | ".join(f"{W[k][m].mean():.3f}" for k in names) + " | " + " | ".join(f"{W[k][m].std():.3f}" for k in names) + " |"
        L.append(row)
        for k in names:
            out["per_bin"][k]["mass"].append(float(W[k][m].sum() / n)); out["per_bin"][k]["mean"].append(float(W[k][m].mean())); out["per_bin"][k]["std"].append(float(W[k][m].std()))
    L.append("")
    L.append("| weights | Var(w) explained by bin (R2) | corr(bin mean w, bin hazard) | weight mass on the 4 highest-hazard bins (data share " + f"{100 * np.isin(bins, np.argsort(-hazard)[:4]).mean():.1f}%) |")
    L.append("|---|---|---|---|")
    top4 = np.isin(bins, np.argsort(-hazard)[:4])
    for k in names:
        w = W[k]; bm = np.array(out["per_bin"][k]["mean"])
        r2b = 1 - sum(((w[bins == b] - w[bins == b].mean()) ** 2).sum() for b in range(a.n_bins)) / ((w - w.mean()) ** 2).sum()
        L.append(f"| {k} | {r2b:.4f} | {np.corrcoef(bm, hazard)[0, 1]:+.3f} | {100 * w[top4].sum() / n:.2f}% |")
    L.append(f"\nHighest-hazard bins: {np.argsort(-hazard)[:4].tolist()}.")

    # 4. what the weight tracks
    L += ["\n## 4. What each weight correlates with (Spearman)\n", "| weights | reward r_t | G^H (H-step return) | steps until episode end | episode completes motion (AUC) | within-bin: G^H | within-bin: episode completes (AUC) |", "|---|---|---|---|---|---|---|"]
    def sp(x, y):
        return float(np.corrcoef(np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float))[0, 1])
    out["tracks"] = {}
    for k in names:
        w = W[k]
        row = dict(r=sp(w, r), gH=sp(w, gH) if gH is not None else float("nan"), steps=sp(w, steps_to_end.astype(float)), auc_ok=auc(w, ep_ok),
                   wb_gH=wb_sp(w, gH) if gH is not None else float("nan"), wb_auc_ok=binned_auc(w, ep_ok, bins, a.n_bins)[0])
        out["tracks"][k] = row
        L.append(f"| {k} | {row['r']:+.3f} | {row['gH']:+.3f} | {row['steps']:+.3f} | {row['auc_ok']:.3f} | {row['wb_gH']:+.3f} | {row['wb_auc_ok']:.3f} |")
    L.append("\nAUC = probability that a transition from an episode that completes the motion gets a higher weight than one from an episode that ends by bad tracking (0.5 = no information).")

    # 5. imminent failure
    L += ["\n## 5. Transitions that are about to terminate by bad tracking\n",
          "`fail<=K`: the episode ends by bad tracking within the next K steps (K counted from the transition). Mean weight of those rows (1.0 = dataset mean), and AUC of the weight for separating them from all other rows "
          "(AUC < 0.5 means the weighting DOWN-weights soon-to-fail transitions; within-bin AUC removes the effect of where in the motion they are).\n",
          "| K | share of rows | " + " | ".join(f"{k} mean w" for k in names) + " | " + " | ".join(f"{k} AUC" for k in names) + " | " + " | ".join(f"{k} within-bin AUC" for k in names) + " |", "|---|---|" + "---|" * (3 * len(names))]
    out["imminent"] = {}
    for K in (1, 5, 10, 25, 50, 100, 200):
        y = ep_bad & (steps_to_end < K)
        row = f"| {K} | {100 * y.mean():.1f}% | " + " | ".join(f"{W[k][y].mean():.3f}" for k in names)
        aucs = {k: auc(W[k], y) for k in names}; wba = {k: binned_auc(W[k], y, bins, a.n_bins)[0] for k in names}
        row += " | " + " | ".join(f"{aucs[k]:.3f}" for k in names) + " | " + " | ".join(f"{wba[k]:.3f}" for k in names) + " |"
        L.append(row); out["imminent"][K] = dict(share=float(y.mean()), mean_w={k: float(W[k][y].mean()) for k in names}, auc=aucs, within_bin_auc=wba)

    # 6. post-H failure (outside the window that built the H-step return)
    alive = steps_to_end >= a.H
    L += [f"\n## 6. Failure AFTER the H={a.H} window (rows whose episode is still alive at t+H; {100 * alive.mean():.1f}% of rows)\n",
          "Outcome: bad tracking within [t+H, t+H+M). This cannot be read off the H-step return itself.\n",
          "| M | failure rate | " + " | ".join(f"{k} mean w (fail / survive)" for k in names) + " | " + " | ".join(f"{k} within-bin AUC" for k in names) + " |", "|---|---|" + "---|" * (2 * len(names))]
    out["post_h"] = {}
    for M in (50, 100, 150):
        y = ep_bad & (steps_to_end < a.H + M)
        ya, ba_ = y[alive], bins[alive]
        row = f"| {M} | {100 * ya.mean():.1f}% | " + " | ".join(f"{W[k][alive][ya].mean():.3f} / {W[k][alive][~ya].mean():.3f}" for k in names)
        wba = {k: binned_auc(W[k][alive], ya, ba_, a.n_bins)[0] for k in names}
        row += " | " + " | ".join(f"{wba[k]:.3f}" for k in names) + " |"
        L.append(row); out["post_h"][M] = dict(rate=float(ya.mean()), within_bin_auc=wba, mean_w_fail={k: float(W[k][alive][ya].mean()) for k in names}, mean_w_survive={k: float(W[k][alive][~ya].mean()) for k in names})

    # 7. weight mass by episode outcome
    L += ["\n## 7. Where the weight mass goes\n", "| weights | mass on rows of completing episodes (data share " + f"{100 * ep_ok.mean():.1f}%) | mass on the last 25 steps before a bad-tracking end (data share {100 * (ep_bad & (steps_to_end < 25)).mean():.1f}%) |", "|---|---|---|"]
    for k in names:
        w = W[k]
        L.append(f"| {k} | {100 * w[ep_ok].sum() / n:.1f}% | {100 * w[ep_bad & (steps_to_end < 25)].sum() / n:.1f}% |")

    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    (od / "summary.md").write_text("\n".join(L) + "\n")
    (od / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
