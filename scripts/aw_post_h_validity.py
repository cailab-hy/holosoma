#!/usr/bin/env python3
# ruff: noqa: N803, E501, ICN001, RET504, PERF401
"""Post-H predictive validity of the progress-relative advantage A^H = G^H - b(kappa).

Question: within a fixed task-progress bin, do transitions with a larger progress-relative
H-step return go on to fail (bad tracking) less often *after* the H-step window that A^H itself
was computed from?  This validates A^H as a transition-quality proxy; it does not claim A^H is
the true advantage.

Sample and outcome (H = return horizon of the sidecar, M = validation window length)
---------------------------------------------------------------------------------
For a transition at row t of an episode whose last recorded row is e (inclusive):

* enters the validation window  <=>  e - t >= H       (the episode is still alive at row t+H)
  Transitions whose episode ends before t+H are excluded whatever the reason: a bad-tracking end
  there is already inside the return that built A^H, and a motion end there leaves no post-H
  trajectory to validate on.
* validation window = rows [t+H, t+H+M).  First termination inside it decides the outcome:
    bad_tracking  -> Y = 1
    motion_ends   -> Y = 0   (the motion completed; no failure can follow, so this is an observed
                              negative, NOT a censored row -- treating it as censored would drop
                              the surviving high-A rows and inflate the effect)
    timeout / segment_ends / truncation / incomplete episode -> censored, row excluded
  No termination inside the window -> Y = 0 (window fully observed).

Analyses
--------
Panel A  (primary): within each progress bin, quartiles of A^H (Q1 lowest) among the analysed rows;
         P(Y=1 | Q_j, bin) per bin; aggregated across bins by row count (also unweighted).
         Because b(kappa) is constant inside a bin, the within-bin quartiles are identical for A^H
         and for raw G^H: this panel tests "G^H is informative after conditioning on progress", it
         cannot distinguish the phase baseline from a global one.
Panel B  (why progress conditioning): pooled quartiles over all analysed rows of raw G^H (whose
         ordering equals that of any dataset-wide scalar baseline) versus A^H, with each quartile's
         progress composition (mean phase, prepend / motion / append share, hazard-wall share).
         Caveat: raw G^H of rows near the motion end is a *truncated* sum (fewer than H rewards),
         so part of raw-Q1's late-phase excess is mechanical truncation, not difficulty.
Logistic regression with progress-bin fixed effects: logit P(Y=1) = beta_A * A/sigma(A) + gamma_bin.
         beta_A is identical for A^H and G^H (b(kappa) is absorbed by the fixed effects). 95% CI from
         an episode-level cluster bootstrap (episodes resampled with replacement).

Usage
-----
python scripts/aw_post_h_validity.py --h5 offline_data/<dataset>.h5 --npz <sidecar.npz> \
    --M 100 --extra-M 50 150 --out-dir validation/results/post_h_validity
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Progress-bin layout, used only for the composition panel (defaults: G1 largebox, 20 bins).
DEFAULT_LAYOUT = {
    "prepend": [0, 1, 2, 3],
    "append": [16, 17, 18, 19],
    "wall": [4, 5, 12, 13],
    "wall_name": "hazard wall",
}


# ----------------------------------------------------------------------------- loading
def load_h5(path: Path) -> dict[str, np.ndarray]:
    keys = [
        "episode_id",
        "motion_phase",
        "rewards",
        "next_done_bad_tracking",
        "next_done_motion_ends",
        "next_done_timeout",
        "next_done_segment_ends",
        "truncations",
        "episode_data_complete",
    ]
    with h5py.File(path, "r") as f:
        n = int(f.attrs.get("num_samples", f["episode_id"].shape[0]))
        d = {k: f[k][:n] for k in keys if k in f}
    return d


def rhash(rewards: np.ndarray) -> str:
    """Same pairing hash as scripts/aw_precompute_weights.py (float64, first+last 1000 rewards)."""
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    return hashlib.sha256(
        np.ascontiguousarray(r[:1000]).tobytes() + np.ascontiguousarray(r[-1000:]).tobytes()
    ).hexdigest()[:16]


def episode_blocks(episode_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    change = np.flatnonzero(np.diff(episode_id) != 0) + 1
    starts = np.r_[0, change]
    ends = np.r_[change - 1, episode_id.size - 1]
    return starts, ends


# ----------------------------------------------------------------------------- outcome
def post_h_outcome(d: dict[str, np.ndarray], H: int, M: int) -> dict[str, np.ndarray]:
    """Per-row: entered (bool), y (0/1), censored (bool), plus exclusion-reason counts."""
    ep = d["episode_id"]
    n = ep.size
    starts, ends = episode_blocks(ep)
    block_len = ends - starts + 1
    row_block = np.repeat(np.arange(starts.size), block_len)
    end_idx = ends[row_block]
    rows = np.arange(n)
    remaining = end_idx - rows  # rows after t up to and including the last row

    bad = d["next_done_bad_tracking"].astype(bool)
    mend = d["next_done_motion_ends"].astype(bool)
    tmo = d.get("next_done_timeout", np.zeros(n, np.uint8)).astype(bool)
    seg = d.get("next_done_segment_ends", np.zeros(n, np.uint8)).astype(bool)
    trunc = d.get("truncations", np.zeros(n, np.uint8)).astype(bool)
    complete = d.get("episode_data_complete", np.ones(n, np.uint8)).astype(bool)
    if bad.sum() != bad[ends].sum() or mend.sum() != mend[ends].sum():
        raise ValueError("bad_tracking / motion_ends flags found inside episodes; expected only at episode ends")

    end_bad = bad[end_idx]
    end_mend = mend[end_idx]
    end_complete = complete[end_idx]
    end_other = ~(end_bad | end_mend) | tmo[end_idx] | seg[end_idx] | trunc[end_idx] | ~end_complete
    # priority: a bad-tracking / motion-end flag on a complete episode is a genuine terminal
    end_bad = end_bad & end_complete
    end_mend = end_mend & end_complete & ~end_bad
    end_other = ~(end_bad | end_mend)

    entered = remaining >= H
    in_window = entered & (remaining < H + M)  # episode terminates inside [t+H, t+H+M)
    y = np.zeros(n, np.int8)
    censored = np.zeros(n, bool)
    y[in_window & end_bad] = 1
    censored[in_window & end_other] = True
    # rows with remaining >= H+M: window fully observed, y = 0; in_window & end_mend: y = 0
    analysed = entered & ~censored

    reasons = {
        "rows_total": int(n),
        "episodes": int(starts.size),
        "excluded_episode_ends_before_t+H": int((~entered).sum()),
        "  of which end reason bad_tracking": int(((~entered) & end_bad).sum()),
        "  of which end reason motion_ends": int(((~entered) & end_mend).sum()),
        "  of which end reason other": int(((~entered) & end_other).sum()),
        "censored_in_window (timeout/segment/truncation/incomplete)": int(censored.sum()),
        "analysed": int(analysed.sum()),
        "  y=1 (bad tracking in window)": int((analysed & (y == 1)).sum()),
        "  y=0 via motion_ends in window": int((analysed & in_window & end_mend).sum()),
        "  y=0 via window fully observed": int((analysed & (remaining >= H + M)).sum()),
    }
    return {"analysed": analysed, "y": y, "row_block": row_block, "reasons": reasons}


# ----------------------------------------------------------------------------- quartiles
def quartile_within_groups(x: np.ndarray, group: np.ndarray, n_groups: int) -> np.ndarray:
    q = np.full(x.size, -1, np.int8)
    for g in range(n_groups):
        m = group == g
        if m.sum() < 4:
            continue
        edges = np.quantile(x[m], [0.25, 0.5, 0.75])
        q[m] = np.searchsorted(edges, x[m], side="right")
    return q


def rate_table(y: np.ndarray, q: np.ndarray, group: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """rates[n_groups, 4], counts[n_groups, 4]."""
    rates = np.full((n_groups, 4), np.nan)
    counts = np.zeros((n_groups, 4), int)
    for g in range(n_groups):
        for j in range(4):
            m = (group == g) & (q == j)
            counts[g, j] = m.sum()
            if m.any():
                rates[g, j] = y[m].mean()
    return rates, counts


def aggregate(rates: np.ndarray, counts: np.ndarray, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    ok = counts.sum(1) >= min_count
    w = counts[ok].astype(float)
    weighted = np.nansum(rates[ok] * w, 0) / w.sum(0)
    unweighted = np.nanmean(rates[ok], 0)
    return weighted, unweighted


# ----------------------------------------------------------------------------- logistic (IRLS)
def logistic_fit(
    X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, iters: int = 25, tol: float = 1e-8
) -> np.ndarray:
    n, p = X.shape
    w = np.ones(n) if w is None else w
    beta = np.zeros(p)
    for _ in range(iters):
        z = X @ beta
        mu = 1.0 / (1.0 + np.exp(-z))
        s = w * mu * (1.0 - mu)
        grad = X.T @ (w * (y - mu))
        hess = (X * s[:, None]).T @ X
        hess[np.diag_indices(p)] += 1e-8
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.abs(step).max() < tol:
            break
    return beta


def bin_fe_logistic(
    a_std: np.ndarray, bins: np.ndarray, y: np.ndarray, row_block: np.ndarray, n_bins: int, n_boot: int, seed: int
) -> dict:
    present = np.unique(bins)
    col = {b: i for i, b in enumerate(present)}
    X = np.zeros((a_std.size, 1 + present.size), np.float64)
    X[:, 0] = a_std
    X[np.arange(a_std.size), 1 + np.vectorize(col.get)(bins)] = 1.0
    yf = y.astype(np.float64)
    beta = logistic_fit(X, yf)
    rng = np.random.default_rng(seed)
    n_ep = row_block.max() + 1
    boots = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_ep, n_ep), minlength=n_ep).astype(float)
        boots.append(logistic_fit(X, yf, w=counts[row_block])[0])
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "beta_A_per_sigma": float(beta[0]),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "odds_ratio_per_sigma": float(np.exp(beta[0])),
        "n_rows": int(a_std.size),
        "n_episodes": int(n_ep),
        "n_boot": int(n_boot),
    }


def cluster_bootstrap_rates(
    y: np.ndarray,
    q: np.ndarray,
    group: np.ndarray,
    row_block: np.ndarray,
    n_groups: int,
    min_count: int,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """95% CI of the count-weighted aggregate P(Y=1|Q_j) over episode resamples -> [4, 2]."""
    rng = np.random.default_rng(seed)
    n_ep = row_block.max() + 1
    counts_full = np.zeros((n_groups, 4), int)
    for g in range(n_groups):
        for j in range(4):
            counts_full[g, j] = ((group == g) & (q == j)).sum()
    ok_groups = counts_full.sum(1) >= min_count
    keep = ok_groups[group] & (q >= 0)
    yk, qk, bk = y[keep].astype(float), q[keep], row_block[keep]
    out = np.zeros((n_boot, 4))
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, n_ep, n_ep), minlength=n_ep).astype(float)[bk]
        for j in range(4):
            m = qk == j
            out[i, j] = (w[m] * yk[m]).sum() / w[m].sum()
    return np.percentile(out, [2.5, 97.5], axis=0).T


# ----------------------------------------------------------------------------- plots
def plot_panel_a(agg_w: np.ndarray, ci: np.ndarray, agg_u: np.ndarray, H: int, M: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4))
    x = np.arange(4)
    ax.bar(x, 100 * agg_w, color="#4C72B0", width=0.6, label="count-weighted over progress bins")
    ax.errorbar(x, 100 * agg_w, yerr=100 * np.abs(ci.T - agg_w), fmt="none", ecolor="k", capsize=4, lw=1)
    ax.plot(x, 100 * agg_u, "o", color="#DD8452", ms=6, label="unweighted mean over bins")
    ax.set_xticks(x, ["Q1\n(lowest A)", "Q2", "Q3", "Q4\n(highest A)"])
    ax.set_xlabel(r"within-progress-bin quartile of $A^H = G^H - b(\kappa)$")
    ax.set_ylabel(f"P(bad tracking in [t+{H}, t+{H + M}))  (%)")
    ax.set_title("Post-H predictive validity within progress bins", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_panel_b(
    rate_g: np.ndarray,
    rate_a: np.ndarray,
    comp_g: np.ndarray,
    comp_a: np.ndarray,
    mean_phase_g: np.ndarray,
    mean_phase_a: np.ndarray,
    H: int,
    M: int,
    path: Path,
    layout: dict,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(4)
    ax = axes[0]
    ax.plot(x, 100 * rate_g, "s-", color="#8C8C8C", label=r"raw $G^H$ (= any dataset-wide scalar baseline)")
    ax.plot(x, 100 * rate_a, "o-", color="#4C72B0", label=r"$A^H = G^H - b(\kappa)$ (progress-relative)")
    ax.set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("pooled quartile over all analysed transitions")
    ax.set_ylabel(f"P(bad tracking in [t+{H}, t+{H + M}))  (%)")
    ax.set_title("Pooled quartiles: failure rate", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    def fmt(b):
        return ",".join(map(str, b)) if b else "none"

    labels = [
        f"prepend (bins {fmt(layout['prepend'])})",
        f"motion, {layout['wall_name']} ({fmt(layout['wall'])})",
        "motion, other",
        f"append ({fmt(layout['append'])})",
    ]
    colors = ["#C9C9C9", "#C44E52", "#55A868", "#8172B2"]
    for ax, comp, mp, name in [
        (axes[1], comp_g, mean_phase_g, r"raw $G^H$"),
        (axes[2], comp_a, mean_phase_a, r"$A^H$"),
    ]:
        bottom = np.zeros(4)
        for k in range(4):
            ax.bar(x, 100 * comp[:, k], bottom=100 * bottom, color=colors[k], width=0.6, label=labels[k])
            bottom += comp[:, k]
        for j in range(4):
            ax.text(j, 102, f"phase {mp[j]:.2f}", ha="center", fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
        ax.set_ylabel("share of quartile (%)")
        ax.set_title(f"Progress composition of {name} quartiles", fontsize=10)
    axes[2].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmap(rates: np.ndarray, counts: np.ndarray, min_count: int, H: int, M: int, path: Path) -> None:
    n_bins = rates.shape[0]
    show = 100 * rates.copy()
    show[counts < min_count] = np.nan
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(show, aspect="auto", cmap="viridis")
    ax.set_xticks(range(4), ["Q1", "Q2", "Q3", "Q4"])
    ax.set_yticks(range(n_bins), [f"{b}" for b in range(n_bins)])
    ax.set_ylabel("progress bin")
    ax.set_xlabel(r"within-bin quartile of $A^H$")
    ax.set_title(f"P(bad tracking in [t+{H}, t+{H + M})) (%), grey = < {min_count} rows", fontsize=9)
    for b in range(n_bins):
        for j in range(4):
            if np.isfinite(show[b, j]):
                ax.text(j, b, f"{show[b, j]:.1f}\n(n={counts[b, j]})", ha="center", va="center", fontsize=6, color="w")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def composition(bins: np.ndarray, phase: np.ndarray, q: np.ndarray, layout: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per quartile: [prepend, wall, motion other, append] share and mean phase."""
    prepend, append, wall = (np.isin(bins, layout[k]) for k in ("prepend", "append", "wall"))
    motion_other = ~(prepend | append | wall)
    comp = np.zeros((4, 4))
    mean_phase = np.zeros(4)
    for j in range(4):
        m = q == j
        comp[j] = [prepend[m].mean(), wall[m].mean(), motion_other[m].mean(), append[m].mean()]
        mean_phase[j] = phase[m].mean()
    return comp, mean_phase


def run_one(
    d: dict,
    sc: dict,
    H: int,
    M: int,
    n_bins: int,
    min_count: int,
    n_boot: int,
    seed: int,
    out_dir: Path,
    make_plots: bool,
    layout: dict,
) -> dict:
    oc = post_h_outcome(d, H, M)
    an = oc["analysed"]
    y = oc["y"][an]
    A = sc["advantage"][an].astype(np.float64)
    G = sc["gH"][an].astype(np.float64)
    bins = sc["phase_bin"][an].astype(int)
    phase = d["motion_phase"][an].astype(np.float64)
    rb = oc["row_block"][an]
    rb = np.unique(rb, return_inverse=True)[1]  # compact episode index for bootstrap

    # Panel A: within-bin quartiles (identical for A and G)
    q_within = quartile_within_groups(A, bins, n_bins)
    rates, counts = rate_table(y, q_within, bins, n_bins)
    agg_w, agg_u = aggregate(rates, counts, min_count)
    ci = cluster_bootstrap_rates(y, q_within, bins, rb, n_bins, min_count, n_boot, seed)

    # Panel B: pooled quartiles
    q_g = quartile_within_groups(G, np.zeros(G.size, int), 1)
    q_a = quartile_within_groups(A, np.zeros(A.size, int), 1)
    rate_g = np.array([y[q_g == j].mean() for j in range(4)])
    rate_a = np.array([y[q_a == j].mean() for j in range(4)])
    comp_g, mp_g = composition(bins, phase, q_g, layout)
    comp_a, mp_a = composition(bins, phase, q_a, layout)

    # logistic with bin fixed effects
    sigma = float(sc["sigma"])
    logit = bin_fe_logistic(A / sigma, bins, y, rb, n_bins, n_boot, seed)

    tag = f"H{H}_M{M}"
    if make_plots:
        plot_panel_a(agg_w, ci, agg_u, H, M, out_dir / f"panel_a_within_bin_{tag}.png")
        plot_panel_b(rate_g, rate_a, comp_g, comp_a, mp_g, mp_a, H, M, out_dir / f"panel_b_pooled_{tag}.png", layout)
        plot_heatmap(rates, counts, min_count, H, M, out_dir / f"heatmap_bin_quartile_{tag}.png")
    with (out_dir / f"bin_quartile_rates_{tag}.csv").open("w") as f:
        f.write("bin,quartile,n,p_bad_tracking\n")
        for b in range(n_bins):
            for j in range(4):
                f.write(f"{b},{j + 1},{counts[b, j]},{rates[b, j]:.6f}\n")
    with (out_dir / f"pooled_quartiles_{tag}.csv").open("w") as f:
        f.write("signal,quartile,p_bad_tracking,mean_phase,share_prepend,share_wall,share_motion_other,share_append\n")
        for name, r, c, mp in [("raw_GH", rate_g, comp_g, mp_g), ("A_phase", rate_a, comp_a, mp_a)]:
            for j in range(4):
                f.write(f"{name},{j + 1},{r[j]:.6f},{mp[j]:.4f}," + ",".join(f"{v:.4f}" for v in c[j]) + "\n")
    return {
        "H": H,
        "M": M,
        "reasons": oc["reasons"],
        "within_bin_weighted": agg_w.tolist(),
        "within_bin_weighted_ci95": ci.tolist(),
        "within_bin_unweighted": agg_u.tolist(),
        "bins_used": int((counts.sum(1) >= min_count).sum()),
        "pooled_rate_rawGH": rate_g.tolist(),
        "pooled_rate_A": rate_a.tolist(),
        "pooled_meanphase_rawGH": mp_g.tolist(),
        "pooled_meanphase_A": mp_a.tolist(),
        "pooled_wallshare_rawGH": comp_g[:, 1].tolist(),
        "pooled_wallshare_A": comp_a[:, 1].tolist(),
        "pooled_appendshare_rawGH": comp_g[:, 3].tolist(),
        "pooled_appendshare_A": comp_a[:, 3].tolist(),
        "logistic": logit,
        "monotone_Q1_to_Q4": bool(np.all(np.diff(agg_w) < 0)),
    }


def fmt_row(vals, ci=None):
    if ci is None:
        return " | ".join(f"{100 * v:.2f}" for v in vals)
    return " | ".join(f"{100 * v:.2f} [{100 * lo:.2f}, {100 * hi:.2f}]" for v, (lo, hi) in zip(vals, ci))


def write_summary(results: list[dict], meta: dict, out_dir: Path) -> None:
    r0 = results[0]
    L = []
    L.append("# Post-H predictive validity of A^H = G^H - b(kappa)\n")
    L.append(
        f"Dataset `{meta['h5']}`, sidecar `{meta['npz']}` (H={meta['H']}, {meta['n_bins']} progress bins, "
        f"sigma(A)={meta['sigma']:.4f}, rhash verified). Primary window M={r0['M']}; min rows per bin for the aggregate: {meta['min_count']}; "
        f"episode-cluster bootstrap with {meta['n_boot']} resamples.\n"
    )
    L.append("## Sample construction (primary M)\n")
    L.append("| item | rows |\n|---|---|")
    for k, v in r0["reasons"].items():
        L.append(f"| {k} | {v:,} |")
    L.append("")
    L.append("## Panel A: within-progress-bin quartiles of A^H -> P(bad tracking in post-H window) (%)\n")
    L.append("| M | Q1 (lowest A) | Q2 | Q3 | Q4 (highest A) | monotone | bins used |\n|---|---|---|---|---|---|---|")
    for r in results:
        cells = fmt_row(r["within_bin_weighted"], r["within_bin_weighted_ci95"]).split(" | ")
        L.append(
            f"| {r['M']} | "
            + " | ".join(cells)
            + f" | {'yes' if r['monotone_Q1_to_Q4'] else 'no'} | {r['bins_used']} |"
        )
    L.append(
        "\nCount-weighted aggregate over bins with 95% episode-cluster bootstrap CI; unweighted mean over bins: "
        + "; ".join(f"M={r['M']}: " + fmt_row(r["within_bin_unweighted"]) for r in results)
        + ".\n"
    )
    L.append("## Logistic regression with progress-bin fixed effects: logit P(Y=1) = beta_A * A/sigma + gamma_bin\n")
    L.append(
        "| M | beta_A (per 1 sigma of A) | 95% cluster CI | odds ratio per sigma | rows | episodes |\n|---|---|---|---|---|---|"
    )
    for r in results:
        g = r["logistic"]
        L.append(
            f"| {r['M']} | {g['beta_A_per_sigma']:.4f} | [{g['ci95_lo']:.4f}, {g['ci95_hi']:.4f}] | {g['odds_ratio_per_sigma']:.3f} | {g['n_rows']:,} | {g['n_episodes']:,} |"
        )
    L.append("")
    L.append("## Panel B: pooled quartiles, raw G^H (= any dataset-wide scalar baseline) vs A^H (primary M)\n")
    L.append(
        f"| signal | Q | P(bad tracking) % | mean phase | {meta['wall_name']} bins {meta['wall']} share % | append share % |\n|---|---|---|---|---|---|"
    )
    for name, rk, mk, wk, ak in [
        (
            "raw G^H",
            "pooled_rate_rawGH",
            "pooled_meanphase_rawGH",
            "pooled_wallshare_rawGH",
            "pooled_appendshare_rawGH",
        ),
        ("A^H", "pooled_rate_A", "pooled_meanphase_A", "pooled_wallshare_A", "pooled_appendshare_A"),
    ]:
        for j in range(4):
            L.append(
                f"| {name} | Q{j + 1} | {100 * r0[rk][j]:.2f} | {r0[mk][j]:.3f} | {100 * r0[wk][j]:.1f} | {100 * r0[ak][j]:.1f} |"
            )
    L.append("")
    L.append("## Caveats to state in the paper\n")
    L.append(
        "- Rows whose episode ends before t+H (by bad tracking or by motion end) are excluded: a failure there is already "
        "inside the return that built A^H, and a motion end there leaves no post-H trajectory. This exclusion avoids using "
        "outcomes already contained in the return used to construct A^H."
    )
    L.append(
        "- motion_ends inside the validation window is an observed negative (Y=0), not a censored row; only timeout / "
        "segment end / truncation / incomplete episodes are censored (excluded)."
    )
    L.append(
        "- Within a progress bin the quartiles of A^H and of raw G^H coincide, and beta_A is the same for both under bin fixed "
        "effects: Panel A and the regression test predictive information *after* conditioning on progress; they do not "
        "separate the progress-conditioned baseline from a global one. Panel B does that through quartile composition."
    )
    L.append(
        "- Raw G^H of rows near the motion end is a truncated sum (fewer than H rewards), so part of raw-Q1's late-phase "
        "excess is mechanical truncation rather than difficulty; both effects are what the progress-conditioned baseline removes."
    )
    L.append(
        "- The result is an association on the fixed dataset (transitions within an episode are correlated; CIs are "
        "episode-clustered). It shows A^H retains predictive information about downstream trajectory quality after "
        "progress normalisation; it does not show A^H is the true advantage."
    )
    (out_dir / "summary.md").write_text("\n".join(L) + "\n")
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default="offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5")
    ap.add_argument("--npz", default="offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5.aw_weights.H50.npz")
    ap.add_argument("--M", type=int, default=100, help="primary validation window length")
    ap.add_argument("--extra-M", type=int, nargs="*", default=[50, 150], help="additional window lengths (tables only)")
    ap.add_argument(
        "--min-count", type=int, default=200, help="min analysed rows per bin to enter the aggregate / heatmap"
    )
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="validation/results/post_h_validity")
    ap.add_argument("--prepend-bins", type=int, nargs="*", default=DEFAULT_LAYOUT["prepend"])
    ap.add_argument("--append-bins", type=int, nargs="*", default=DEFAULT_LAYOUT["append"])
    ap.add_argument("--wall-bins", type=int, nargs="*", default=DEFAULT_LAYOUT["wall"])
    ap.add_argument("--wall-name", default=DEFAULT_LAYOUT["wall_name"], help="label of the --wall-bins category")
    a = ap.parse_args()
    layout = {"prepend": a.prepend_bins, "append": a.append_bins, "wall": a.wall_bins, "wall_name": a.wall_name}

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load_h5(Path(a.h5))
    sc = dict(np.load(a.npz))
    if str(sc["rhash"]) != rhash(d["rewards"]):
        raise SystemExit(f"sidecar rhash {sc['rhash']} != dataset rhash {rhash(d['rewards'])}")
    H = int(sc["H"])
    n_bins = int(sc["phase_bin"].max()) + 1
    meta = {
        "h5": a.h5,
        "npz": a.npz,
        "H": H,
        "n_bins": n_bins,
        "sigma": float(sc["sigma"]),
        "min_count": a.min_count,
        "n_boot": a.n_boot,
        "prepend": a.prepend_bins,
        "append": a.append_bins,
        "wall": a.wall_bins,
        "wall_name": a.wall_name,
    }
    results = []
    for i, M in enumerate([a.M] + [m for m in a.extra_M if m != a.M]):
        print(f"[run] H={H} M={M}")
        r = run_one(d, sc, H, M, n_bins, a.min_count, a.n_boot, a.seed, out_dir, make_plots=(i == 0), layout=layout)
        results.append(r)
        print(
            "   within-bin Q1..Q4 (%):",
            fmt_row(r["within_bin_weighted"]),
            "| beta_A:",
            f"{r['logistic']['beta_A_per_sigma']:.4f}",
            f"[{r['logistic']['ci95_lo']:.4f}, {r['logistic']['ci95_hi']:.4f}]",
        )
    write_summary(results, meta, out_dir)
    print("[written]", out_dir / "summary.md")


if __name__ == "__main__":
    main()
