#!/usr/bin/env python3
"""
AW-CQL v0 : per-transition penalty-weight precompute + Stage-0 gate report.

    w_t     = clip(exp(A_hat_t / beta), max=W_MAX), then global mean-normalized to 1
    A_hat_t = G^H_t - baseline(motion_id, phase_bin)          (baseline = per-group mean)
    G^H_t   = sum_{k<H} gamma^k * r_{t+k}, truncated at episode end (never crosses episodes)

beta calibration:
    default   : beta = beta_scale * std(A_hat)  computed on THIS h5   (use on the TRAINING cell)
    --beta-abs: absolute beta in reward units                          (use for cross-dataset
                checks, e.g. expert auto-shutoff 0-b, with beta taken from the replay cell)

Output: <h5>.aw_weights.npz  {weight, advantage, gH, phase_bin, meta...}
        + printed report: ESS table over {0.5,1,2}*sigma, per-bin stats, gate verdicts.

Usage:
    python aw_precompute_weights.py offline_data/xxx.h5 \
        --gamma 0.99 --H 50 --n-bins 20 --w-max 10.0 --beta-scale 1.0
"""
import argparse, os, sys
import numpy as np


def find_key(f, candidates, required=True, label=""):
    for k in candidates:
        if k in f:
            return k
    if required:
        raise KeyError(f"none of {candidates} found in h5 (looking for {label}); "
                       f"available: {list(f.keys())[:40]}")
    return None


def load_arrays(path):
    import h5py
    f = h5py.File(path, "r")
    def get(cands, required=True, label=""):
        k = find_key(f, cands, required, label)
        if k is None:
            return None
        a = np.asarray(f[k])
        return a.squeeze() if a.ndim > 1 and a.shape[-1] == 1 else a
    r      = get(["rewards", "reward"], label="rewards").astype(np.float64)
    phase  = get(["motion_phase", "motion_phases", "phase"], label="phase").astype(np.float64)
    dones  = get(["dones", "done"], label="dones").astype(bool)
    truncs = get(["truncations", "truncation"], label="truncations").astype(bool)
    mid    = get(["motion_id", "motion_ids"], required=False)
    # done flavors (exporter convention: next_done_*; hazard analyzer reads the same keys).
    # dones/truncations semantics vary per H5, so failure stats must come from these, not dones.
    bad    = get(["next_done_bad_tracking", "done_bad_tracking"], required=False)
    mends  = get(["next_done_motion_ends", "done_motion_ends"], required=False)
    tmo    = get(["next_done_timeout", "done_timeout"], required=False)
    bad    = None if bad is None else bad.astype(bool)
    mends  = None if mends is None else mends.astype(bool)
    tmo    = None if tmo is None else tmo.astype(bool)
    sem    = None
    for attr_host in (f, f.get(find_key(f, ["motion_phase", "motion_phases", "phase"],
                                        False) or "", None)):
        try:
            if attr_host is not None and "motion_phase_semantics" in attr_host.attrs:
                sem = str(attr_host.attrs["motion_phase_semantics"]); break
        except Exception:
            pass
    f.close()
    return r, phase, dones, truncs, mid, sem, bad, mends, tmo


def episode_bounds(dones, truncs):
    """Episode end at row t if dones[t] | truncs[t]; file end forced as boundary."""
    end = dones | truncs
    end = end.copy(); end[-1] = True
    ends = np.flatnonzero(end)
    starts = np.concatenate([[0], ends[:-1] + 1])
    return starts, ends  # inclusive ends


def truncated_returns(r, starts, ends, gamma, H):
    """G^H_t within each episode via backward recurrence with sliding-window correction.

    The suffix recurrence is sequentially dependent, so the inner loop stays, but on a
    segment-local array (no global fancy indexing per step). Correctness is pinned by
    the brute-force comparison test.
    """
    g = np.zeros_like(r)
    gH = gamma ** H
    for s, e in zip(starts, ends):            # e inclusive
        seg = r[s:e + 1]
        L = len(seg)
        out = np.empty(L)
        for t in range(L - 1, -1, -1):
            acc = seg[t] + gamma * (out[t + 1] if t + 1 < L else 0.0)
            if t + H < L:
                acc -= gH * seg[t + H]
            out[t] = acc
        g[s:e + 1] = out
    return g


def per_group_baseline(g, group):
    uniq, inv = np.unique(group, return_inverse=True)
    sums = np.bincount(inv, weights=g)
    cnts = np.bincount(inv).astype(np.float64)
    return (sums / np.maximum(cnts, 1.0))[inv]


def ess_frac(w):
    return (w.sum() ** 2) / (np.maximum((w * w).sum(), 1e-12) * len(w))


def make_weights(A, beta, w_max):
    z = np.clip(A / max(beta, 1e-12), -20.0, 20.0)
    w = np.minimum(np.exp(z), w_max)
    clip_frac = float((w >= w_max - 1e-9).mean())
    w = w / w.mean()
    return w.astype(np.float32), clip_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--H", type=int, default=50)
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--w-max", type=float, default=10.0)
    ap.add_argument("--beta-scale", type=float, default=1.0,
                    help="beta = beta_scale * std(A_hat) on this h5")
    ap.add_argument("--beta-abs", type=float, default=None,
                    help="absolute beta (reward units); overrides beta-scale. "
                         "Use for cross-dataset checks (expert 0-b).")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    r, phase, dones, truncs, mid, sem, bad, mends, tmo = load_arrays(a.h5)
    N = len(r)
    print(f"[load] N={N:,}  phase_semantics={sem}  motion_id={'yes' if mid is not None else 'no (single motion assumed)'}")

    starts, ends = episode_bounds(dones, truncs)
    ep_len = ends - starts + 1
    # Failure stats from explicit done flavors, never from raw dones (whose semantics
    # vary per H5). CROSS-CHECK before trusting anything downstream: episode count and
    # bad_frac must match the hazard analyzer's table for this dataset (it segments by
    # episode_id and reads next_done_bad_tracking) — a mismatch means episode_bounds
    # draws different boundaries and G^H is computed on a different population.
    def _end_frac(flags):
        return f"{flags[ends].mean():.3f}" if flags is not None else "N/A(key missing)"
    print(f"[episodes] n={len(starts):,}  len mean={ep_len.mean():.1f} p50={np.median(ep_len):.0f} "
          f"p90={np.percentile(ep_len,90):.0f}")
    print(f"[episodes] bad_frac={_end_frac(bad)}  motion_ends_frac={_end_frac(mends)}  "
          f"timeout_frac={_end_frac(tmo)}  (from next_done_* flavors; "
          f"dones[ends]={dones[ends].mean():.3f} semantics-unverified, shown for reference only)")
    if bad is not None and mends is not None:
        unattributed = float((~(bad[ends] | mends[ends] | (tmo[ends] if tmo is not None else False))).mean())
        if unattributed > 0.01:
            print(f"[WARN] {unattributed:.3f} of episode ends carry no done flavor -> "
                  f"boundary convention may differ from the hazard analyzer; verify before trusting bins")

    # soft sanity: within-episode phase should be ~monotone non-decreasing (pre_step)
    dphi = np.diff(phase); boundary = np.zeros(N - 1, bool); boundary[ends[:-1]] = True
    neg = float((dphi[~boundary] < -1e-6).mean())
    if neg > 0.001:
        print(f"[WARN] within-episode negative phase deltas frac={neg:.4f} "
              f"(expected ~0 with pre_step; check semantics before trusting bins)")

    g = truncated_returns(r, starts, ends, a.gamma, a.H)

    bins = np.clip((phase * a.n_bins).astype(int), 0, a.n_bins - 1)
    group = bins if mid is None else (mid.astype(np.int64) * 10_000 + bins)
    A = g - per_group_baseline(g, group)
    sigma = float(A.std())
    print(f"[advantage] sigma(A_hat)={sigma:.5f}  (mean|A|={np.abs(A).mean():.5f})  "
          f"reward-scale-free downstream: beta scales with sigma")

    # ---- Stage 0-a: ESS scan over {0.5, 1, 2} * sigma (always printed) ----
    print("\n== Stage 0-a: beta scan ==")
    print(f"{'beta':>12} {'ESS/N':>8} {'clip%':>7} {'w_p01':>7} {'w_p50':>7} {'w_p99':>7} {'w_max':>8}")
    for s in (0.5, 1.0, 2.0):
        w_s, cf = make_weights(A, s * sigma, a.w_max)
        q = np.percentile(w_s, [1, 50, 99])
        print(f"{s:>6.1f}*sigma {ess_frac(w_s):>8.3f} {100*cf:>6.2f}% "
              f"{q[0]:>7.3f} {q[1]:>7.3f} {q[2]:>7.3f} {w_s.max():>8.3f}")

    beta = a.beta_abs if a.beta_abs is not None else a.beta_scale * sigma
    src = "ABS (cross-dataset mode)" if a.beta_abs is not None else f"{a.beta_scale}*sigma"
    w, clip_frac = make_weights(A, beta, a.w_max)
    e = ess_frac(w)
    print(f"\n[chosen] beta={beta:.6f} ({src})  ESS/N={e:.3f}  clip%={100*clip_frac:.2f}")

    # ---- diag: does G^H penalize censoring, or is it censoring-agnostic? ----
    # A over the last H steps of motion_ends episodes. The phase-bin baseline should
    # absorb the truncation-induced G^H drop there (everyone in a late bin is cut at a
    # similar distance). Prediction: mean(A_tail) ~ 0. Significantly negative means
    # early-success arrival is being penalized -> success trajectories' tails get
    # down-weighted, late-bin anchors weaken, and a kappa-2 (motion_ends) failure could
    # be a weighting bug instead of the method. Look at this BEFORE launching.
    if mends is not None:
        tail = np.zeros(N, bool)
        for s_i, e_i in zip(starts, ends):
            if mends[e_i]:
                tail[max(s_i, e_i - a.H + 1): e_i + 1] = True
        if tail.any():
            tA = A[tail] / max(sigma, 1e-12)
            q10, q50, q90 = np.percentile(tA, [10, 50, 90])
            print(f"[diag motion_ends-tail] n={int(tail.sum()):,}  mean(A)/sigma={tA.mean():+.3f}  "
                  f"p10/p50/p90={q10:+.3f}/{q50:+.3f}/{q90:+.3f}  mean_w={w[tail].mean():.3f} (global=1)")
            if tA.mean() < -0.2:
                print("[WARN] motion_ends tail mean(A) significantly negative -> success-penalty leak; "
                      "late-bin anchors will be down-weighted (v1: tail correction), interpret kappa-2 with care")
        else:
            print("[diag motion_ends-tail] no motion_ends episodes found")
    else:
        print("[diag motion_ends-tail] skipped: next_done_motion_ends not in h5")

    # ---- per-bin table (0-c lite: does w vary WHERE it matters, i.e. wall bins) ----
    print("\n== per-bin stats (chosen beta) ==")
    print(f"{'bin':>4} {'range':>13} {'count':>9} {'mean_gH':>9} {'mean_w':>7} {'std_w':>7} {'p10_w':>7} {'p90_w':>7}")
    std_by_bin = np.zeros(a.n_bins)
    for b in range(a.n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        std_by_bin[b] = w[m].std()
        p10, p90 = np.percentile(w[m], [10, 90])
        print(f"{b:>4} [{b/a.n_bins:.2f},{(b+1)/a.n_bins:.2f}) {int(m.sum()):>9,} "
              f"{g[m].mean():>9.4f} {w[m].mean():>7.3f} {w[m].std():>7.3f} {p10:>7.3f} {p90:>7.3f}")

    # ---- gate verdicts ----
    print("\n== GATE VERDICTS ==")
    ok = True
    if e > 0.90:
        print("KILL(0-a upper): ESS/N > 0.90 -> weights ~uniform, method is a no-op on this cell."); ok = False
    elif e < 0.05:
        print("KILL(0-a lower): ESS/N < 0.05 -> mass collapsed to few transitions (soft data deletion)."); ok = False
    else:
        print(f"PASS(0-a): ESS/N={e:.3f} in [0.05, 0.90].")
    order = np.argsort(-std_by_bin)
    print(f"top-3 within-bin std(w) at bins {order[:3].tolist()} "
          f"(pre-registered expectation: wall bins e.g. 4,5,13 for largebox replay)")
    if std_by_bin.max() < 0.05:
        print("KILL(0-c lite): within-bin std(w) ~0 everywhere -> no filtering signal where it matters."); ok = False

    out = a.out or (a.h5 + ".aw_weights.npz")
    np.savez_compressed(out, weight=w, advantage=A.astype(np.float32),
                        gH=g.astype(np.float32), phase_bin=bins.astype(np.int16),
                        beta=beta, sigma=sigma, gamma=a.gamma, H=a.H,
                        w_max=a.w_max, n=N, ess_frac=e, clip_frac=clip_frac,
                        h5=os.path.basename(a.h5))
    print(f"\n[saved] {out}  ({'LAUNCH OK' if ok else 'DO NOT LAUNCH'})")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
