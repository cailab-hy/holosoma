#!/usr/bin/env python3
"""
Measurement C : per-bin anchor w-mass decomposition by wall outcome.

For rows in phase-bin b, split their EPISODES into
  FAIL : bad_tracking terminal within [b, b+span]   (dies at this wall)
  SURV : everything else (reaches b+span+1 or beyond, or motion_ends)
then compare unweighted count share vs w-mass share of the anchor.

Registered prediction (largebox replay, wall-1): at bin 5 the SURV w-mass share
exceeds 0.5 (net anchor direction flipped to survivors) even though FAIL dominates
counts. If NOT flipped -> prime suspect for AW-CQL's residual wall-1 hazard (4.4%).

Usage:
  python aw_measurement_c.py offline_data/xxx.h5 [--npz path] [--bins 4 5 13 14]
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--bins", type=int, nargs="+", default=[4, 5, 13, 14])
    ap.add_argument("--wall-span", type=int, default=1,
                    help="FAIL = bad terminal in [b, b+span] (default 1: this bin or next)")
    a = ap.parse_args()

    import h5py, hashlib
    z = np.load(a.npz or (a.h5 + ".aw_weights.npz"))
    w = z["weight"].astype(np.float64)
    bins = z["phase_bin"].astype(int)

    f = h5py.File(a.h5, "r")
    r = np.asarray(f["rewards"]).squeeze().astype(np.float64)
    dones = np.asarray(f["dones"]).squeeze().astype(bool)
    truncs = np.asarray(f["truncations"]).squeeze().astype(bool)
    bad = None
    for k in ("next_done_bad_tracking", "done_bad_tracking"):
        if k in f:
            bad = np.asarray(f[k]).squeeze().astype(bool)
            break
    f.close()
    assert bad is not None, "bad-tracking flag missing in h5"

    # pairing guards (same convention as the precompute sidecar)
    rh = hashlib.sha256(np.ascontiguousarray(r[:1000]).tobytes()
                        + np.ascontiguousarray(r[-1000:]).tobytes()).hexdigest()[:16]
    if "rhash" in z.files:
        assert str(z["rhash"]) == rh, "rhash mismatch: npz belongs to a different h5 build"
    assert len(w) == len(r) == int(z["n"]), "row-count mismatch between npz and h5"

    end = (dones | truncs).copy(); end[-1] = True
    ends = np.flatnonzero(end)
    starts = np.concatenate([[0], ends[:-1] + 1])
    N = len(r)
    ep_id = np.zeros(N, np.int64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        ep_id[s:e + 1] = i
    term_bin = bins[ends]
    term_bad = bad[ends]
    print(f"[pairing] rhash OK  N={N:,}  episodes={len(starts):,}  bad_frac={term_bad.mean():.3f}")
    print(f"[meta] beta={float(z['beta']):.6f}  ESS/N={float(z['ess_frac']):.3f}  "
          f"(weights are the exact array the AW arm trained on)")

    for b in a.bins:
        m = bins == b
        if not m.any():
            print(f"\nbin {b:2d}: no rows"); continue
        eb = ep_id[m]
        fail_ep = term_bad & (term_bin >= b) & (term_bin <= b + a.wall_span)
        is_fail = fail_ep[eb]
        wm = w[m]
        wf, ws = wm[is_fail], wm[~is_fail]
        cf, cs = int(is_fail.sum()), int((~is_fail).sum())
        Sf, Ss = float(wf.sum()), float(ws.sum())
        surv_cnt_share = cs / max(cf + cs, 1)
        surv_w_share = Ss / max(Sf + Ss, 1e-12)
        top = np.argsort(wm)[-max(len(wm) // 10, 1):]
        top_surv = float((~is_fail)[top].mean())
        print(f"\nbin {b:2d}: rows={int(m.sum()):,}  FAIL(bad@[{b},{b + a.wall_span}]) rows={cf:,}  SURV rows={cs:,}")
        print(f"  count-share SURV = {surv_cnt_share:.3f}   w-mass-share SURV = {surv_w_share:.3f}   "
              f"-> net anchor {'FLIPPED to survivors' if Ss > Sf else 'STILL on failures'}")
        mf = wf.mean() if cf else float('nan')
        ms = ws.mean() if cs else float('nan')
        print(f"  mean_w FAIL={mf:.3f}  SURV={ms:.3f}   top-decile-by-w SURV frac = {top_surv:.3f}")


if __name__ == "__main__":
    main()
