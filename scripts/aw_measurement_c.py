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
  python scripts/aw_measurement_c.py offline_data/xxx.h5 --bins 6 7 8
"""
import argparse
import numpy as np

try:
    from scripts.aw_precompute_weights import h5_reward_fingerprint
except ModuleNotFoundError:
    from aw_precompute_weights import h5_reward_fingerprint


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--bins", type=int, nargs="+", default=[4, 5, 13, 14])
    ap.add_argument("--wall-span", type=int, default=1,
                    help="FAIL = bad terminal in [b, b+span] (default 1: this bin or next)")
    a = ap.parse_args(argv)

    import h5py
    sidecar_path = a.npz or (a.h5 + ".aw_weights.npz")
    with np.load(sidecar_path, allow_pickle=False) as sidecar:
        w = sidecar["weight"].astype(np.float64)
        bins = sidecar["phase_bin"].astype(int)
        stored_hash = str(np.asarray(sidecar["rhash"]).item()) if "rhash" in sidecar else None
        sidecar_n = int(np.asarray(sidecar["n"]).item())
        beta = float(np.asarray(sidecar["beta"]).item())
        sidecar_ess = float(np.asarray(sidecar["ess_frac"]).item())

    with h5py.File(a.h5, "r") as h5_file:
        dones = np.asarray(h5_file["dones"]).squeeze().astype(bool)
        truncs = np.asarray(h5_file["truncations"]).squeeze().astype(bool)
        bad = None
        for key in ("next_done_bad_tracking", "done_bad_tracking"):
            if key in h5_file:
                bad = np.asarray(h5_file[key]).squeeze().astype(bool)
                break
    if bad is None:
        raise KeyError("bad-tracking flag missing in H5")

    # pairing guards (same convention as the precompute sidecar)
    actual_hash, num_rows, _ = h5_reward_fingerprint(a.h5)
    if stored_hash is None:
        raise KeyError("sidecar has no rhash; regenerate it with scripts/aw_precompute_weights.py")
    if stored_hash != actual_hash:
        raise ValueError(
            f"rhash mismatch: sidecar '{sidecar_path}' has {stored_hash}, "
            f"H5 '{a.h5}' has {actual_hash}"
        )
    if len(w) != num_rows or sidecar_n != num_rows:
        raise ValueError(
            f"row-count mismatch: weights={len(w):,}, sidecar_n={sidecar_n:,}, H5={num_rows:,}"
        )

    end = (dones | truncs).copy(); end[-1] = True
    ends = np.flatnonzero(end)
    starts = np.concatenate([[0], ends[:-1] + 1])
    N = num_rows
    ep_id = np.zeros(N, np.int64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        ep_id[s:e + 1] = i
    term_bin = bins[ends]
    term_bad = bad[ends]
    print(f"[pairing] rhash OK  N={N:,}  episodes={len(starts):,}  bad_frac={term_bad.mean():.3f}")
    print(f"[meta] beta={beta:.6f}  ESS/N={sidecar_ess:.3f}  "
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
