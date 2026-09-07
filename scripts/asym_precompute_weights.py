#!/usr/bin/env python3
"""Build Asym-CQL weights from an existing, paired AW-CQL sidecar.

The output stores ``weight`` (w+, compatible with AWCQLAgent) and
``weight_minus`` (w-).  By default w- is the reverse-rank assignment of w+:
the two arrays have the exact same empirical distribution, but the largest
weights are assigned to the smallest advantages.  ``--minus-mode exp`` keeps
the original exploratory Norm[clip(exp(-A/beta))] construction available.

Crucially, this script does not recompute w+.  It loads the exact ``weight``
array used by AW-CQL, so mirror mode is a strict same-budget/different-
assignment control even if the AW sidecar used non-default precompute options.
"""

import argparse
import os
import sys

import numpy as np

try:  # module import (tests / tooling)
    from scripts.aw_precompute_weights import (
        make_weights, verify_sidecar,
    )
except ModuleNotFoundError:  # direct ``python scripts/asym_precompute_weights.py``
    from aw_precompute_weights import (
        make_weights, verify_sidecar,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("h5")
    parser.add_argument(
        "--aw-npz",
        default=None,
        help="existing AW sidecar; default: <h5>.aw_weights.npz",
    )
    parser.add_argument(
        "--minus-mode",
        choices=("mirror", "exp"),
        default="mirror",
        help="mirror (default): reverse-rank w+ with identical distribution; "
             "exp: independently normalized exp(-A/beta) exploratory control",
    )
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def make_mirrored_weights(advantage, weight_plus):
    """Assign descending w+ values to ascending advantages.

    This is a permutation only: dtype, values, global mean, ESS, clipping
    fraction, and every empirical quantile are exactly identical to w+.
    Stable sorting makes the assignment deterministic when advantages tie.
    """
    advantage = np.asarray(advantage)
    weight_plus = np.asarray(weight_plus)
    if advantage.shape != weight_plus.shape:
        raise ValueError(
            f"advantage and weight_plus shapes differ: {advantage.shape} != {weight_plus.shape}"
        )
    ascending_advantage = np.argsort(advantage, kind="stable")
    descending_weight = np.sort(weight_plus, kind="stable")[::-1]
    weight_minus = np.empty_like(weight_plus)
    weight_minus[ascending_advantage] = descending_weight
    return weight_minus


def distribution_stats(weight):
    """Distribution diagnostics in float64 to avoid reduction-order noise."""
    weight64 = np.asarray(weight, dtype=np.float64)
    return {
        "mean": float(weight64.mean()),
        "std": float(weight64.std()),
        "min": float(weight64.min()),
        "max": float(weight64.max()),
        "p95": float(np.percentile(weight64, 95)),
        "p99": float(np.percentile(weight64, 99)),
        "ess_frac": float((weight64.sum() ** 2) / ((weight64 * weight64).sum() * len(weight64))),
    }


def main(argv=None):
    args = parse_args(argv)
    aw_path = args.aw_npz or f"{args.h5}.aw_weights.npz"
    if not verify_sidecar(args.h5, aw_path):
        raise ValueError("Refusing to build Asym weights from an unpaired AW sidecar")
    with np.load(aw_path, allow_pickle=False) as aw:
        required = ("weight", "advantage", "phase_bin", "beta", "w_max", "n", "rhash")
        missing = [key for key in required if key not in aw]
        if missing:
            raise ValueError(f"AW sidecar is missing keys required by Asym-CQL: {missing}")
        # Copy, never reconstruct: this is the exact w+ used by the paired AW run.
        weight_plus = np.asarray(aw["weight"], dtype=np.float32).copy()
        advantage = np.asarray(aw["advantage"], dtype=np.float32).copy()
        bins = np.asarray(aw["phase_bin"], dtype=np.int16).copy()
        beta = float(aw["beta"])
        w_max = float(aw["w_max"])
        n = int(aw["n"])
        rhash = str(np.asarray(aw["rhash"]).item())
        g_h = np.asarray(aw["gH"], dtype=np.float32).copy() if "gH" in aw else np.empty(0, np.float32)
        sigma = float(aw["sigma"]) if "sigma" in aw else float(advantage.std())
        gamma = float(aw["gamma"]) if "gamma" in aw else np.nan
        horizon = int(aw["H"]) if "H" in aw else -1
        plus_clip_frac = float(aw["clip_frac"]) if "clip_frac" in aw else float((weight_plus == weight_plus.max()).mean())

    if args.minus_mode == "mirror":
        weight_minus = make_mirrored_weights(advantage, weight_plus)
        minus_clip_frac = plus_clip_frac
    else:
        weight_minus, minus_clip_frac = make_weights(-advantage, beta, w_max)
    plus_stats = distribution_stats(weight_plus)
    minus_stats = distribution_stats(weight_minus)
    stored_minus_ess = plus_stats["ess_frac"] if args.minus_mode == "mirror" else minus_stats["ess_frac"]
    out = args.out or f"{args.h5}.asym_weights.npz"
    np.savez_compressed(
        out,
        weight=weight_plus,
        weight_plus=weight_plus,
        weight_minus=weight_minus,
        advantage=advantage.astype(np.float32),
        gH=g_h,
        phase_bin=bins.astype(np.int16),
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        H=horizon,
        w_max=w_max,
        n=n,
        plus_ess_frac=plus_stats["ess_frac"],
        minus_ess_frac=stored_minus_ess,
        # Compatibility aliases consumed by AWCQLAgent's established loader.
        ess_frac=plus_stats["ess_frac"],
        plus_clip_frac=plus_clip_frac,
        minus_clip_frac=minus_clip_frac,
        minus_mode=args.minus_mode,
        clip_frac=plus_clip_frac,
        h5=os.path.basename(args.h5),
        rhash=rhash,
        source_aw_sidecar=os.path.abspath(aw_path),
    )
    print(f"[saved] {out}")
    print(f"[w- mode] {args.minus_mode}")
    print(f"[source AW] {aw_path}")
    print(f"mean(w+) = {plus_stats['mean']:.4f}")
    print(f"mean(w-) = {minus_stats['mean']:.4f}")
    print(f"ESS/N w+ = {plus_stats['ess_frac']:.4f}")
    print(f"ESS/N w- = {minus_stats['ess_frac']:.4f}")
    if args.minus_mode == "mirror":
        same_distribution = np.array_equal(np.sort(weight_plus), np.sort(weight_minus))
        assert same_distribution
        for key in ("mean", "std", "min", "max", "p95", "p99", "ess_frac"):
            if not np.isclose(plus_stats[key], minus_stats[key], rtol=0.0, atol=1e-12):
                raise AssertionError(f"mirror distribution mismatch for {key}")
        print(f"same_sorted_distribution = {same_distribution}")
        print("stat          w+              w-")
        for key in ("mean", "std", "min", "max", "p95", "p99"):
            print(f"{key:<5} {plus_stats[key]:>14.8f} {minus_stats[key]:>14.8f}")
    for phase_bin in range(int(bins.max()) + 1):
        mask = bins == phase_bin
        if mask.any():
            print(
                f"[phase {phase_bin:02d}] n={mask.sum():7d} "
                f"w+={weight_plus[mask].mean():.4f} w-={weight_minus[mask].mean():.4f} "
                f"diff={(weight_minus[mask] - weight_plus[mask]).mean():+.4f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
