#!/usr/bin/env python3
"""Build the ODPR-CQL weight sidecar from OPER-A (ODPR-A) priority weights.

AW-CQL (holosoma.agents.aw_cql) reads a ``.npz`` sidecar with a per-transition mean-one ``weight``
and multiplies it into the conservative bracket. This script writes such a sidecar from the output of
``scripts/oper_precompute_weights.py`` so that "OPER-A CQL" is exactly the AW-CQL code path with only the
weight table swapped:

    python scripts/odpr_make_sidecar.py \\
        --oper offline_data/<h5>.oper_a.seed1.npz offline_data/<h5>.oper_a.seed2.npz offline_data/<h5>.oper_a.seed3.npz \\
        --out offline_data/<h5>.oper_a.aw_sidecar.npz
    python src/holosoma/holosoma/offline_train_agent.py exp:g1-29dof-wbt-aw-cql algo:aw-cql \\
        --algo.config.aw-weights-path offline_data/<h5>.oper_a.aw_sidecar.npz

Weight modes (``--mode``), all applied to the seed-averaged iteration weights and returned mean-one:
  odpr  (default) ODPR's load-time normalisation: linear rescale of the weight std to ``--std`` (2.0) with
        a floor of ``--eps`` (0.1); this is what ODPR itself trains with.
  ess   same linear rescale, with the std chosen so ESS/N matches the sidecar given by ``--match-ess``
        (e.g. the AW H50 sidecar), isolating *which* transitions are preferred from how sharply.
  raw   the unscaled OPER-A weights (nearly uniform on our data).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_oper_vs_aw import ess_frac, odpr_ess_matched, odpr_replace_weights_linear  # noqa: E402


def build_weight(opers: list, mode: str, std: float, eps: float, oper_iter: int | None, target_ess: float | None):
    if oper_iter is not None:
        per_seed = [z["weights_by_iter"][oper_iter - 1].astype(np.float64) for z in opers]
    else:
        per_seed = [z["weight"].astype(np.float64) for z in opers]
    weight = np.mean(per_seed, axis=0)
    weight = weight / weight.mean()
    used_std = 0.0
    if mode == "odpr":
        weight, used_std = odpr_replace_weights_linear(weight, std=std, eps=eps), std
    elif mode == "ess":
        if target_ess is None:
            raise ValueError("--mode ess requires --match-ess <sidecar.npz>")
        weight, used_std = odpr_ess_matched(weight, target_ess, eps=eps)
    elif mode != "raw":
        raise ValueError(f"unknown mode {mode!r}")
    return (weight / weight.mean()).astype(np.float32), float(used_std)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oper", nargs="+", required=True, help="OPER-A npz files; several seeds are averaged")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("odpr", "ess", "raw"), default="odpr")
    ap.add_argument("--std", type=float, default=2.0, help="ODPR std parameter (mode odpr)")
    ap.add_argument("--eps", type=float, default=0.1, help="ODPR weight floor (modes odpr/ess)")
    ap.add_argument("--match-ess", default=None, help="sidecar whose ESS/N is matched (mode ess)")
    ap.add_argument("--oper-iter", type=int, default=None, help="use this OPER iteration instead of the final one")
    a = ap.parse_args(argv)

    opers = [np.load(p, allow_pickle=False) for p in a.oper]
    ns = {int(z["n"]) for z in opers}
    hashes = {str(np.asarray(z["rhash"]).item()) for z in opers}
    if len(ns) != 1 or len(hashes) != 1:
        raise SystemExit(f"OPER files disagree on the dataset: n={ns} rhash={hashes}")
    n, rhash = ns.pop(), hashes.pop()

    target_ess = None
    if a.match_ess:
        with np.load(a.match_ess, allow_pickle=False) as ref:
            if str(np.asarray(ref["rhash"]).item()) != rhash:
                raise SystemExit(f"--match-ess sidecar is for another dataset (rhash != {rhash})")
            target_ess = ess_frac(ref["weight"])

    weight, used_std = build_weight(opers, a.mode, a.std, a.eps, a.oper_iter, target_ess)
    if weight.shape != (n,) or not np.isfinite(weight).all() or weight.min() < 0:
        raise SystemExit(f"invalid weights: shape={weight.shape}")
    floor = float(weight.min())
    np.savez(
        a.out,
        weight=weight,
        n=np.int64(n),
        rhash=np.asarray(rhash),
        h5=np.asarray(Path(json.loads(str(opers[0]["config_json"]))["h5"]).name),
        # Keys the AW-CQL loader logs; beta has no OPER meaning, clip_frac = share sitting on the eps floor.
        beta=np.float64(0.0),
        ess_frac=np.float64(ess_frac(weight)),
        clip_frac=np.float64(float((weight <= floor * (1 + 1e-6)).mean()) if a.mode != "raw" else 0.0),
        phase_bin=opers[0]["phase_bin"],
        source=np.asarray("oper_a"),
        mode=np.asarray(a.mode),
        odpr_std=np.float64(used_std),
        odpr_eps=np.float64(a.eps if a.mode != "raw" else 0.0),
        num_seeds=np.int64(len(opers)),
        oper_files=np.asarray([Path(p).name for p in a.oper]),
    )
    print(
        f"[odpr_make_sidecar] wrote {a.out}: mode={a.mode} seeds={len(opers)} n={n} rhash={rhash} "
        f"mean={weight.mean():.6f} std={weight.std():.3f} min={weight.min():.3f} max={weight.max():.2f} "
        f"ESS/N={ess_frac(weight):.3f} odpr_std={used_std:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
