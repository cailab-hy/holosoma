#!/usr/bin/env python3
# ruff: noqa: E501
"""Frozen V_H baseline ablation for PRe / AW-CQL weights.

PRe compares each transition's H-step return with the mean of its progress bin:
    A_i = G_i^H - b_{k(i)}.
This script swaps ONLY the baseline for a learned state-conditioned one,
    A_i^V = G_i^H - V_H(s_i),        V_H(s) ~ E[G^H | s]   (behaviour policy),
and keeps everything downstream identical to scripts/aw_precompute_weights.py
(sigma_A from this A, exp(A / sigma_A), w_max clip, unit-mean normalisation).

V_H
  target   the same G^H as PRe (same H, gamma, episode-boundary truncation; reuses the PRe code path)
  input    critic observation + reference progress kappa (so V_H contains b_k's information)
  loss     MSE regression on G^H (conditional mean; no TD bootstrap, no expectile/quantile)
  fitting  K-fold cross-fitting BY EPISODE: every transition's V_H(s_i) is predicted by a model that
           never saw its episode (neighbouring rows share almost all of G^H, so a transition-level
           split would leak). Early stopping uses a validation split carved out of the K-1 training
           folds, so the predicted fold is never used for model selection either.
  frozen   computed once before CQL training; no iteration.

Outputs (AW-CQL-compatible sidecars, loadable via --algo.config.aw-weights-path):
  <h5>.aw_weights.H{H}.vh.npz            sigma_A recomputed from A^V            (main result)
  <h5>.aw_weights.H{H}.vh_presigma.npz   PRe's sigma_A kept (same temperature)  (sensitivity; needs --pre)
  <out-dir>/report.md, report.json       fit quality, kappa share of V_H, weight statistics vs PRe

Usage:
  python scripts/vh_precompute_weights.py offline_data/<dataset>.h5 \
      --pre offline_data/<dataset>.h5.aw_weights.H50.npz --out-dir validation/results/vh_baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aw_precompute_weights import (  # noqa: E402
    episode_bounds,
    ess_frac,
    load_arrays,
    make_weights,
    per_group_baseline,
    reward_fingerprint,
    truncated_returns,
)


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1.0 - np.square(y - pred).sum() / np.square(y - y.mean()).sum())


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def within_bin_spearman(x: np.ndarray, y: np.ndarray, bins: np.ndarray, n_bins: int) -> tuple[float, list[float]]:
    per_bin, counts = [], []
    for k in range(n_bins):
        m = bins == k
        if m.sum() > 10:
            per_bin.append(spearman(x[m], y[m]))
            counts.append(int(m.sum()))
    return float(np.average(per_bin, weights=counts)), per_bin


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int):
        super().__init__()
        mods: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def predict(model, x, mean, std, y_mean, y_std, idx, batch=262144):
    out = torch.empty(len(idx), device=x.device)
    model.eval()
    for i in range(0, len(idx), batch):
        sl = idx[i : i + batch]
        out[i : i + batch] = model((x[sl] - mean) / std) * y_std + y_mean
    return out


def fit_fold(x, y, train_idx, val_idx, a, gen, tag):
    """MSE regression of y on x with early stopping on val_idx (rows from the training folds only)."""
    mean = x[train_idx].mean(0, keepdim=True)
    std = x[train_idx].std(0, keepdim=True) + 1e-3
    y_mean, y_std = y[train_idx].mean(), y[train_idx].std() + 1e-8
    model = MLP(x.shape[1], a.hidden, a.layers).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps_per_epoch = max(1, len(train_idx) // a.batch_size)
    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, a.max_epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device=x.device, generator=gen)]
        tr_loss = 0.0
        for s in range(steps_per_epoch):
            b = perm[s * a.batch_size : (s + 1) * a.batch_size]
            loss = torch.mean((model((x[b] - mean) / std) - (y[b] - y_mean) / y_std) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += float(loss)
        val_pred = predict(model, x, mean, std, y_mean, y_std, val_idx)
        val_mse = float(torch.mean((val_pred - y[val_idx]) ** 2))
        history.append((epoch, tr_loss / steps_per_epoch * float(y_std) ** 2, val_mse))
        if val_mse < best_val * (1 - 1e-4):
            best_val, best_epoch, bad = val_mse, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    print(f"[{tag}] epochs={epoch} best_epoch={best_epoch} val_mse={best_val:.5f}", flush=True)
    return model, (mean, std, y_mean, y_std), best_epoch, history


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5")
    ap.add_argument("--pre", default=None, help="PRe sidecar (same H) for sigma, G^H check and weight comparison")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--H", type=int, default=50)
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--w-max", type=float, default=10.0)
    ap.add_argument("--obs-key", default="critic_observations")
    ap.add_argument("--no-kappa", action="store_true", help="do not append kappa to the V_H input")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.1, help="share of training episodes held out for early stopping")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="main sidecar path (default <h5>.aw_weights.H{H}.vh.npz)")
    ap.add_argument("--out-dir", default="validation/results/vh_baseline")
    a = ap.parse_args(argv)
    t0 = time.time()

    r, phase, dones, truncs, mid, _sem, _bad, _mends, _tmo = load_arrays(a.h5)
    if mid is not None:
        raise SystemExit("multi-motion datasets are not supported by this ablation script")
    n = len(r)
    starts, ends = episode_bounds(dones, truncs)
    g = truncated_returns(r, starts, ends, a.gamma, a.H)  # identical to PRe's G^H
    bins = np.clip((phase * a.n_bins).astype(int), 0, a.n_bins - 1)
    b_k = per_group_baseline(g, bins)
    a_pre = g - b_k
    rhash = reward_fingerprint(r)
    episode_of_row = np.repeat(np.arange(len(starts)), ends - starts + 1)
    print(f"[load] N={n:,} episodes={len(starts):,} H={a.H} gamma={a.gamma} rhash={rhash}")

    pre = None
    if a.pre:
        pre = np.load(a.pre, allow_pickle=False)
        if str(np.asarray(pre["rhash"]).item()) != rhash or int(pre["H"]) != a.H:
            raise SystemExit("--pre sidecar belongs to another dataset or horizon")
        err = float(np.abs(pre["gH"].astype(np.float64) - g).max())
        if err > 1e-3:
            raise SystemExit(f"G^H differs from the PRe sidecar (max abs diff {err:.4g})")
        print(f"[check] G^H matches PRe sidecar (max abs diff {err:.2e}); PRe sigma={float(pre['sigma']):.5f}")

    dev = torch.device(a.device)
    with h5py.File(a.h5, "r") as f:
        obs = torch.from_numpy(f[a.obs_key][:n].astype(np.float32))
    if not a.no_kappa:
        obs = torch.cat([obs, torch.from_numpy(phase.astype(np.float32)).unsqueeze(1)], dim=1)
    x = obs.to(dev)
    del obs
    y = torch.from_numpy(g.astype(np.float32)).to(dev)
    print(f"[input] {a.obs_key} dim={x.shape[1] - (0 if a.no_kappa else 1)} + kappa={'no' if a.no_kappa else 'yes'}")

    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)
    torch.manual_seed(a.seed)
    fold_of_episode = rng.permutation(len(starts)) % a.folds
    fold_of_row = fold_of_episode[episode_of_row]

    v_h = np.empty(n, dtype=np.float64)
    fold_stats = []
    for k in range(a.folds):
        train_eps = np.flatnonzero(fold_of_episode != k)
        val_eps = rng.choice(train_eps, size=max(1, int(a.val_frac * len(train_eps))), replace=False)
        is_val_ep = np.zeros(len(starts), bool)
        is_val_ep[val_eps] = True
        row_val = (fold_of_row != k) & is_val_ep[episode_of_row]
        row_train = (fold_of_row != k) & ~row_val
        row_oof = fold_of_row == k
        tr = torch.from_numpy(np.flatnonzero(row_train)).to(dev)
        va = torch.from_numpy(np.flatnonzero(row_val)).to(dev)
        oo = torch.from_numpy(np.flatnonzero(row_oof)).to(dev)
        model, norm, best_epoch, _hist = fit_fold(x, y, tr, va, a, gen, f"fold {k + 1}/{a.folds}")
        p_tr = predict(model, x, *norm, tr).cpu().numpy().astype(np.float64)
        p_va = predict(model, x, *norm, va).cpu().numpy().astype(np.float64)
        p_oo = predict(model, x, *norm, oo).cpu().numpy().astype(np.float64)
        v_h[row_oof] = p_oo
        stats = {
            "fold": k + 1,
            "rows_train": int(row_train.sum()),
            "rows_val": int(row_val.sum()),
            "rows_oof": int(row_oof.sum()),
            "best_epoch": best_epoch,
            "r2_train": r2(g[row_train], p_tr),
            "r2_val": r2(g[row_val], p_va),
            "r2_oof": r2(g[row_oof], p_oo),
        }
        stats["train_minus_oof_r2"] = stats["r2_train"] - stats["r2_oof"]
        fold_stats.append(stats)
        print(f"         R2 train={stats['r2_train']:.4f} val={stats['r2_val']:.4f} out-of-fold={stats['r2_oof']:.4f}", flush=True)
        del model
    train_seconds = time.time() - t0

    a_v = g - v_h
    sigma_v = float(a_v.std())
    w_v, clip_v = make_weights(a_v, sigma_v, a.w_max)
    variants = {"vh": (w_v, clip_v, sigma_v)}
    if pre is not None:
        sigma_pre = float(pre["sigma"])
        w_vp, clip_vp = make_weights(a_v, sigma_pre, a.w_max)
        variants["vh_presigma"] = (w_vp, clip_vp, sigma_pre)

    bin_mean_v = per_group_baseline(v_h, bins)
    report = {
        "h5": os.path.basename(a.h5),
        "rhash": rhash,
        "N": n,
        "episodes": len(starts),
        "H": a.H,
        "gamma": a.gamma,
        "config": {k: getattr(a, k) for k in ("obs_key", "no_kappa", "folds", "val_frac", "hidden", "layers", "lr", "batch_size", "max_epochs", "patience", "seed")},
        "train_seconds": train_seconds,
        "folds": fold_stats,
        "r2_oof_all": r2(g, v_h),
        "r2_bin_baseline": r2(g, b_k),
        "vh_variance_explained_by_kappa_bins": r2(v_h, bin_mean_v),
        "corr_vh_bk": float(np.corrcoef(v_h, b_k)[0, 1]),
        "sigma_A_pre": float(a_pre.std()),
        "sigma_A_vh": sigma_v,
        "corr_A_vh_A_pre": float(np.corrcoef(a_v, a_pre)[0, 1]),
        "within_bin_spearman_A": within_bin_spearman(a_v, a_pre, bins, a.n_bins)[0],
        "weights": {},
    }
    w_pre = None if pre is None else pre["weight"].astype(np.float64)
    for name, (w, clip, sigma) in variants.items():
        w64 = w.astype(np.float64)
        entry = {
            "sigma_used": sigma,
            "ess_frac": float(ess_frac(w64)),
            "clip_frac": clip,
            "max_w": float(w64.max()),
            "std_w": float(w64.std()),
            "bin_weight_mass": [float(w64[bins == k].sum() / w64.sum()) for k in range(a.n_bins)],
            "bin_std_w": [float(w64[bins == k].std()) for k in range(a.n_bins)],
        }
        if w_pre is not None:
            entry["pearson_w_vs_pre"] = float(np.corrcoef(w64, w_pre)[0, 1])
            entry["within_bin_spearman_w_vs_pre"] = within_bin_spearman(w64, w_pre, bins, a.n_bins)[0]
        report["weights"][name] = entry
    if w_pre is not None:
        report["weights"]["pre"] = {
            "sigma_used": float(pre["sigma"]),
            "ess_frac": float(ess_frac(w_pre)),
            "clip_frac": float(pre["clip_frac"]),
            "max_w": float(w_pre.max()),
            "std_w": float(w_pre.std()),
            "bin_weight_mass": [float(w_pre[bins == k].sum() / w_pre.sum()) for k in range(a.n_bins)],
            "bin_std_w": [float(w_pre[bins == k].std()) for k in range(a.n_bins)],
        }

    base = a.out or f"{a.h5}.aw_weights.H{a.H}.vh.npz"
    paths = {"vh": base, "vh_presigma": base.replace(".vh.npz", ".vh_presigma.npz")}
    for name, (w, clip, sigma) in variants.items():
        np.savez_compressed(
            paths[name],
            baseline=name,
            weight=w,
            advantage=a_v.astype(np.float32),
            gH=g.astype(np.float32),
            v_h=v_h.astype(np.float32),
            fold=fold_of_row.astype(np.int8),
            phase_bin=bins.astype(np.int16),
            beta=sigma,
            sigma=sigma,
            gamma=a.gamma,
            H=a.H,
            w_max=a.w_max,
            n=n,
            ess_frac=float(ess_frac(w.astype(np.float64))),
            clip_frac=clip,
            h5=os.path.basename(a.h5),
            rhash=rhash,
        )
        print(f"[saved] {paths[name]}")
    report["sidecars"] = {k: paths[k] for k in variants}

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    lines = [
        "# Frozen V_H baseline vs PRe phase-bin baseline\n",
        f"dataset `{report['h5']}` (rhash {rhash}), N={n:,}, episodes={len(starts):,}, H={a.H}, gamma={a.gamma}\n",
        f"V_H: MLP {a.layers}x{a.hidden} ReLU, MSE on G^H, input {a.obs_key}{'' if a.no_kappa else ' + kappa'}, "
        f"{a.folds}-fold cross-fitting by episode, early stopping on {a.val_frac:.0%} of training episodes. "
        f"Total time {train_seconds / 60:.1f} min.\n",
        "## Fit quality (R2 of G^H)\n",
        "| fold | rows train | rows OOF | best epoch | R2 train | R2 val | R2 out-of-fold | train - OOF |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in fold_stats:
        lines.append(f"| {s['fold']} | {s['rows_train']:,} | {s['rows_oof']:,} | {s['best_epoch']} | {s['r2_train']:.4f} | {s['r2_val']:.4f} | {s['r2_oof']:.4f} | {s['train_minus_oof_r2']:+.4f} |")
    lines += [
        f"\nAll rows, out-of-fold: R2(G^H, V_H) = **{report['r2_oof_all']:.4f}**; phase-bin baseline: R2(G^H, b_k) = **{report['r2_bin_baseline']:.4f}**.\n",
        "## How much of V_H is progress\n",
        f"- share of Var(V_H) explained by the {a.n_bins} kappa bins: **{report['vh_variance_explained_by_kappa_bins']:.4f}**",
        f"- corr(V_H(s_i), b_k(i)) = **{report['corr_vh_bk']:.4f}**",
        f"- sigma_A: PRe {report['sigma_A_pre']:.5f} -> V_H {report['sigma_A_vh']:.5f}",
        f"- corr(A^V, A^PRe) = {report['corr_A_vh_A_pre']:.4f}; within-bin Spearman = {report['within_bin_spearman_A']:.4f}\n",
        "## Weights\n",
        "| weights | sigma used | ESS/N | clip % | max w | std w | Pearson vs PRe | within-bin Spearman vs PRe |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in ("pre", "vh", "vh_presigma"):
        if name in report["weights"]:
            e = report["weights"][name]
            lines.append(
                f"| {name} | {e['sigma_used']:.5f} | {e['ess_frac']:.3f} | {100 * e['clip_frac']:.2f} | {e['max_w']:.2f} | {e['std_w']:.3f} | "
                f"{e.get('pearson_w_vs_pre', float('nan')):.3f} | {e.get('within_bin_spearman_w_vs_pre', float('nan')):.3f} |"
            )
    lines += ["\n## Weight mass per progress bin (%)\n", "| bin | data | " + " | ".join(n_ for n_ in ("pre", "vh", "vh_presigma") if n_ in report["weights"]) + " |", "|---|---|" + "---|" * sum(n_ in report["weights"] for n_ in ("pre", "vh", "vh_presigma"))]
    for k in range(a.n_bins):
        row = f"| {k} | {100 * (bins == k).mean():.2f} | "
        row += " | ".join(f"{100 * report['weights'][n_]['bin_weight_mass'][k]:.2f}" for n_ in ("pre", "vh", "vh_presigma") if n_ in report["weights"])
        lines.append(row + " |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[report] {out_dir / 'report.md'}")
    for name in variants:
        e = report["weights"][name]
        print(f"[{name}] sigma={e['sigma_used']:.5f} ESS/N={e['ess_frac']:.3f} clip%={100 * e['clip_frac']:.2f} max_w={e['max_w']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
