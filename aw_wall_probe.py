#!/usr/bin/env python3
"""
Wall-bin Q-contrast probe  (FINAL measurement of this project phase).

For each checkpoint, on FIXED dataset rows in wall bins {4,5}:
  split rows SURV/FAIL by episode outcome (same rule as aw_measurement_c.py:
  FAIL = episode ends bad within [b, b+1]), then compute

    Q_surv, Q_fail : mean twin-min Q(s, a_data) per (bin, label) cell
    Delta          : Q_surv - Q_fail            (anchor contrast in Q-space)
    span           : p99 - p01 of Q(s, a_data) on a fixed global 20k sample
    Delta_hat      : Delta / span               (contrast survives shrinkage?)
    d_surv, d_fail : mean ||pi(s) - a_data||    (freeze signature, actor side)

Output: probe_results.csv, one row per (run, step, bin).

The ONLY thing you must fill in is build_scorer() below — paste the same
ckpt-load + normalize + critic/actor forward code you already used for the
IQL weight-stats probe. Everything else is complete.

Usage:
  python aw_wall_probe.py <h5> \
      --ckpt aw,cql,100000,/path/aw/model_0100000.pt \
      --ckpt aw,cql,300000,/path/aw/model_0300000.pt \
      --ckpt b,cql,300000,/path/os_aw/model_0300000.pt \
      --ckpt iql,iql,300000,/path/iql/model_0300000.pt \
      [--bins 4 5] [--per-cell 2000] [--span-n 20000] [--dry-run]

  --ckpt format: RUNLABEL,ALGO,STEP,PATH   (ALGO in {cql, iql}; cql covers AW/B too)
  --dry-run: random stub instead of real models -> verifies plumbing + CSV.
"""
import argparse, csv, os, sys
import numpy as np


# ----------------------------------------------------------------------------
# ADAPTER — the one block you fill in (reuse your weight-stats probe code).
# Must return two callables mapping numpy -> numpy:
#   q_fn(critic_obs [B,Dc], act [B,Da]) -> Q [B]      (twin-min, RAW network obs
#                                                      i.e. apply the SAME critic_obs
#                                                      normalizer the trainer used)
#   pi_fn(obs [B,Do]) -> act [B,Da]                    (deterministic actor mean)
# ----------------------------------------------------------------------------
def build_scorer(algo: str, ckpt_path: str, device: str = "cuda"):
    """Restore a scalar CQL-family or IQL checkpoint as NumPy scoring callables."""
    from pathlib import Path
    from typing import Mapping

    repo_root = Path(__file__).resolve().parent
    source_root = repo_root / "src" / "holosoma"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from holosoma.agents.cql.cql import Actor, DoubleQCritic
    from holosoma.utils.safe_torch_import import torch

    algo = algo.lower()
    if algo not in {"cql", "iql"}:
        raise ValueError(f"unsupported algo '{algo}'; expected one of: cql, iql")

    torch_device = torch.device(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in checkpoint or "qnet_state_dict" not in checkpoint:
        raise KeyError(f"checkpoint lacks actor_state_dict or qnet_state_dict: {ckpt_path}")

    def strip_prefixes(state_dict):
        cleaned = {}
        for original_key, value in state_dict.items():
            key = original_key
            changed = True
            while changed:
                changed = False
                for prefix in ("module.", "_orig_mod."):
                    if key.startswith(prefix):
                        key = key[len(prefix) :]
                        changed = True
            cleaned[key] = value
        return cleaned

    raw_args = checkpoint.get("args", {})
    args = dict(raw_args) if isinstance(raw_args, Mapping) else vars(raw_args)
    if bool(args.get("use_cnn_encoder", False)):
        raise NotImplementedError("aw_wall_probe.py supports the WBT MLP actor, not CNNActor checkpoints")

    actor_state = strip_prefixes(checkpoint["actor_state_dict"])
    critic_state = strip_prefixes(checkpoint["qnet_state_dict"])
    if "net.0.weight" not in actor_state or "fc_mu.0.weight" not in actor_state:
        raise ValueError(f"unsupported actor state layout: {ckpt_path}")
    if "q1.net.0.weight" not in critic_state:
        raise ValueError(f"unsupported twin-Q state layout: {ckpt_path}")

    actor_obs_dim = int(actor_state["net.0.weight"].shape[1])
    actor_hidden_dim = int(actor_state["net.0.weight"].shape[0])
    action_dim = int(actor_state["fc_mu.0.weight"].shape[0])
    critic_hidden_dim = int(critic_state["q1.net.0.weight"].shape[0])
    critic_obs_dim = int(critic_state["q1.net.0.weight"].shape[1]) - action_dim
    use_layer_norm = bool(args.get("use_layer_norm", "net.1.weight" in actor_state))

    def obs_layout(size, key):
        return {key: {"start": 0, "end": size, "size": size}}

    actor = Actor(
        obs_indices=obs_layout(actor_obs_dim, "actor_obs"),
        obs_keys=["actor_obs"],
        n_act=action_dim,
        num_envs=1,
        hidden_dim=actor_hidden_dim,
        log_std_max=float(args.get("log_std_max", 0.0)),
        log_std_min=float(args.get("log_std_min", -5.0)),
        use_tanh=bool(args.get("use_tanh", True)),
        use_layer_norm=use_layer_norm,
        device=torch_device,
    )
    critic = DoubleQCritic(
        obs_indices=obs_layout(critic_obs_dim, "critic_obs"),
        obs_keys=["critic_obs"],
        n_act=action_dim,
        hidden_dim=critic_hidden_dim,
        use_layer_norm=use_layer_norm,
        device=torch_device,
    )
    actor.load_state_dict(actor_state, strict=True)
    critic.load_state_dict(critic_state, strict=True)
    actor.eval()
    critic.eval()

    normalize_observations = bool(args.get("obs_normalization", True))
    actor_normalizer = strip_prefixes(checkpoint.get("obs_normalizer_state") or {})
    critic_normalizer = strip_prefixes(checkpoint.get("critic_obs_normalizer_state") or {})

    def normalize(values, state, expected_dim, label):
        if values.ndim != 2 or values.shape[1] != expected_dim:
            raise ValueError(f"{label} shape mismatch: got {tuple(values.shape)}, expected [B, {expected_dim}]")
        if not normalize_observations:
            return values
        if "_mean" not in state or "_std" not in state:
            raise ValueError(f"checkpoint enables obs normalization but has no valid {label} normalizer")
        mean = state["_mean"].to(device=torch_device, dtype=values.dtype)
        std = state["_std"].to(device=torch_device, dtype=values.dtype)
        if mean.shape[-1] != expected_dim or std.shape[-1] != expected_dim:
            raise ValueError(f"stored {label} normalizer does not match dimension {expected_dim}")
        return (values - mean) / (std + 1e-2)

    def q_fn(critic_obs, act):
        with torch.inference_mode():
            critic_obs_tensor = torch.as_tensor(critic_obs, device=torch_device, dtype=torch.float32)
            action_tensor = torch.as_tensor(act, device=torch_device, dtype=torch.float32)
            if action_tensor.ndim != 2 or action_tensor.shape[1] != action_dim:
                raise ValueError(
                    f"action shape mismatch: got {tuple(action_tensor.shape)}, expected [B, {action_dim}]"
                )
            critic_obs_tensor = normalize(
                critic_obs_tensor,
                critic_normalizer,
                critic_obs_dim,
                "critic observation",
            )
            q1, q2 = critic(critic_obs_tensor, action_tensor)
            return torch.minimum(q1, q2).float().cpu().numpy()

    def pi_fn(obs):
        with torch.inference_mode():
            obs_tensor = torch.as_tensor(obs, device=torch_device, dtype=torch.float32)
            obs_tensor = normalize(obs_tensor, actor_normalizer, actor_obs_dim, "actor observation")
            deterministic_action = actor(obs_tensor)[0]
            return deterministic_action.float().cpu().numpy()

    return q_fn, pi_fn


def _stub_scorer(rng):
    def q_fn(co, a):
        return rng.normal(0, 10, size=len(co))
    def pi_fn(o):
        return rng.normal(0, 1, size=(len(o), 3))
    return q_fn, pi_fn


# ---------------------------- data & labels ---------------------------------
def find_key(f, cands):
    for k in cands:
        if k in f:
            return k
    return None


def load_arrays(path):
    import h5py, hashlib
    f = h5py.File(path, "r")
    def get(cands, required=True):
        k = find_key(f, cands)
        if k is None:
            if required:
                raise KeyError(f"none of {cands} in h5; available: {list(f.keys())[:40]}")
            return None
        a = np.asarray(f[k])
        return a.squeeze() if a.ndim > 1 and a.shape[-1] == 1 else a
    r      = get(["rewards", "reward"]).astype(np.float64)
    phase  = get(["motion_phase", "motion_phases", "phase"]).astype(np.float64)
    dones  = get(["dones", "done"]).astype(bool)
    truncs = get(["truncations", "truncation"]).astype(bool)
    bad = None
    bad = None
    for k in ("next_done_bad_tracking", "done_bad_tracking"):
        if k in f:
            bad = np.asarray(f[k]).squeeze().astype(bool); break
    obs    = get(["observations", "obs", "actor_observations", "policy_observations"])
    cobs   = get(["critic_observations", "critic_obs"], required=False)
    acts   = get(["actions", "action"])
    f.close()
    if cobs is None:
        cobs = obs
    assert bad is not None, "bad-tracking flag missing"
    rh = None
    try:
        import hashlib as _h
        rh = _h.sha256(np.ascontiguousarray(r[:1000]).tobytes()
                       + np.ascontiguousarray(r[-1000:]).tobytes()).hexdigest()[:16]
    except Exception:
        pass
    return r, phase, dones, truncs, bad, obs, cobs, acts, rh


def label_rows(phase, dones, truncs, bad, n_bins=20):
    end = (dones | truncs).copy(); end[-1] = True
    ends = np.flatnonzero(end)
    starts = np.concatenate([[0], ends[:-1] + 1])
    N = len(phase)
    ep_id = np.zeros(N, np.int64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        ep_id[s:e + 1] = i
    bins = np.clip((phase * n_bins).astype(int), 0, n_bins - 1)
    term_bin, term_bad = bins[ends], bad[ends]
    return bins, ep_id, term_bin, term_bad


def select_rows(bins, ep_id, term_bin, term_bad, wall_bins, per_cell, span_n,
                cache, rhash, seed=0):
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if rhash is None or str(z["rhash"]) == str(rhash):
            print(f"[rows] loaded cached indices from {cache}")
            cached_cells = {
                (int(k[0]), str(k[1])): np.asarray(v, dtype=np.int64)
                for k, v in zip(z["cell_keys"], z["cell_idx"])
            }
            required_cells = {(int(b), lab) for b in wall_bins for lab in ("SURV", "FAIL")}
            if required_cells.issubset(cached_cells):
                return cached_cells, np.asarray(z["span_idx"], dtype=np.int64)
            print("[rows] cache does not contain all requested bins -> reselecting")
        else:
            print("[rows] cache rhash mismatch -> reselecting")
    rng = np.random.default_rng(seed)
    cells = {}
    for b in wall_bins:
        m = bins == b
        fail_ep = term_bad & (term_bin >= b) & (term_bin <= b + 1)
        is_fail = fail_ep[ep_id[m]]
        idx = np.flatnonzero(m)
        for lab, lm in (("SURV", ~is_fail), ("FAIL", is_fail)):
            pool = idx[lm]
            take = pool if len(pool) <= per_cell else rng.choice(pool, per_cell, replace=False)
            cells[(b, lab)] = np.sort(take)
            print(f"[rows] bin {b} {lab}: {len(take):,} rows (pool {len(pool):,})")
    span_idx = np.sort(rng.choice(len(bins), min(span_n, len(bins)), replace=False))
    if cache:
        np.savez_compressed(cache,
                            cell_keys=np.array(list(cells.keys())),
                            cell_idx=np.array(list(cells.values()), dtype=object),
                            span_idx=span_idx, rhash=str(rhash))
    return cells, span_idx


def batched(fn, *arrs, bs=4096):
    outs = []
    for i in range(0, len(arrs[0]), bs):
        outs.append(fn(*[a[i:i + bs] for a in arrs]))
    return np.concatenate(outs)


# --------------------------------- main -------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--ckpt", action="append", required=True,
                    help="RUNLABEL,ALGO,STEP,PATH (repeatable)")
    ap.add_argument("--bins", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--per-cell", type=int, default=2000)
    ap.add_argument("--span-n", type=int, default=20000)
    ap.add_argument("--index-cache", default="probe_rows.npz")
    ap.add_argument("--out", default="probe_results.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    r, phase, dones, truncs, bad, obs, cobs, acts, rh = load_arrays(a.h5)
    print(f"[load] N={len(r):,}  rhash={rh}")
    bins, ep_id, term_bin, term_bad = label_rows(phase, dones, truncs, bad)
    cells, span_idx = select_rows(bins, ep_id, term_bin, term_bad, a.bins,
                                  a.per_cell, a.span_n, a.index_cache, rh)

    rows_out = []
    stub_rng = np.random.default_rng(1)
    for spec in a.ckpt:
        run, algo, step, path = spec.split(",", 3)
        step = int(step)
        if a.dry_run:
            q_fn, pi_fn = _stub_scorer(stub_rng)
        else:
            assert os.path.exists(path), f"ckpt not found: {path}"
            q_fn, pi_fn = build_scorer(algo, path, a.device)
        q_span = batched(q_fn, cobs[span_idx], acts[span_idx])
        span = float(np.percentile(q_span, 99) - np.percentile(q_span, 1))
        for b in a.bins:
            stats = {}
            for lab in ("SURV", "FAIL"):
                idx = cells[(b, lab)]
                q = batched(q_fn, cobs[idx], acts[idx])
                pa = batched(pi_fn, obs[idx])
                d = np.linalg.norm(pa - acts[idx], axis=-1)
                stats[lab] = (float(q.mean()), float(d.mean()), len(idx))
            (qs, ds, ns), (qf, df, nf) = stats["SURV"], stats["FAIL"]
            delta = qs - qf
            rows_out.append(dict(run=run, algo=algo, step=step, bin=b,
                                 n_surv=ns, n_fail=nf,
                                 Q_surv=round(qs, 4), Q_fail=round(qf, 4),
                                 Delta=round(delta, 4), span=round(span, 4),
                                 Delta_hat=round(delta / max(span, 1e-9), 5),
                                 d_surv=round(ds, 4), d_fail=round(df, 4),
                                 d_ratio=round(df / max(ds, 1e-9), 4)))
            print(f"[{run} @{step} bin{b}] Δ={delta:+.3f} span={span:.2f} "
                  f"Δ̂={delta/max(span,1e-9):+.4f}  d_ratio={df/max(ds,1e-9):.3f}")

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print(f"[saved] {a.out}  ({len(rows_out)} rows)")


if __name__ == "__main__":
    sys.exit(main())
