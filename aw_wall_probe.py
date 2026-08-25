#!/usr/bin/env python3
"""
Wall-bin Q-contrast probe  (FINAL measurement of this project phase).

For each checkpoint, on FIXED dataset rows in wall bins {4,5}:
  split rows SURV/FAIL by episode outcome (same rule as aw_measurement_c.py:
  FAIL = episode ends bad within [b, b+1]), then compute

    Q_surv, Q_fail : mean twin-min Q(s, a_data) per (bin, label) cell
    Delta          : Q_surv - Q_fail            (anchor contrast in Q-space)
    span           : p99 - p01 of Q(s, a_data) on the configured global rows
    Delta_hat      : Delta / span               (contrast survives shrinkage?)
    d_surv, d_fail : mean ||pi(s) - a_data||    (freeze signature, actor side)

Output: probe_results.csv, one row per (run, step, bin).

The ONLY thing you must fill in is build_scorer() below — paste the same
ckpt-load + normalize + critic/actor forward code you already used for the
IQL weight-stats probe. Everything else is complete.

Usage:
  # Cell-1 defaults: the dataset, bins 4/5, fixed-row cache, and a 1k grid
  # through the final saved checkpoint are built in. Only the run is required.
  python aw_wall_probe.py --ckpt-dir logs/WholeBodyTracking/<run>

  # Checkpoint directory + retrospective grid. A:B includes every saved
  # checkpoint in the inclusive interval, while individual steps are exact.
  python aw_wall_probe.py <h5> \
      --ckpt-dir logs/WholeBodyTracking/<run> \
      --steps '20k,60k,100k,140k,170k,180k:220k,230k,260k,300k' \
      --run-label cql_alpha1 --algo cql \
      --bins 4 5 --index-cache probe_rows_cell1.npz

  python aw_wall_probe.py <h5> \
      --ckpt aw,cql,100000,/path/aw/model_0100000.pt \
      --ckpt aw,cql,300000,/path/aw/model_0300000.pt \
      --ckpt b,cql,300000,/path/os_aw/model_0300000.pt \
      --ckpt iql,iql,300000,/path/iql/model_0300000.pt \
      [--bins 4 5] [--per-cell all] [--span-n 3000000] [--dry-run]

  --ckpt format: RUNLABEL,ALGO,STEP,PATH   (ALGO in {cql, iql, td3bc}; cql covers AW variants)
  --dry-run: random stub instead of real models -> verifies plumbing + CSV.
"""
import argparse, csv, hashlib, os, re, sys
from pathlib import Path

import numpy as np


CHECKPOINT_PATTERN = re.compile(r"^model_(\d+)\.pt$")
DEFAULT_H5 = "offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5"
DEFAULT_GRID = "1k:end:1k"
DEFAULT_INDEX_CACHE = "probe_rows_cell1_v3_full.npz"
DEFAULT_OUTPUT = "probe_results_cell1_v3_full.csv"
DEFAULT_SPAN_N = 3_000_000


def parse_step(value: str | int) -> int:
    """Parse checkpoint steps such as ``20000``, ``20k``, or ``0.3m``."""
    if isinstance(value, int):
        return value
    text = str(value).strip().lower().replace("_", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([km]?)", text)
    if match is None:
        raise ValueError(f"invalid checkpoint step: {value!r}")
    scale = {"": 1, "k": 1_000, "m": 1_000_000}[match.group(2)]
    parsed = float(match.group(1)) * scale
    if not parsed.is_integer():
        raise ValueError(f"checkpoint step must be integral: {value!r}")
    return int(parsed)


def discover_checkpoints(directory: str | Path) -> list[tuple[int, Path]]:
    """Discover unique ``model_{step}.pt`` checkpoints below a run directory."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"checkpoint directory not found: {root}")
    paths = list(root.glob("model_*.pt")) or list(root.rglob("model_*.pt"))
    by_step: dict[int, Path] = {}
    for path in paths:
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step in by_step:
            raise ValueError(f"duplicate checkpoint step {step}: {by_step[step]} and {path}")
        by_step[step] = path
    if not by_step:
        raise FileNotFoundError(f"no model_{{step}}.pt checkpoints found under {root}")
    return sorted(by_step.items())


def select_checkpoint_grid(
    checkpoints: list[tuple[int, Path]],
    grid: str | None,
    *,
    label: str = "run",
) -> list[tuple[int, Path]]:
    """Select exact steps and inclusive saved-checkpoint ranges from ``grid``.

    A token such as ``20k`` requests an exact checkpoint. ``180k:220k`` or
    ``180k-220k`` selects every checkpoint actually saved in that interval.
    Missing exact steps are reported but do not abort the retrospective scan.
    """
    if not grid:
        print(f"[grid:{label}] no grid supplied -> using all {len(checkpoints)} checkpoints")
        return checkpoints

    by_step = dict(checkpoints)
    selected: set[int] = set()
    missing: list[int] = []
    range_reports: list[tuple[int, int, list[int]]] = []
    tokens = [token.strip() for token in grid.split(",") if token.strip()]
    if not tokens:
        raise ValueError("checkpoint grid is empty")

    for token in tokens:
        stepped_match = re.fullmatch(r"(.+?)\s*:\s*(end|.+?)\s*:\s*(.+)", token)
        if stepped_match is not None:
            start = parse_step(stepped_match.group(1))
            end_text = stepped_match.group(2).strip().lower()
            end = checkpoints[-1][0] if end_text == "end" else parse_step(end_text)
            stride = parse_step(stepped_match.group(3))
            if stride <= 0:
                raise ValueError(f"checkpoint grid stride must be positive: {token!r}")
            if start > end:
                raise ValueError(f"checkpoint grid start exceeds end: {token!r}")
            expected = list(range(start, end + 1, stride))
            found = [step for step in expected if step in by_step]
            missing_grid = [step for step in expected if step not in by_step]
            selected.update(found)
            print(
                f"[grid:{label}] stepped grid [{start}, {end}] stride={stride}: "
                f"found={len(found)}/{len(expected)}"
            )
            if missing_grid:
                preview = missing_grid[:20]
                suffix = " ..." if len(missing_grid) > len(preview) else ""
                print(
                    f"[grid:{label}] WARNING missing grid checkpoints "
                    f"({len(missing_grid)}): {preview}{suffix}"
                )
            continue
        range_match = re.fullmatch(r"(.+?)\s*[:-]\s*(.+)", token)
        if range_match is None:
            step = parse_step(token)
            if step in by_step:
                selected.add(step)
            else:
                missing.append(step)
            continue
        start = parse_step(range_match.group(1))
        end = parse_step(range_match.group(2))
        if start > end:
            raise ValueError(f"checkpoint range start exceeds end: {token!r}")
        found = [step for step, _ in checkpoints if start <= step <= end]
        selected.update(found)
        range_reports.append((start, end, found))

    for start, end, found in range_reports:
        if found:
            gaps = [right - left for left, right in zip(found, found[1:])]
            max_gap = max(gaps) if gaps else 0
            print(
                f"[grid:{label}] dense range [{start}, {end}]: "
                f"found={found}, max_saved_gap={max_gap}"
            )
        else:
            print(f"[grid:{label}] WARNING dense range [{start}, {end}] has no saved checkpoint")
    if missing:
        print(f"[grid:{label}] WARNING missing exact checkpoints: {sorted(set(missing))}")
    chosen = [(step, by_step[step]) for step in sorted(selected)]
    if not chosen:
        raise ValueError(f"checkpoint grid selected no available checkpoints for {label}")
    chosen_steps = [step for step, _ in chosen]
    gaps = [right - left for left, right in zip(chosen_steps, chosen_steps[1:])]
    print(
        f"[grid:{label}] selected={chosen_steps}; "
        f"temporal_resolution_max_gap={max(gaps) if gaps else 0}"
    )
    return chosen


def checkpoint_specs_from_directory(
    directory: str | Path,
    grid: str | None,
    run_label: str | None,
    algo: str,
) -> list[str]:
    root = Path(directory).expanduser()
    label = run_label or root.name
    selected = select_checkpoint_grid(discover_checkpoints(root), grid, label=label)
    return [f"{label},{algo},{step},{path}" for step, path in selected]


# ----------------------------------------------------------------------------
# ADAPTER — the one block you fill in (reuse your weight-stats probe code).
# Must return two callables mapping numpy -> numpy:
#   q_fn(critic_obs [B,Dc], act [B,Da]) -> Q [B]      (twin-min, RAW network obs
#                                                      i.e. apply the SAME critic_obs
#                                                      normalizer the trainer used)
#   pi_fn(obs [B,Do]) -> act [B,Da]                    (deterministic actor mean)
# ----------------------------------------------------------------------------
def build_scorer(
    algo: str,
    ckpt_path: str,
    device: str = "cuda",
    *,
    critic_state_key: str = "qnet_state_dict",
    include_target_q: bool = False,
):
    """Restore an offline-RL checkpoint as NumPy scoring callables.

    ``critic_state_key`` defaults to the online critic used by the original
    wall probe. Retrospective audits may request ``include_target_q=True`` to
    receive a third callable backed by ``qnet_target_state_dict`` without
    loading the checkpoint twice.
    """
    from pathlib import Path
    from typing import Mapping

    repo_root = Path(__file__).resolve().parent
    source_root = repo_root / "src" / "holosoma"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from holosoma.agents.cql.cql import Actor as StochasticActor
    from holosoma.agents.cql.cql import DoubleQCritic
    from holosoma.agents.td3.td3 import Actor as DeterministicActor
    from holosoma.utils.safe_torch_import import torch

    algo = algo.lower()
    if algo not in {"cql", "iql", "td3bc"}:
        raise ValueError(f"unsupported algo '{algo}'; expected one of: cql, iql, td3bc")

    torch_device = torch.device(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in checkpoint or critic_state_key not in checkpoint:
        raise KeyError(
            f"checkpoint lacks actor_state_dict or {critic_state_key}: {ckpt_path}"
        )

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
    critic_state = strip_prefixes(checkpoint[critic_state_key])

    def state_checksum(state_dict) -> str:
        digest = hashlib.sha256()
        for key in sorted(state_dict):
            tensor = state_dict[key].detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(np.asarray(tensor.float()).tobytes())
        return digest.hexdigest()[:16]

    critic_checksum = state_checksum(critic_state)
    actor_output_key = "fc_mu.weight" if algo == "td3bc" else "fc_mu.0.weight"
    if "net.0.weight" not in actor_state or actor_output_key not in actor_state:
        raise ValueError(f"unsupported actor state layout: {ckpt_path}")
    if "q1.net.0.weight" not in critic_state:
        raise ValueError(f"unsupported twin-Q state layout: {ckpt_path}")

    actor_obs_dim = int(actor_state["net.0.weight"].shape[1])
    actor_hidden_dim = int(actor_state["net.0.weight"].shape[0])
    action_dim = int(actor_state[actor_output_key].shape[0])
    critic_hidden_dim = int(critic_state["q1.net.0.weight"].shape[0])
    critic_obs_dim = int(critic_state["q1.net.0.weight"].shape[1]) - action_dim
    use_layer_norm = bool(args.get("use_layer_norm", "net.1.weight" in actor_state))

    def obs_layout(size, key):
        return {key: {"start": 0, "end": size, "size": size}}

    actor_kwargs = dict(
        obs_indices=obs_layout(actor_obs_dim, "actor_obs"),
        obs_keys=["actor_obs"],
        n_act=action_dim,
        num_envs=1,
        hidden_dim=actor_hidden_dim,
        use_tanh=bool(args.get("use_tanh", True)),
        use_layer_norm=use_layer_norm,
        device=torch_device,
    )
    if algo == "td3bc":
        actor = DeterministicActor(**actor_kwargs)
    else:
        actor = StochasticActor(
            **actor_kwargs,
            log_std_max=float(args.get("log_std_max", 0.0)),
            log_std_min=float(args.get("log_std_min", -5.0)),
        )
    critic = DoubleQCritic(
        obs_indices=obs_layout(critic_obs_dim, "critic_obs"),
        obs_keys=["critic_obs"],
        n_act=action_dim,
        hidden_dim=critic_hidden_dim,
        use_layer_norm=use_layer_norm,
        device=torch_device,
    )
    target_critic = None
    if include_target_q:
        if "qnet_target_state_dict" not in checkpoint:
            raise KeyError(f"checkpoint lacks qnet_target_state_dict: {ckpt_path}")
        target_critic = DoubleQCritic(
            obs_indices=obs_layout(critic_obs_dim, "critic_obs"),
            obs_keys=["critic_obs"],
            n_act=action_dim,
            hidden_dim=critic_hidden_dim,
            use_layer_norm=use_layer_norm,
            device=torch_device,
        )
    actor.load_state_dict(actor_state, strict=True)
    critic.load_state_dict(critic_state, strict=True)
    if target_critic is not None:
        target_critic.load_state_dict(
            strip_prefixes(checkpoint["qnet_target_state_dict"]),
            strict=True,
        )
    actor.eval()
    critic.eval()
    if target_critic is not None:
        target_critic.eval()

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

    def score_critic(model, critic_obs, act):
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
            q1, q2 = model(critic_obs_tensor, action_tensor)
            return torch.minimum(q1, q2).float().cpu().numpy()

    def q_fn(critic_obs, act):
        return score_critic(critic, critic_obs, act)

    def direct_reference_q_fn(critic_obs, act):
        """Independent MLP Q path bypassing q_fn and critic/QNetwork.forward."""
        with torch.inference_mode():
            critic_obs_tensor = torch.as_tensor(
                critic_obs,
                device=torch_device,
                dtype=torch.float32,
            )
            action_tensor = torch.as_tensor(act, device=torch_device, dtype=torch.float32)
            critic_obs_tensor = normalize(
                critic_obs_tensor,
                critic_normalizer,
                critic_obs_dim,
                "critic observation",
            )
            critic_input = torch.cat([critic_obs_tensor, action_tensor], dim=1)
            q1 = critic.q1.net(critic_input).squeeze(-1)
            q2 = critic.q2.net(critic_input).squeeze(-1)
            return torch.minimum(q1, q2).float().cpu().numpy()

    q_fn.probe_metadata = {
        "critic_checksum": critic_checksum,
        "first_layer_sum": float(critic_state["q1.net.0.weight"].float().sum().item()),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "checkpoint_path": str(ckpt_path),
    }
    q_fn.direct_reference_q_fn = direct_reference_q_fn

    def pi_fn(obs):
        with torch.inference_mode():
            obs_tensor = torch.as_tensor(obs, device=torch_device, dtype=torch.float32)
            obs_tensor = normalize(obs_tensor, actor_normalizer, actor_obs_dim, "actor observation")
            deterministic_action = actor(obs_tensor)[0]
            return deterministic_action.float().cpu().numpy()

    if target_critic is not None:
        def target_q_fn(critic_obs, act):
            return score_critic(target_critic, critic_obs, act)

        return q_fn, pi_fn, target_q_fn
    return q_fn, pi_fn


def _stub_scorer(rng, action_dim):
    def q_fn(co, a):
        return rng.normal(0, 10, size=len(co))
    def pi_fn(o):
        return rng.normal(0, 1, size=(len(o), action_dim))
    q_fn.probe_metadata = {
        "critic_checksum": "dry-run",
        "first_layer_sum": float("nan"),
        "checkpoint_global_step": -1,
        "checkpoint_path": "dry-run",
    }
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
                cache, rhash, seed=0, strict_cache=False):
    per_cell_mode = "all" if per_cell is None else str(per_cell)
    span_n_mode = "all" if span_n >= len(bins) else str(span_n)
    if cache and os.path.exists(cache):
        with np.load(cache, allow_pickle=True) as z:
            cached_hash = str(np.asarray(z["rhash"]).item()) if "rhash" in z.files else ""
            cache_per_cell_mode = (
                str(np.asarray(z["per_cell_mode"]).item()) if "per_cell_mode" in z.files else ""
            )
            cache_span_n_mode = (
                str(np.asarray(z["span_n_mode"]).item()) if "span_n_mode" in z.files else ""
            )
            hash_matches = rhash is None or cached_hash == str(rhash)
            mode_mismatches = []
            if cache_per_cell_mode != per_cell_mode:
                mode_mismatches.append(
                    f"per-cell cache={cache_per_cell_mode or '<missing>'} current={per_cell_mode}"
                )
            if cache_span_n_mode != span_n_mode:
                mode_mismatches.append(
                    f"span-n cache={cache_span_n_mode or '<missing>'} current={span_n_mode}"
                )

            if not hash_matches:
                if strict_cache:
                    raise ValueError(
                        f"strict index cache rhash mismatch: cache={cached_hash}, dataset={rhash}"
                    )
                print("[rows] cache rhash mismatch -> reselecting")
            elif mode_mismatches:
                message = "; ".join(mode_mismatches)
                if strict_cache:
                    raise ValueError(f"strict index cache selection mode mismatch: {message}")
                print(f"[rows] cache selection mode mismatch ({message}) -> reselecting")
            else:
                print(
                    f"[rows] loaded cached indices from {cache} "
                    f"(per-cell={per_cell_mode}, span-n={span_n_mode})"
                )
                cached_cells = {
                    (int(k[0]), str(k[1])): np.asarray(v, dtype=np.int64)
                    for k, v in zip(z["cell_keys"], z["cell_idx"])
                }
                required_cells = {(int(b), lab) for b in wall_bins for lab in ("SURV", "FAIL")}
                if required_cells.issubset(cached_cells):
                    return cached_cells, np.asarray(z["span_idx"], dtype=np.int64)
                if strict_cache:
                    missing = sorted(required_cells.difference(cached_cells))
                    raise KeyError(f"strict index cache lacks requested cells: {missing}")
                print("[rows] cache does not contain all requested bins -> reselecting")
    elif strict_cache:
        raise FileNotFoundError(f"strict index cache not found: {cache}")
    rng = np.random.default_rng(seed)
    cells = {}
    for b in wall_bins:
        m = bins == b
        fail_ep = term_bad & (term_bin >= b) & (term_bin <= b + 1)
        is_fail = fail_ep[ep_id[m]]
        idx = np.flatnonzero(m)
        for lab, lm in (("SURV", ~is_fail), ("FAIL", is_fail)):
            pool = idx[lm]
            if per_cell is None or len(pool) <= per_cell:
                take = pool
            else:
                take = rng.choice(pool, per_cell, replace=False)
            cells[(b, lab)] = np.sort(take)
            print(f"[rows] bin {b} {lab}: {len(take):,} rows (pool {len(pool):,})")
    if span_n >= len(bins):
        span_idx = np.arange(len(bins), dtype=np.int64)
    else:
        span_idx = np.sort(rng.choice(len(bins), span_n, replace=False))
    print(f"[rows] span: {len(span_idx):,} rows (mode={span_n_mode})")
    if cache:
        np.savez_compressed(cache,
                            cell_keys=np.array(list(cells.keys())),
                            cell_idx=np.array(list(cells.values()), dtype=object),
                            span_idx=span_idx, rhash=str(rhash),
                            per_cell_mode=per_cell_mode,
                            span_n_mode=span_n_mode)
    return cells, span_idx


def _index_checksum(indices: np.ndarray) -> str:
    values = np.ascontiguousarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def validate_selected_rows(
    bins: np.ndarray,
    ep_id: np.ndarray,
    term_bin: np.ndarray,
    term_bad: np.ndarray,
    wall_bins: list[int],
    cells: dict[tuple[int, str], np.ndarray],
    span_idx: np.ndarray,
    *,
    require_full_cells: bool,
    require_full_span: bool,
    verbose: bool = False,
) -> None:
    """Verify that cached row coordinates still encode the requested cells.

    This is deliberately recomputed from the current H5 labels. It prevents a
    stale or malformed object-array cache from silently pairing a cell name
    with rows from another phase/outcome cell.
    """
    num_rows = len(bins)
    row_bad_failure = term_bad[ep_id]
    row_terminal_bin = term_bin[ep_id]

    for bin_index in wall_bins:
        in_bin = bins == bin_index
        is_fail = row_bad_failure & (row_terminal_bin >= bin_index) & (
            row_terminal_bin <= bin_index + 1
        )
        for label, label_mask in (("SURV", ~is_fail), ("FAIL", is_fail)):
            key = (int(bin_index), label)
            if key not in cells:
                raise KeyError(f"selected rows lack requested cell {key}")
            selected = np.asarray(cells[key], dtype=np.int64)
            if selected.ndim != 1:
                raise ValueError(f"cell {key} indices must be one-dimensional, got {selected.shape}")
            if selected.size == 0:
                raise ValueError(f"cell {key} has no rows")
            if selected.min() < 0 or selected.max() >= num_rows:
                raise IndexError(f"cell {key} contains out-of-range dataset indices")
            if np.unique(selected).size != selected.size:
                raise ValueError(f"cell {key} contains duplicate indices")

            expected = np.flatnonzero(in_bin & label_mask)
            if require_full_cells:
                if not np.array_equal(selected, expected):
                    raise ValueError(
                        f"full cell {key} does not exactly match current H5 labels: "
                        f"selected={len(selected)}, expected={len(expected)}"
                    )
            elif not np.all(np.isin(selected, expected, assume_unique=False)):
                raise ValueError(f"cell {key} contains rows from another phase/outcome cell")
            if verbose:
                print(
                    f"[audit:rows] cell={key} rows={len(selected):,} "
                    f"checksum={_index_checksum(selected)} "
                    f"first={selected[:5].tolist()}"
                )

    span_idx = np.asarray(span_idx, dtype=np.int64)
    if span_idx.ndim != 1 or span_idx.size == 0:
        raise ValueError("span indices must be a non-empty one-dimensional array")
    if span_idx.min() < 0 or span_idx.max() >= num_rows:
        raise IndexError("span indices contain out-of-range dataset rows")
    if np.unique(span_idx).size != span_idx.size:
        raise ValueError("span indices contain duplicates")
    if require_full_span and not np.array_equal(span_idx, np.arange(num_rows, dtype=np.int64)):
        raise ValueError("full span cache is not the identity row ordering")
    if verbose:
        print(
            f"[audit:rows] span rows={len(span_idx):,} "
            f"checksum={_index_checksum(span_idx)} full={require_full_span}"
        )


def batched(fn, *arrs, bs=4096):
    outs = []
    for i in range(0, len(arrs[0]), bs):
        outs.append(fn(*[a[i:i + bs] for a in arrs]))
    return np.concatenate(outs)


# --------------------------------- main -------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "h5",
        nargs="?",
        default=DEFAULT_H5,
        help=f"offline H5 dataset (default: {DEFAULT_H5})",
    )
    ap.add_argument("--ckpt", action="append", default=[],
                    help="RUNLABEL,ALGO,STEP,PATH (repeatable)")
    ap.add_argument("--ckpt-dir", help="run directory containing model_{step}.pt checkpoints")
    ap.add_argument(
        "--steps",
        default=DEFAULT_GRID,
        help=(
            "checkpoint grid for --ckpt-dir; comma-separated exact steps and inclusive "
            "saved ranges, or start:end:stride (default: 1k:end:1k)"
        ),
    )
    ap.add_argument("--run-label", help="CSV run label for --ckpt-dir (default: directory name)")
    ap.add_argument("--algo", choices=("cql", "iql", "td3bc"), default="cql")
    ap.add_argument("--bins", type=int, nargs="+", default=[4, 5])
    ap.add_argument(
        "--per-cell",
        default="all",
        help="'all' = use each cell's entire pool (deterministic), or an integer sample size",
    )
    ap.add_argument("--span-n", type=int, default=DEFAULT_SPAN_N)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="inference chunk size; lower to 2048 on GPU OOM without changing results",
    )
    ap.add_argument("--index-cache", default=DEFAULT_INDEX_CACHE)
    ap.add_argument(
        "--strict-index-cache",
        action="store_true",
        help="require the existing cache to match dataset rhash and requested bins; never reselect rows",
    )
    ap.add_argument("--out", default=DEFAULT_OUTPUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--audit-sample-n",
        type=int,
        default=0,
        help=(
            "also score a deterministic legacy-size sample from each selected cell; "
            "use 2000 to compare full-pool v3 against the old estimator"
        ),
    )
    a = ap.parse_args()

    checkpoint_specs = list(a.ckpt)
    if a.ckpt_dir:
        checkpoint_specs.extend(
            checkpoint_specs_from_directory(a.ckpt_dir, a.steps, a.run_label, a.algo)
        )
    if not checkpoint_specs:
        ap.error("provide at least one --ckpt or --ckpt-dir")
    if str(a.per_cell).lower() == "all":
        per_cell = None
    else:
        try:
            per_cell = int(a.per_cell)
        except ValueError:
            ap.error("--per-cell must be 'all' or a positive integer")
        if per_cell <= 0:
            ap.error("--per-cell must be 'all' or a positive integer")
    if a.span_n <= 0:
        ap.error("--span-n must be a positive integer")
    if a.batch_size <= 0:
        ap.error("--batch-size must be a positive integer")
    if a.audit_sample_n < 0:
        ap.error("--audit-sample-n must be non-negative")

    r, phase, dones, truncs, bad, obs, cobs, acts, rh = load_arrays(a.h5)
    print(f"[load] N={len(r):,}  rhash={rh}")
    bins, ep_id, term_bin, term_bad = label_rows(phase, dones, truncs, bad)
    cells, span_idx = select_rows(bins, ep_id, term_bin, term_bad, a.bins,
                                  per_cell, a.span_n, a.index_cache, rh,
                                  strict_cache=a.strict_index_cache)
    validate_selected_rows(
        bins,
        ep_id,
        term_bin,
        term_bad,
        a.bins,
        cells,
        span_idx,
        require_full_cells=per_cell is None,
        require_full_span=a.span_n >= len(bins),
        verbose=a.audit_sample_n > 0,
    )

    rows_out = []
    stub_rng = np.random.default_rng(1)
    checksum_paths: dict[str, str] = {}
    for spec in checkpoint_specs:
        run, algo, step, path = spec.split(",", 3)
        step = int(step)
        if a.dry_run:
            q_fn, pi_fn = _stub_scorer(stub_rng, acts.shape[-1])
        else:
            assert os.path.exists(path), f"ckpt not found: {path}"
            q_fn, pi_fn = build_scorer(algo, path, a.device)
        checkpoint_metadata = getattr(q_fn, "probe_metadata", {})
        critic_checksum = str(checkpoint_metadata.get("critic_checksum", "unknown"))
        first_layer_sum = float(checkpoint_metadata.get("first_layer_sum", float("nan")))
        checkpoint_global_step = int(checkpoint_metadata.get("checkpoint_global_step", -1))
        print(
            f"[ckpt] run={run} requested_step={step} loaded_step={checkpoint_global_step} "
            f"critic_checksum={critic_checksum} first_layer_sum={first_layer_sum:+.9f} "
            f"path={path}"
        )
        previous_path = checksum_paths.get(critic_checksum)
        if previous_path is not None and previous_path != path and critic_checksum != "dry-run":
            print(
                f"[ckpt] WARNING identical critic checksum loaded from distinct paths: "
                f"{previous_path} and {path}"
            )
        checksum_paths[critic_checksum] = path

        # All checkpoint-dependent Q evaluation stays inside this loop, after
        # this checkpoint's state dict has been loaded into a fresh critic.
        q_span = batched(q_fn, cobs[span_idx], acts[span_idx], bs=a.batch_size)
        span = float(np.percentile(q_span, 99) - np.percentile(q_span, 1))
        print(
            f"[span] run={run} step={step} span={span:.9f} "
            f"mean={q_span.mean():+.9f} p01={np.percentile(q_span, 1):+.9f} "
            f"p99={np.percentile(q_span, 99):+.9f}"
        )
        if a.audit_sample_n > 0:
            print(
                f"[audit:q] run={run} step={step} global "
                f"mean={q_span.mean():+.6f} std={q_span.std():.6f} "
                f"p01={np.percentile(q_span, 1):+.6f} "
                f"p50={np.percentile(q_span, 50):+.6f} "
                f"p99={np.percentile(q_span, 99):+.6f}"
            )
        audit_rng = np.random.default_rng(0)
        for b in a.bins:
            stats = {}
            for lab in ("SURV", "FAIL"):
                idx = cells[(b, lab)]
                q = batched(q_fn, cobs[idx], acts[idx], bs=a.batch_size)
                pa = batched(pi_fn, obs[idx], bs=a.batch_size)
                d = np.linalg.norm(pa - acts[idx], axis=-1)
                stats[lab] = (float(q.mean()), float(d.mean()), len(idx))
                if a.audit_sample_n > 0:
                    sample_size = min(a.audit_sample_n, len(idx))
                    sample_idx = (
                        idx
                        if sample_size == len(idx)
                        else np.sort(audit_rng.choice(idx, sample_size, replace=False))
                    )
                    q_sample = batched(
                        q_fn,
                        cobs[sample_idx],
                        acts[sample_idx],
                        bs=a.batch_size,
                    )
                    direct_q_fn = getattr(q_fn, "direct_reference_q_fn", None)
                    direct_mean = float("nan")
                    direct_max_abs_diff = float("nan")
                    if direct_q_fn is not None:
                        q_direct = direct_q_fn(cobs[sample_idx], acts[sample_idx])
                        direct_mean = float(q_direct.mean())
                        direct_max_abs_diff = float(np.max(np.abs(q_direct - q_sample)))
                        if not np.allclose(q_direct, q_sample, rtol=1e-5, atol=1e-5):
                            raise RuntimeError(
                                f"independent direct-Q mismatch for run={run}, step={step}, "
                                f"bin={b}, label={lab}: max_abs_diff={direct_max_abs_diff}"
                            )
                    print(
                        f"[audit:q] run={run} step={step} bin={b} label={lab} "
                        f"full_n={len(idx):,} full_mean={q.mean():+.6f} "
                        f"full_std={q.std():.6f} sample_n={sample_size:,} "
                        f"sample_mean={q_sample.mean():+.6f} "
                        f"sample_std={q_sample.std():.6f} "
                        f"direct_mean={direct_mean:+.6f} "
                        f"direct_max_abs_diff={direct_max_abs_diff:.3e}"
                    )
            (qs, ds, ns), (qf, df, nf) = stats["SURV"], stats["FAIL"]
            delta = qs - qf
            rows_out.append(dict(run=run, algo=algo, step=step, bin=b,
                                 checkpoint_global_step=checkpoint_global_step,
                                 critic_checksum=critic_checksum,
                                 first_layer_sum=round(first_layer_sum, 9),
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
