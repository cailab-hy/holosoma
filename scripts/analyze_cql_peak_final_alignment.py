#!/usr/bin/env python3
"""Peak vs Final CQL checkpoints: same-state alignment with logged offline behaviour, by motion phase.

Question (descriptive, not causal): as offline CQL degrades late in training, does the Final policy
move closer to the logged dataset actions than the Peak policy, and does that happen particularly
for LOW-utility logged behaviour in failure-prone phases?

Pipeline
--------
1. Fixed evaluation manifest (built once, then always reused; ``--manifest``):
   phase-stratified sample of ``--samples-per-phase`` transitions per phase bin, drawn with
   ``--eval-sampling-seed`` (independent of every training seed). It freezes, per transition:
   H5 row, episode id, motion phase / bin, A^H, utility group, F^H, the raw actor observation and
   the logged action, plus dataset-level constants (per-dimension action std, per-bin failure
   rates, utility thresholds). Every seed and every checkpoint is evaluated on exactly these rows.
2. Definitions reused from the repository, never redefined here:
   * phase bin      : clip(floor(motion_phase * K), 0, K-1)   (== AW sidecar ``phase_bin``)
   * A^H            : ``advantage`` from the AW sidecar written by ``scripts/aw_precompute_weights.py``
                      (H-step truncated return minus the phase-bin baseline); pairing is verified
                      with ``aw_precompute_weights.verify_sidecar``.
   * F^H            : ``dataset_failure_association.future_failure_flags`` (bad tracking within H
                      steps; transitions whose H-step future is censored are excluded, -1 in the
                      manifest).
   * peak / final   : ``phase_failure_alignment.peak_and_final_steps`` (max eval success, earliest
                      on ties / last eval step) when the checkpoint config is generated.
   * policy action  : checkpoint-rebuilt ``Actor`` (``probe_q_level.CheckpointModels``), observation
                      normaliser applied with ``update=False``, deterministic ``actor(obs)[0]``
                      = tanh(mean) * action_scale + action_bias, i.e. ``CQLAgent.get_inference_policy``
                      (``_to_env_actions`` is the identity). Actor in eval mode under ``torch.no_grad``.
3. Metrics (per training seed, then aggregated over seeds from within-seed paired differences):
   d_i = sqrt(mean_j ((a_pi_ij - a_D_ij) / (sigma_j + eps))^2)   same-state, own logged action
   D_P(k), D_F(k), dD(k) = D_F - D_P ; same for the high (+) / low (-) utility groups
   G(k) = D^-(k) - D^+(k)  (>0: closer to high- than low-utility behaviour), dG = G_F - G_P
   Raw (unnormalised) RMSE versions are stored as a sanity check. D_PF(k) is the same-state distance
   between the Peak and the Final action (how much the policy itself moved on logged states).

Outputs (``--output-dir``)
--------------------------
  sample_level_alignment_seed<N>.npz   per transition: d_peak, d_final, delta, raw RMSE, Peak-Final distance,
                                       both policy actions, manifest fields
  phase_summary_per_seed.csv / phase_summary_aggregated.csv
  utility_alignment_per_seed.csv / utility_alignment_aggregated.csv
  utility_alignment_global_quantile_per_seed.csv (supplementary), failure_phase_contrast.csv
  plot1_peak_final_distance.png, plot2_delta_distance.png, plot3_utility_alignment.png,
  plot4_preference_gap.png, plot5_failure_association.png (same phase axis), alignment_by_phase.png
  summary.md, run_info.json. The manifest itself lives at ``--manifest``.

The optional phase-centroid visualisation (spec section 19) is not implemented; the primary
evidence is the same-state distance.

Usage
-----
  # 1) write the checkpoint config from TensorBoard (peak = max eval success, final = last)
  python scripts/analyze_cql_peak_final_alignment.py --write-seed-config analysis/cql_peak_final_checkpoints.yaml \
      --method g1_29dof_wbt_cql --seeds 1 2 3 4 5
  # 2) run the analysis
  python scripts/analyze_cql_peak_final_alignment.py \
      --dataset offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5 \
      --seed-config analysis/cql_peak_final_checkpoints.yaml \
      --num-phase-bins 20 --horizon 50 --samples-per-phase 5000 --eval-sampling-seed 1234 \
      --output-dir analysis/cql_peak_final_alignment
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import aw_precompute_weights as AW  # noqa: E402
import dataset_failure_association as DFA  # noqa: E402
import phase_failure_alignment as PFA  # noqa: E402
from probe_q_level import CheckpointModels  # noqa: E402

from holosoma.agents.modules.q_probe import _normalize  # noqa: E402
from holosoma.utils.safe_torch_import import torch  # noqa: E402

MANIFEST_VERSION = 1
EXPECTED_ACTION_SPACE = "env_scaled_action_training_v1"
HIGH, MID, LOW = 1, 0, -1
CKPT_STEP_RE = re.compile(r"model_(\d+)\.pt$")
# categorical slots 1-3 of the validated reference palette, assigned in fixed order
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_MUTED = "#8a8985"

GROUPS = (("all", None), ("high", HIGH), ("low", LOW))
DIST_KINDS = ("", "_raw")  # normalised RMS distance, raw RMSE sanity check


# --------------------------------------------------------------------------------------- manifest
def sidecar_for(h5: Path, horizon: int, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for cand in (Path(f"{h5}.aw_weights.H{horizon}.npz"), Path(f"{h5}.aw_weights.npz")):
        if cand.exists():
            return cand
    raise SystemExit(f"no AW sidecar for {h5}; run scripts/aw_precompute_weights.py {h5} --H {horizon}")


def manifest_request(args, h5: Path, sidecar: Path) -> dict:
    """Parameters that define the manifest; a reused manifest must match all of them."""
    rhash, n_rows, _ = AW.h5_reward_fingerprint(str(h5))
    return {
        "version": MANIFEST_VERSION,
        "dataset_basename": h5.name,
        "dataset_rows": int(n_rows),
        "dataset_rhash": rhash,
        "sidecar_basename": sidecar.name,
        "horizon": int(args.horizon),
        "num_phase_bins": int(args.num_phase_bins),
        "samples_per_phase": int(args.samples_per_phase),
        "eval_sampling_seed": int(args.eval_sampling_seed),
        "high_quantile": float(args.high_quantile),
        "low_quantile": float(args.low_quantile),
    }


def read_rows_and_action_std(h5: Path, idx: np.ndarray, chunk: int = 262_144) -> tuple[np.ndarray, ...]:
    """Raw actor observations / logged actions at sorted ``idx`` plus the full-dataset action std."""
    with h5py.File(h5, "r") as f:
        n = int(f.attrs.get("num_samples", f["actions"].shape[0]))
        obs_out = np.empty((idx.size, f["observations"].shape[1]), np.float32)
        act_out = np.empty((idx.size, f["actions"].shape[1]), np.float32)
        s1 = np.zeros(f["actions"].shape[1])
        s2 = np.zeros(f["actions"].shape[1])
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            act = np.asarray(f["actions"][lo:hi], dtype=np.float64)
            s1 += act.sum(0)
            s2 += np.square(act).sum(0)
            sel = np.flatnonzero((idx >= lo) & (idx < hi))
            if sel.size:
                obs_out[sel] = f["observations"][lo:hi][idx[sel] - lo]
                act_out[sel] = act[idx[sel] - lo]
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean**2, 0.0))
    return obs_out, act_out, std


def build_manifest(args, h5: Path, sidecar: Path, request: dict) -> dict:
    t0 = time.time()
    nb, horizon = args.num_phase_bins, args.horizon
    print(
        f"[manifest] building from {h5} (H={horizon}, {nb} bins, {args.samples_per_phase}/bin, "
        f"eval_sampling_seed={args.eval_sampling_seed})"
    )
    if not AW.verify_sidecar(str(h5), str(sidecar)):
        raise SystemExit(f"AW sidecar {sidecar} is not paired with {h5}")
    with np.load(sidecar) as sc:
        if int(sc["H"]) != horizon:
            raise SystemExit(f"sidecar {sidecar} has H={int(sc['H'])}, but --horizon {horizon}")
        advantage = np.asarray(sc["advantage"], np.float64)
        side_bins = np.asarray(sc["phase_bin"], np.int64)
        side_gamma = float(sc["gamma"])

    data = DFA.load(h5)
    n = data["episode_id"].size
    if advantage.size != n:
        raise SystemExit(f"sidecar rows {advantage.size} != dataset rows {n}")
    phase = data["motion_phase"].astype(np.float64)
    bins = np.clip((phase * nb).astype(np.int64), 0, nb - 1)  # same formula as aw_precompute_weights
    side_nb = int(side_bins.max()) + 1
    if side_nb == nb and not np.array_equal(side_bins, bins):
        raise SystemExit("phase bins differ from the AW sidecar's phase_bin; phase semantics mismatch")

    flags = DFA.future_failure_flags(data, horizon)
    z, det, end_bad = flags["z"], flags["determinable"], flags["end_bad"]
    with h5py.File(h5, "r") as f:
        dones = np.asarray(f["dones"][:n]).astype(bool)
        truncs = np.asarray(f["truncations"][:n]).astype(bool)
    aw_starts, _ = AW.episode_bounds(dones, truncs)
    n_ep, n_ep_aw = int(flags["starts"].size), int(aw_starts.size)
    if n_ep != n_ep_aw:
        print(f"[WARN] episode count differs: episode_id blocks={n_ep}, AW dones|truncations bounds={n_ep_aw}")

    lo_thr = np.empty(nb)
    hi_thr = np.empty(nb)
    count_ds = np.bincount(bins, minlength=nb)
    for k in range(nb):
        ak = advantage[bins == k]
        lo_thr[k] = np.quantile(ak, args.low_quantile)
        hi_thr[k] = np.quantile(ak, 1.0 - args.high_quantile)
        if not lo_thr[k] < hi_thr[k]:
            raise SystemExit(f"degenerate utility thresholds in bin {k}: low {lo_thr[k]} >= high {hi_thr[k]}")
    group = np.where(advantage >= hi_thr[bins], HIGH, np.where(advantage <= lo_thr[bins], LOW, MID))
    g_lo, g_hi = np.quantile(advantage, [args.low_quantile, 1.0 - args.high_quantile])
    group_global = np.where(advantage >= g_hi, HIGH, np.where(advantage <= g_lo, LOW, MID))

    rng = np.random.default_rng(args.eval_sampling_seed)
    picks = []
    for k in range(nb):
        rows = np.flatnonzero(bins == k)
        if rows.size < args.samples_per_phase:
            print(f"[WARN] phase bin {k} has only {rows.size} transitions (< {args.samples_per_phase}); using all")
            picks.append(rows)
        else:
            picks.append(rng.choice(rows, size=args.samples_per_phase, replace=False))
    idx = np.sort(np.concatenate(picks))
    obs, act, action_std = read_rows_and_action_std(h5, idx)

    with np.errstate(divide="ignore", invalid="ignore"):
        det_k = np.bincount(bins, weights=det, minlength=nb)
        fail_h = np.bincount(bins, weights=z & det, minlength=nb) / det_k
        fail_ep = np.bincount(bins, weights=end_bad, minlength=nb) / count_ds
    meta = {
        **request,
        "dataset_path": str(h5),
        "sidecar_path": str(sidecar),
        "sidecar_gamma": side_gamma,
        "sidecar_num_phase_bins": side_nb,
        "phase_semantics": data["_phase_semantics"],
        "episodes_episode_id": n_ep,
        "episodes_aw_bounds": n_ep_aw,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest = {
        "transition_index": idx.astype(np.int64),
        "episode_id": data["episode_id"][idx].astype(np.int64),
        "phase": phase[idx].astype(np.float32),
        "phase_bin": bins[idx].astype(np.int16),
        "advantage_H": advantage[idx].astype(np.float32),
        "utility_group": group[idx].astype(np.int8),
        "utility_group_global": group_global[idx].astype(np.int8),
        "future_bad_tracking": np.where(det[idx], z[idx], -1).astype(np.int8),
        "episode_bad_tracking": end_bad[idx].astype(np.int8),
        "observations": obs,
        "dataset_actions": act,
        "action_std": action_std.astype(np.float64),
        "dataset_bin_count": count_ds.astype(np.int64),
        "dataset_future_bad_tracking_rate": fail_h,
        "dataset_episode_bad_tracking_rate": fail_ep,
        "dataset_undeterminable_frac": 1.0 - det_k / np.maximum(count_ds, 1),
        "utility_low_threshold": lo_thr,
        "utility_high_threshold": hi_thr,
        "utility_global_thresholds": np.array([g_lo, g_hi]),
        "meta": np.array(json.dumps(meta)),
    }
    path = Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **manifest)
    print(f"[manifest] saved {path} ({idx.size} transitions, {time.time() - t0:.0f}s)")
    return manifest


def load_or_build_manifest(args, h5: Path, sidecar: Path) -> dict:
    request = manifest_request(args, h5, sidecar)
    path = Path(args.manifest)
    if not path.exists():
        return build_manifest(args, h5, sidecar, request)
    with np.load(path) as m:
        manifest = {k: m[k] for k in m.files}
    meta = json.loads(str(manifest["meta"]))
    mismatch = {k: (meta.get(k), v) for k, v in request.items() if meta.get(k) != v}
    if mismatch:
        lines = "\n".join(f"  {k}: manifest={a!r} requested={b!r}" for k, (a, b) in mismatch.items())
        raise SystemExit(
            f"existing manifest {path} was built with different settings; it is never resampled.\n{lines}\n"
            "Pass a different --manifest path to build a separate manifest."
        )
    print(f"[manifest] reusing {path} (built {meta['created']}, {manifest['transition_index'].size} transitions)")
    return manifest


# -------------------------------------------------------------------------------------- policies
def ckpt_step(path: Path) -> int | None:
    m = CKPT_STEP_RE.search(path.name)
    return int(m.group(1)) if m else None


_TB_CACHE: dict[str, dict[int, float]] = {}


def tb_success(run_dir: Path) -> dict[int, float]:
    """Eval success (%) by step from the run's TensorBoard log (cached per run directory)."""
    key = str(run_dir.resolve())
    if key not in _TB_CACHE:
        _TB_CACHE[key] = _read_tb_success(run_dir)
    return _TB_CACHE[key]


def _read_tb_success(run_dir: Path) -> dict[int, float]:
    """Success at every eval step. Eval steps come from ``Eval/num_episodes`` (always written); the
    success scalar is absent at steps where no episode succeeded, so missing values are 0 (the same
    convention as ``summarize_validation_results`` and ``phase_failure_alignment``)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: PLC0415

    events = sorted(run_dir.glob("events.out.tfevents*"))
    if not events:
        return {}
    ea = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    if PFA.EPISODES_TAG not in tags:
        return {}
    success = {e.step: e.value for e in ea.Scalars(PFA.SUCCESS_TAG)} if PFA.SUCCESS_TAG in tags else {}
    return {e.step: float(success.get(e.step, 0.0)) for e in ea.Scalars(PFA.EPISODES_TAG)}


@torch.no_grad()
def policy_actions(ckpt_path: Path, obs: np.ndarray, dataset_name: str, device: torch.device, batch: int):
    """Deterministic env-space actions of a checkpoint on raw observations (inference semantics)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck["args"]
    problems = []
    if ck.get("action_space_mode") != EXPECTED_ACTION_SPACE:
        problems.append(f"action_space_mode={ck.get('action_space_mode')!r} (expected {EXPECTED_ACTION_SPACE})")
    if not args.get("use_tanh", False):
        problems.append("use_tanh=False")
    if args.get("use_cnn_encoder", False):
        problems.append("use_cnn_encoder=True is not supported")
    if Path(args["offline_dataset_path"]).name != dataset_name:
        problems.append(f"trained on {args['offline_dataset_path']}, analysing {dataset_name}")
    in_dim = int(ck["actor_state_dict"]["net.0.weight"].shape[1])
    if in_dim != obs.shape[1]:
        problems.append(f"actor input dim {in_dim} != observation dim {obs.shape[1]}")
    if problems:
        raise SystemExit(f"{ckpt_path}: " + "; ".join(problems))

    models = CheckpointModels(ck, device)
    actor, norm = models.actor, models.obs_normalizer
    normalized = not isinstance(norm, torch.nn.Identity)
    if bool(args.get("obs_normalization", False)) != normalized:
        raise SystemExit(
            f"{ckpt_path}: obs_normalization={args.get('obs_normalization')} but normaliser state "
            f"{'present' if normalized else 'missing'}"
        )
    actor.eval()
    out = np.empty((obs.shape[0], int(actor.action_scale.numel())), np.float32)
    for lo in range(0, obs.shape[0], batch):
        x = torch.as_tensor(obs[lo : lo + batch], device=device)
        out[lo : lo + batch] = actor(_normalize(norm, x))[0].float().cpu().numpy()
    info = {
        "path": str(ckpt_path),
        "global_step": int(ck.get("global_step", -1)),
        "action_space_mode": ck.get("action_space_mode"),
        "obs_normalization": normalized,
        "normalizer_count": float(norm.count) if normalized else None,
        "action_scale_min": float(actor.action_scale.min()),
        "action_scale_max": float(actor.action_scale.max()),
    }
    del models, ck
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out, info


def distances(a_pi: np.ndarray, a_d: np.ndarray, sigma: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    diff = a_pi.astype(np.float64) - a_d.astype(np.float64)
    norm = np.sqrt(np.mean(np.square(diff / (sigma + eps)), axis=1))
    raw = np.sqrt(np.mean(np.square(diff), axis=1))
    return norm, raw


# --------------------------------------------------------------------------------------- metrics
def region_mask(k, bins: np.ndarray, regions: dict[str, np.ndarray] | None) -> np.ndarray:
    """Transitions of phase bin ``k``, of all bins (``"all"``) or of a pooled region (e.g. ``"failure"``)."""
    if k == "all":
        return np.ones(bins.size, bool)
    if isinstance(k, str):
        return regions[k]
    return bins == k


def seed_phase_metrics(
    dist: dict[str, np.ndarray], bins: np.ndarray, group: np.ndarray, nb: int, regions: dict | None = None
) -> list[dict]:
    """Per phase bin (and pooled 'all'): D_P, D_F, dD for all/high/low, G_P, G_F, dG (both distance kinds)."""
    rows = []
    for k in [*range(nb), "all", *(regions or {})]:
        in_k = region_mask(k, bins, regions)
        row: dict = {"phase_bin": k}
        for kind in DIST_KINDS:
            d_p, d_f = dist[f"peak{kind}"], dist[f"final{kind}"]
            for gname, gval in GROUPS:
                m = in_k if gval is None else in_k & (group == gval)
                sfx = "" if gname == "all" else f"_{gname}"
                if kind == "":
                    row[f"n{sfx or '_all'}"] = int(m.sum())
                row[f"D_P{sfx}{kind}"] = float(d_p[m].mean())
                row[f"D_F{sfx}{kind}"] = float(d_f[m].mean())
                row[f"dD{sfx}{kind}"] = row[f"D_F{sfx}{kind}"] - row[f"D_P{sfx}{kind}"]
                row[f"D_PF{sfx}{kind}"] = float(dist[f"peak_final{kind}"][m].mean())
            row[f"G_P{kind}"] = row[f"D_P_low{kind}"] - row[f"D_P_high{kind}"]
            row[f"G_F{kind}"] = row[f"D_F_low{kind}"] - row[f"D_F_high{kind}"]
            row[f"dG{kind}"] = row[f"G_F{kind}"] - row[f"G_P{kind}"]
        rows.append(row)
    return rows


def episode_bootstrap(delta, episodes, group, n_boot: int, rng, block: int = 100) -> dict[str, tuple[float, float]]:
    """Episode-cluster bootstrap 95% CIs for mean(delta) of all/high/low and for dG = mean_low - mean_high."""
    uniq, inv = np.unique(episodes, return_inverse=True)
    u = uniq.size

    def sums(mask):
        return (
            np.bincount(inv[mask], weights=delta[mask], minlength=u),
            np.bincount(inv[mask], minlength=u).astype(np.float64),
        )

    parts = {"all": sums(np.ones(delta.size, bool)), "high": sums(group == HIGH), "low": sums(group == LOW)}
    reps = {name: [] for name in (*parts, "gap")}
    for start in range(0, n_boot, block):
        draw = rng.integers(0, u, size=(min(block, n_boot - start), u))
        means = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            for name, (s, c) in parts.items():
                means[name] = s[draw].sum(1) / c[draw].sum(1)
                reps[name].append(means[name])
        reps["gap"].append(means["low"] - means["high"])
    out = {}
    for name, chunks in reps.items():
        values = np.concatenate(chunks)
        out[name] = tuple(np.nanpercentile(values, [2.5, 97.5])) if np.isfinite(values).any() else (math.nan,) * 2
    return out


def paired_transition_stats(dist, bins, group, episodes, nb, n_boot, rng, regions: dict | None = None) -> list[dict]:
    """Same-transition Peak->Final statistics per phase bin (and pooled), one seed."""
    delta = dist["final"] - dist["peak"]
    rows = []
    for k in [*range(nb), "all", *(regions or {})]:
        in_k = region_mask(k, bins, regions)
        ci = episode_bootstrap(delta[in_k], episodes[in_k], group[in_k], n_boot, rng)
        for gname, gval in GROUPS:
            m = in_k if gval is None else in_k & (group == gval)
            dm = delta[m]
            rows.append(
                {
                    "phase_bin": k,
                    "group": gname,
                    "n": int(m.sum()),
                    "n_episodes": int(np.unique(episodes[m]).size),
                    "mean_delta": float(dm.mean()),
                    "median_delta": float(np.median(dm)),
                    "frac_final_closer": float((dist["final"][m] < dist["peak"][m]).mean()),
                    "boot_ci_lo": ci[gname][0],
                    "boot_ci_hi": ci[gname][1],
                }
            )
        lo_m, hi_m = in_k & (group == LOW), in_k & (group == HIGH)
        rows.append(
            {
                "phase_bin": k,
                "group": "gap(dG)",
                "n": int(lo_m.sum() + hi_m.sum()),
                "n_episodes": int(np.unique(episodes[lo_m | hi_m]).size),
                "mean_delta": float(delta[lo_m].mean() - delta[hi_m].mean()),
                "median_delta": math.nan,
                "frac_final_closer": math.nan,
                "boot_ci_lo": ci["gap"][0],
                "boot_ci_hi": ci["gap"][1],
            }
        )
    return rows


def seed_stats(values: list[float]) -> dict[str, float]:
    from scipy import stats  # noqa: PLC0415

    v = np.asarray(values, np.float64)
    n = v.size
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n)) if n > 1 else math.nan
    return {"mean": mean, "std": std, "ci_lo": mean - half, "ci_hi": mean + half, "n_seeds": n}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats  # noqa: PLC0415

    ok = np.isfinite(x) & np.isfinite(y)
    return float(stats.spearmanr(x[ok], y[ok]).statistic) if ok.sum() > 2 else math.nan


# ----------------------------------------------------------------------------------- seed config
def write_seed_config(args) -> None:
    import yaml  # noqa: PLC0415

    runs = PFA.discover_runs(Path(args.log_root), [args.method], args.final_step).get(args.method, [])
    entries = []
    for run_dir in runs:
        seed = int(PFA.RUN_RE.match(run_dir.name)["seed"])
        if args.seeds and seed not in args.seeds:
            continue
        success = tb_success(run_dir)
        if not success:
            raise SystemExit(f"{run_dir}: no {PFA.EPISODES_TAG} eval scalars")
        steps = sorted(success)
        peak, final = PFA.peak_and_final_steps(steps, success)
        paths = {name: run_dir / f"model_{step:07d}.pt" for name, step in (("peak", peak), ("final", final))}
        for name, p in paths.items():
            if not p.exists():
                raise SystemExit(f"{name} checkpoint {p} does not exist (eval step without a saved checkpoint)")
        entries.append(
            {
                "seed": seed,
                "peak": str(paths["peak"]),
                "final": str(paths["final"]),
                "peak_step": peak,
                "final_step": final,
                "peak_success": round(float(success[peak]), 3),
                "final_success": round(float(success[final]), 3),
            }
        )
    if not entries:
        raise SystemExit(f"no finished runs of {args.method} under {args.log_root}")
    out = Path(args.write_seed_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# generated by analyze_cql_peak_final_alignment.py from {args.log_root}\n"
        "# peak = eval step with max Eval/stop_reason_percent/motion_ends (earliest on ties); final = last\n"
    )
    out.write_text(header + yaml.safe_dump({"method": args.method, "seeds": entries}, sort_keys=False))
    for e in entries:
        print(
            f"seed {e['seed']}: peak {e['peak_step']} ({e['peak_success']}%)  final {e['final_step']} "
            f"({e['final_success']}%)"
        )
    print(f"written: {out}")


def read_seed_config(path: Path) -> list[dict]:
    import yaml  # noqa: PLC0415

    cfg = yaml.safe_load(path.read_text())
    entries = cfg["seeds"] if isinstance(cfg, dict) else cfg
    seen = set()
    for e in entries:
        for key in ("seed", "peak", "final"):
            if key not in e:
                raise SystemExit(f"{path}: entry {e} lacks '{key}'")
        if e["seed"] in seen:
            raise SystemExit(f"{path}: seed {e['seed']} listed twice")
        seen.add(e["seed"])
        for key in ("peak", "final"):
            if not Path(e[key]).exists():
                raise SystemExit(f"{path}: seed {e['seed']} {key} checkpoint {e[key]} does not exist")
    return entries


# ---------------------------------------------------------------------------------------- output
# Named per-seed metrics: (output name, source, key). Sources: "m" = seed_phase_metrics row,
# "p_<group>" = paired_transition_stats row of that group. Every delta is a within-seed Final - Peak.
NAMED_METRICS = (
    ("peak_distance_mean", "m", "D_P"),
    ("final_distance_mean", "m", "D_F"),
    ("delta_distance", "m", "dD"),
    ("median_delta_distance", "p_all", "median_delta"),
    ("fraction_final_closer_all", "p_all", "frac_final_closer"),
    ("peak_high_distance", "m", "D_P_high"),
    ("final_high_distance", "m", "D_F_high"),
    ("delta_high_distance", "m", "dD_high"),
    ("fraction_final_closer_high", "p_high", "frac_final_closer"),
    ("peak_low_distance", "m", "D_P_low"),
    ("final_low_distance", "m", "D_F_low"),
    ("delta_low_distance", "m", "dD_low"),
    ("fraction_final_closer_low", "p_low", "frac_final_closer"),
    ("peak_preference_gap", "m", "G_P"),
    ("final_preference_gap", "m", "G_F"),
    ("delta_preference_gap", "m", "dG"),
    ("peak_final_action_distance", "m", "D_PF"),
    ("peak_distance_raw", "m", "D_P_raw"),
    ("final_distance_raw", "m", "D_F_raw"),
    ("delta_distance_raw", "m", "dD_raw"),
    ("delta_high_distance_raw", "m", "dD_high_raw"),
    ("delta_low_distance_raw", "m", "dD_low_raw"),
    ("delta_preference_gap_raw", "m", "dG_raw"),
)
METRIC_NAMES = tuple(name for name, _, _ in NAMED_METRICS)
PHASE_METRICS = (
    "peak_distance_mean",
    "final_distance_mean",
    "delta_distance",
    "median_delta_distance",
    "fraction_final_closer_all",
    "peak_final_action_distance",
    "peak_distance_raw",
    "final_distance_raw",
    "delta_distance_raw",
)
UTILITY_METRICS = (
    "peak_high_distance",
    "final_high_distance",
    "delta_high_distance",
    "fraction_final_closer_high",
    "peak_low_distance",
    "final_low_distance",
    "delta_low_distance",
    "fraction_final_closer_low",
    "peak_preference_gap",
    "final_preference_gap",
    "delta_preference_gap",
    "delta_high_distance_raw",
    "delta_low_distance_raw",
    "delta_preference_gap_raw",
)
CONTRAST_METRICS = (
    "delta_distance",
    "delta_high_distance",
    "delta_low_distance",
    "delta_preference_gap",
    "peak_preference_gap",
    "final_preference_gap",
    "peak_final_action_distance",
    "delta_distance_raw",
    "delta_preference_gap_raw",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fmt(v: float, digits: int = 3, signed: bool = True) -> str:
    if v is None or not np.isfinite(v):
        return "-"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def named_seed_metrics(metric_row: dict, paired: dict[str, dict]) -> dict[str, float]:
    src = {"m": metric_row, "p_all": paired["all"], "p_high": paired["high"], "p_low": paired["low"]}
    return {name: float(src[s][key]) for name, s, key in NAMED_METRICS}


# ----------------------------------------------------------------------------------------- plots
def _phase_axis(ax, nb: int, failure_bins: list[int]) -> None:
    """Identical phase axis on every plot so the panels can be placed side by side."""
    for k in failure_bins:
        ax.axvspan(k - 0.5, k + 0.5, color="#f0efec", lw=0, zorder=0)
    ax.set_xlim(-0.5, nb - 0.5)
    ax.set_xticks(np.arange(nb))
    ax.set_xlabel("motion phase bin")
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _band(ax, agg: dict, metric: str, nb: int, color: str, label: str) -> None:
    x = np.arange(nb)
    m = np.array([agg[metric][k]["mean"] for k in range(nb)])
    lo = np.array([agg[metric][k]["ci_lo"] for k in range(nb)])
    hi = np.array([agg[metric][k]["ci_hi"] for k in range(nb)])
    ax.plot(x, m, color=color, lw=2, marker="o", ms=4, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)


def _seed_lines(ax, series_by_seed: dict, nb: int) -> None:
    for i, series in enumerate(series_by_seed.values()):
        ax.plot(np.arange(nb), series, color=C_MUTED, lw=0.8, alpha=0.7, label="individual seeds" if i == 0 else None)


def make_plots(out: Path, nb: int, fail_rate, hazard, failure_bins, agg, per_seed_series) -> list[str]:
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    files: list[str] = []
    size = (8, 3.3)
    shade = "shaded = failure-prone bins"

    def save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        files.append(name)

    def title(ax, text: str) -> None:
        ax.set_title(text, loc="left", fontsize=10)

    # Plot 1: Peak vs Final distance
    fig, ax = plt.subplots(figsize=size)
    _phase_axis(ax, nb, failure_bins)
    _band(ax, agg, "peak_distance_mean", nb, C_BLUE, "Peak")
    _band(ax, agg, "final_distance_mean", nb, C_ORANGE, "Final")
    ax.set_ylabel("normalised distance d")
    title(ax, f"Same-state distance to the logged action (seed mean, 95% CI; {shade})")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    save(fig, "plot1_peak_final_distance.png")

    # Plot 2: Final - Peak
    fig, ax = plt.subplots(figsize=size)
    _phase_axis(ax, nb, failure_bins)
    _seed_lines(ax, per_seed_series["delta_distance"], nb)
    _band(ax, agg, "delta_distance", nb, C_BLUE, "seed mean (95% CI)")
    ax.axhline(0, color=C_MUTED, lw=1)
    ax.set_ylabel("d_Final - d_Peak")
    title(ax, "Peak -> Final change (negative = increased dataset-action alignment)")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "plot2_delta_distance.png")

    # Plot 3: high vs low utility, Peak and Final side by side
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.3), sharey=True)
    for ax, (which, name) in zip(axes, (("peak", "Peak"), ("final", "Final"))):
        _phase_axis(ax, nb, failure_bins)
        _band(ax, agg, f"{which}_high_distance", nb, C_ORANGE, "high utility")
        _band(ax, agg, f"{which}_low_distance", nb, C_AQUA, "low utility")
        title(ax, f"{name}: distance to high- vs low-utility logged actions")
        ax.legend(fontsize=8, frameon=False, ncol=2)
    axes[0].set_ylabel("normalised distance d")
    save(fig, "plot3_utility_alignment.png")

    # Plot 4: preference gap
    fig, ax = plt.subplots(figsize=size)
    _phase_axis(ax, nb, failure_bins)
    _band(ax, agg, "peak_preference_gap", nb, C_BLUE, "Peak")
    _band(ax, agg, "final_preference_gap", nb, C_ORANGE, "Final")
    ax.axhline(0, color=C_MUTED, lw=1)
    ax.set_ylabel("G = D_low - D_high")
    title(ax, "Utility preference gap (positive = closer to high-utility behaviour)")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    save(fig, "plot4_preference_gap.png")

    # Plot 5: failure association (same definitions as the existing hazard / eta_F^H analysis)
    fig, ax = plt.subplots(figsize=size)
    _phase_axis(ax, nb, failure_bins)
    x = np.arange(nb)
    hz = np.asarray(hazard, dtype=float)
    if np.isfinite(hz).any():
        ax.bar(x - 0.2, 100 * fail_rate, width=0.4, color=C_BLUE, label="F^H: bad tracking within H steps")
        ax.bar(x + 0.2, 100 * hz, width=0.4, color=C_ORANGE, label="hazard h(k): terminations / entries")
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.bar(x, 100 * fail_rate, width=0.8, color=C_BLUE)
    ax.set_ylabel("%")
    title(ax, "Dataset future bad-tracking association by phase")
    save(fig, "plot5_failure_association.png")

    # Combined overview: failure association, dD by utility, G, dG
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for ax in axes:
        _phase_axis(ax, nb, failure_bins)
    axes[0].bar(x, 100 * fail_rate, width=0.8, color=C_BLUE)
    axes[0].set_ylabel("F^H rate (%)")
    title(axes[0], f"Dataset: bad tracking within H steps ({shade})")
    for metric, color, label in (
        ("delta_distance", C_BLUE, "all"),
        ("delta_high_distance", C_ORANGE, "high utility"),
        ("delta_low_distance", C_AQUA, "low utility"),
    ):
        _band(axes[1], agg, metric, nb, color, label)
    axes[1].axhline(0, color=C_MUTED, lw=1)
    axes[1].set_ylabel("d_Final - d_Peak")
    title(axes[1], "Peak -> Final change of the same-state distance (<0: Final closer)")
    axes[1].legend(fontsize=8, frameon=False, ncol=3)
    _band(axes[2], agg, "peak_preference_gap", nb, C_BLUE, "Peak")
    _band(axes[2], agg, "final_preference_gap", nb, C_ORANGE, "Final")
    axes[2].axhline(0, color=C_MUTED, lw=1)
    axes[2].set_ylabel("G = D_low - D_high")
    title(axes[2], "Utility preference gap (>0: closer to high- than low-utility behaviour)")
    axes[2].legend(fontsize=8, frameon=False, ncol=2)
    _seed_lines(axes[3], per_seed_series["delta_preference_gap"], nb)
    _band(axes[3], agg, "delta_preference_gap", nb, C_BLUE, "seed mean (95% CI)")
    axes[3].axhline(0, color=C_MUTED, lw=1)
    axes[3].set_ylabel("G_Final - G_Peak")
    title(axes[3], "Peak -> Final change of the preference gap (<0: high-utility preference reduced)")
    axes[3].legend(fontsize=8, frameon=False)
    for ax in axes[:-1]:
        ax.set_xlabel("")
    save(fig, "alignment_by_phase.png")
    return files


# ------------------------------------------------------------------------------- console summary
def console_summary(seeds, named, agg, failure_bins, fail_rate, nb) -> str:
    bar = "=" * 50
    lines: list[str] = []

    def block(name: str) -> None:
        lines.extend([bar, name, bar, ""])

    def ms(metric: str, k="all", signed: bool = True) -> str:
        s = agg[metric][k]
        return f"{s['mean']:+.4f} ± {s['std']:.4f}" if signed else f"{s['mean']:.4f} ± {s['std']:.4f}"

    def neg(metric: str, k="all") -> str:
        return f"{sum(named[(s, k)][metric] < 0 for s in seeds)}/{len(seeds)} seeds < 0"

    block("GLOBAL PEAK -> FINAL ALIGNMENT")
    for s in seeds:
        r = named[(s, "all")]
        lines += [
            f"Seed {s}:",
            f"  Peak distance:  {r['peak_distance_mean']:.4f}",
            f"  Final distance: {r['final_distance_mean']:.4f}",
            f"  Delta:          {r['delta_distance']:+.4f}   "
            f"Final closer on {100 * r['fraction_final_closer_all']:.1f}% of transitions",
            "",
        ]
    a = agg["delta_distance"]["all"]
    lines += [
        "Across seeds:",
        f"  Mean delta: {a['mean']:+.4f}",
        f"  Std:        {a['std']:.4f}",
        f"  95% CI:     [{a['ci_lo']:+.4f}, {a['ci_hi']:+.4f}]   {neg('delta_distance')}",
        "",
    ]
    for name, p, f, d in (
        ("HIGH-UTILITY ALIGNMENT", "peak_high_distance", "final_high_distance", "delta_high_distance"),
        ("LOW-UTILITY ALIGNMENT", "peak_low_distance", "final_low_distance", "delta_low_distance"),
        ("UTILITY PREFERENCE GAP", "peak_preference_gap", "final_preference_gap", "delta_preference_gap"),
    ):
        block(name)
        signed = "gap" in p
        lines += [f"Peak:  {ms(p, signed=signed)}", f"Final: {ms(f, signed=signed)}", f"Delta: {ms(d)}   {neg(d)}", ""]
    block("FAILURE-PRONE PHASES")
    for k in failure_bins:
        lines += [
            f"phase: bin {k} [{k / nb:.2f}, {(k + 1) / nb:.2f})",
            f"  future failure rate:        {100 * fail_rate[k]:.1f}%",
            f"  delta dataset distance:     {ms('delta_distance', k)}   {neg('delta_distance', k)}",
            f"  delta low-utility distance: {ms('delta_low_distance', k)}   {neg('delta_low_distance', k)}",
            f"  delta preference gap:       {ms('delta_preference_gap', k)}   {neg('delta_preference_gap', k)}",
            "",
        ]
    block(f"FAILURE-PRONE PHASES POOLED (bins {' '.join(map(str, failure_bins))}) VS OTHER PHASES")
    for name, p, f, d in (
        ("Dataset-action distance", "peak_distance_mean", "final_distance_mean", "delta_distance"),
        ("High-utility distance", "peak_high_distance", "final_high_distance", "delta_high_distance"),
        ("Low-utility distance", "peak_low_distance", "final_low_distance", "delta_low_distance"),
        ("Preference gap", "peak_preference_gap", "final_preference_gap", "delta_preference_gap"),
    ):
        signed = "gap" in p
        diff = seed_stats([named[(s, "failure")][d] - named[(s, "other")][d] for s in seeds])
        n_neg = sum(named[(s, "failure")][d] - named[(s, "other")][d] < 0 for s in seeds)
        lines += [
            f"{name}:",
            f"  failure  Peak {ms(p, 'failure', signed)}  Final {ms(f, 'failure', signed)}  "
            f"Delta {ms(d, 'failure')}  {neg(d, 'failure')}",
            f"  other    Peak {ms(p, 'other', signed)}  Final {ms(f, 'other', signed)}  "
            f"Delta {ms(d, 'other')}  {neg(d, 'other')}",
            f"  failure - other delta: {diff['mean']:+.4f} ± {diff['std']:.4f}  "
            f"95% CI [{diff['ci_lo']:+.4f}, {diff['ci_hi']:+.4f}]  {n_neg}/{len(seeds)} seeds < 0",
            "",
        ]
    lines.append(
        "Values: seed mean ± std of within-seed Final - Peak differences. Descriptive statistics only; "
        "no causal interpretation is implied."
    )
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5")
    ap.add_argument("--seed-config", default="analysis/cql_peak_final_checkpoints.yaml")
    ap.add_argument("--aw-sidecar", default=None, help="default: <dataset>.aw_weights.H<horizon>.npz")
    ap.add_argument("--num-phase-bins", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--samples-per-phase", type=int, default=5000)
    ap.add_argument("--eval-sampling-seed", type=int, default=1234)
    ap.add_argument("--high-quantile", type=float, default=0.30, help="top fraction of A^H per phase = high utility")
    ap.add_argument("--low-quantile", type=float, default=0.30, help="bottom fraction of A^H per phase = low utility")
    ap.add_argument("--manifest", default="analysis/fixed_alignment_eval_manifest.npz")
    ap.add_argument("--output-dir", default="analysis/cql_peak_final_alignment")
    ap.add_argument(
        "--failure-bins",
        nargs="*",
        type=int,
        default=None,
        help="failure-prone phase bins; default: top --failure-top-frac bins by dataset F^H rate",
    )
    ap.add_argument("--failure-top-frac", type=float, default=0.25)
    ap.add_argument("--hazard-csv", default=None, help="optional hazard.csv (phase_failure_alignment.py) to merge")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--action-std-eps", type=float, default=1e-6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=8192)
    # seed-config generation
    ap.add_argument("--write-seed-config", default=None, help="write the checkpoint YAML from TensorBoard and exit")
    ap.add_argument("--method", default="g1_29dof_wbt_cql", help="run-name prefix for --write-seed-config")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--log-root", default="logs/WholeBodyTracking")
    ap.add_argument("--final-step", type=int, default=100000)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.write_seed_config:
        write_seed_config(args)
        return

    h5 = Path(args.dataset)
    sidecar = sidecar_for(h5, args.horizon, args.aw_sidecar)
    manifest = load_or_build_manifest(args, h5, sidecar)
    meta = json.loads(str(manifest["meta"]))
    nb = int(meta["num_phase_bins"])
    bins = manifest["phase_bin"].astype(np.int64)
    group = manifest["utility_group"].astype(np.int64)
    group_global = manifest["utility_group_global"].astype(np.int64)
    episodes = manifest["episode_id"]
    fbt = manifest["future_bad_tracking"]
    obs, a_data, sigma = manifest["observations"], manifest["dataset_actions"], manifest["action_std"]
    fail_rate = manifest["dataset_future_bad_tracking_rate"]
    counts = np.bincount(bins, minlength=nb)
    n_high = np.bincount(bins, weights=group == HIGH, minlength=nb).astype(int)
    n_low = np.bincount(bins, weights=group == LOW, minlength=nb).astype(int)
    short = [k for k in range(nb) if counts[k] < int(meta["samples_per_phase"])]
    if short:
        print(f"[WARN] phase bins with fewer than {meta['samples_per_phase']} samples: {short}")

    if args.failure_bins:
        failure_bins = sorted(args.failure_bins)
    else:
        top = max(1, round(args.failure_top_frac * nb))
        failure_bins = sorted(np.argsort(-np.nan_to_num(fail_rate, nan=-1.0))[:top].tolist())
    other_bins = [k for k in range(nb) if k not in failure_bins]
    # pooled regions: every sampled transition of the failure-prone bins vs of all other bins
    regions = {"failure": np.isin(bins, failure_bins), "other": ~np.isin(bins, failure_bins)}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = read_seed_config(Path(args.seed_config))
    device = torch.device(args.device)
    rng = np.random.default_rng(args.bootstrap_seed)

    checkpoints, per_seed_rows, global_rows, paired_rows = [], [], [], []
    policy_abs = {}
    for e in sorted(entries, key=lambda e: e["seed"]):
        seed = int(e["seed"])
        dist, acts, seed_ckpts = {}, {}, {}
        for which in ("peak", "final"):
            path = Path(e[which])
            a_pi, info = policy_actions(path, obs, h5.name, device, args.batch_size)
            acts[which] = a_pi
            dist[which], dist[f"{which}_raw"] = distances(a_pi, a_data, sigma, args.action_std_eps)
            policy_abs[(seed, which)] = float(np.abs(a_pi).mean())
            step = ckpt_step(path)
            success = tb_success(path.parent)
            info.update(
                {
                    "seed": seed,
                    "which": which,
                    "step": step,
                    "success": success.get(step, math.nan) if step is not None else math.nan,
                }
            )
            if which == "peak" and success:
                expected, _ = PFA.peak_and_final_steps(sorted(success), success)
                info["tb_peak_step"] = expected
                if step != expected:
                    print(f"[WARN] seed {seed}: configured peak step {step} != TB max-success step {expected}")
            checkpoints.append(info)
            seed_ckpts[which] = info
            print(
                f"seed {seed} {which:5s} step {step}: success {info['success']:.1f}%  "
                f"mean d={dist[which].mean():.3f} (raw {dist[f'{which}_raw'].mean():.4f})"
            )
        dist["peak_final"], dist["peak_final_raw"] = distances(acts["final"], acts["peak"], sigma, args.action_std_eps)
        per_seed_rows.extend({"seed": seed, **row} for row in seed_phase_metrics(dist, bins, group, nb, regions))
        global_rows.extend({"seed": seed, **row} for row in seed_phase_metrics(dist, bins, group_global, nb, regions))
        paired_rows.extend(
            {"seed": seed, **row}
            for row in paired_transition_stats(dist, bins, group, episodes, nb, args.bootstrap, rng, regions)
        )
        np.savez_compressed(
            out / f"sample_level_alignment_seed{seed}.npz",
            transition_index=manifest["transition_index"],
            episode_id=episodes,
            phase=manifest["phase"],
            phase_bin=manifest["phase_bin"],
            utility_group=manifest["utility_group"],
            future_bad_tracking=fbt,
            advantage_H=manifest["advantage_H"],
            d_peak=dist["peak"].astype(np.float32),
            d_final=dist["final"].astype(np.float32),
            delta_d=(dist["final"] - dist["peak"]).astype(np.float32),
            d_peak_raw=dist["peak_raw"].astype(np.float32),
            d_final_raw=dist["final_raw"].astype(np.float32),
            delta_d_raw=(dist["final_raw"] - dist["peak_raw"]).astype(np.float32),
            d_peak_final=dist["peak_final"].astype(np.float32),
            action_peak=acts["peak"],
            action_final=acts["final"],
            meta=np.array(json.dumps({"seed": seed, "manifest": str(args.manifest), **seed_ckpts}, default=float)),
        )

    seeds = sorted(int(e["seed"]) for e in entries)
    paired_idx: dict[tuple, dict[str, dict]] = {}
    for r in paired_rows:
        key = "gap" if r["group"].startswith("gap") else r["group"]
        paired_idx.setdefault((r["seed"], r["phase_bin"]), {})[key] = r
    named = {
        (r["seed"], r["phase_bin"]): named_seed_metrics(r, paired_idx[(r["seed"], r["phase_bin"])])
        for r in per_seed_rows
    }
    keys = [*range(nb), "all", "failure", "other"]
    agg = {m: {k: seed_stats([named[(s, k)][m] for s in seeds]) for k in keys} for m in METRIC_NAMES}
    per_seed_series = {m: {s: np.array([named[(s, k)][m] for k in range(nb)]) for s in seeds} for m in METRIC_NAMES}

    hazard = DFA.read_hazard_csv(Path(args.hazard_csv)) if args.hazard_csv else [math.nan] * nb

    def bin_info(k) -> dict:
        if k in regions:
            member = failure_bins if k == "failure" else other_bins
            return {
                "phase_bin": k,
                "phase_start": math.nan,
                "phase_end": math.nan,
                "n_eval": int(counts[member].sum()),
                "n_high": int(n_high[member].sum()),
                "n_low": int(n_low[member].sum()),
                "future_bad_tracking_rate": float(np.nanmean(fail_rate[member])),
            }
        if k == "all":
            return {
                "phase_bin": "all",
                "phase_start": 0.0,
                "phase_end": 1.0,
                "n_eval": int(counts.sum()),
                "n_high": int(n_high.sum()),
                "n_low": int(n_low.sum()),
                "future_bad_tracking_rate": float(np.nanmean(fail_rate)),
            }
        return {
            "phase_bin": k,
            "phase_start": k / nb,
            "phase_end": (k + 1) / nb,
            "n_eval": int(counts[k]),
            "n_high": int(n_high[k]),
            "n_low": int(n_low[k]),
            "future_bad_tracking_rate": float(fail_rate[k]),
        }

    # ---- per-seed tables ----
    phase_per_seed, utility_per_seed = [], []
    for s in seeds:
        for k in keys:
            info, nm, pr = bin_info(k), named[(s, k)], paired_idx[(s, k)]
            base = {"seed": s, **{c: info[c] for c in ("phase_bin", "phase_start", "phase_end")}}
            phase_per_seed.append(
                {
                    **base,
                    "n_eval": info["n_eval"],
                    "n_episodes": pr["all"]["n_episodes"],
                    "future_bad_tracking_rate": info["future_bad_tracking_rate"],
                    **{m: nm[m] for m in PHASE_METRICS},
                    "delta_boot_ci_lo": pr["all"]["boot_ci_lo"],
                    "delta_boot_ci_hi": pr["all"]["boot_ci_hi"],
                }
            )
            utility_per_seed.append(
                {
                    **base,
                    "n_high": info["n_high"],
                    "n_low": info["n_low"],
                    "future_bad_tracking_rate": info["future_bad_tracking_rate"],
                    **{m: nm[m] for m in UTILITY_METRICS},
                    "delta_high_boot_ci_lo": pr["high"]["boot_ci_lo"],
                    "delta_high_boot_ci_hi": pr["high"]["boot_ci_hi"],
                    "delta_low_boot_ci_lo": pr["low"]["boot_ci_lo"],
                    "delta_low_boot_ci_hi": pr["low"]["boot_ci_hi"],
                    "delta_gap_boot_ci_lo": pr["gap"]["boot_ci_lo"],
                    "delta_gap_boot_ci_hi": pr["gap"]["boot_ci_hi"],
                }
            )
    write_csv(out / "phase_summary_per_seed.csv", phase_per_seed)
    write_csv(out / "utility_alignment_per_seed.csv", utility_per_seed)
    write_csv(out / "utility_alignment_global_quantile_per_seed.csv", global_rows)

    # ---- aggregated tables (seed mean of within-seed differences) ----
    phase_agg, utility_agg = [], []
    for k in keys:
        info = bin_info(k)
        sample_rate = float(np.mean(fbt[(fbt >= 0) & region_mask(k, bins, regions)]))
        is_failure = k == "failure" or (isinstance(k, int) and k in failure_bins)
        extra = {"failure_prone": int(is_failure), "hazard": hazard[k] if isinstance(k, int) else math.nan}
        row = {**info, **extra, "sample_future_bad_tracking_rate": sample_rate}
        for m in PHASE_METRICS:
            st = agg[m][k]
            row.update(
                {f"{m}_mean": st["mean"], f"{m}_std": st["std"], f"{m}_ci_lo": st["ci_lo"], f"{m}_ci_hi": st["ci_hi"]}
            )
            row.update({f"{m}_seed{s}": named[(s, k)][m] for s in seeds})
        phase_agg.append(row)

        a = {m: agg[m][k] for m in METRIC_NAMES}
        urow = {
            **{c: info[c] for c in ("phase_bin", "phase_start", "phase_end", "n_eval", "n_high", "n_low")},
            "future_bad_tracking_rate": info["future_bad_tracking_rate"],
            "peak_distance_mean": a["peak_distance_mean"]["mean"],
            "final_distance_mean": a["final_distance_mean"]["mean"],
            "delta_distance_mean": a["delta_distance"]["mean"],
            "delta_distance_std": a["delta_distance"]["std"],
            "peak_high_distance": a["peak_high_distance"]["mean"],
            "final_high_distance": a["final_high_distance"]["mean"],
            "delta_high_distance": a["delta_high_distance"]["mean"],
            "peak_low_distance": a["peak_low_distance"]["mean"],
            "final_low_distance": a["final_low_distance"]["mean"],
            "delta_low_distance": a["delta_low_distance"]["mean"],
            "peak_preference_gap": a["peak_preference_gap"]["mean"],
            "final_preference_gap": a["final_preference_gap"]["mean"],
            "delta_preference_gap": a["delta_preference_gap"]["mean"],
            "fraction_final_closer_all": a["fraction_final_closer_all"]["mean"],
            "fraction_final_closer_high": a["fraction_final_closer_high"]["mean"],
            "fraction_final_closer_low": a["fraction_final_closer_low"]["mean"],
            **extra,
        }
        for m in ("delta_distance", "delta_high_distance", "delta_low_distance", "delta_preference_gap"):
            urow.update(
                {
                    f"{m}_ci_lo": a[m]["ci_lo"],
                    f"{m}_ci_hi": a[m]["ci_hi"],
                    f"{m}_seeds_negative": sum(named[(s, k)][m] < 0 for s in seeds),
                }
            )
            if m != "delta_distance":
                urow[f"{m}_std"] = a[m]["std"]
        urow["n_seeds"] = len(seeds)
        utility_agg.append(urow)
    write_csv(out / "phase_summary_aggregated.csv", phase_agg)
    write_csv(out / "utility_alignment_aggregated.csv", utility_agg)

    # ---- failure-prone vs other phases, and cross-phase Spearman (20 bins: descriptive only) ----
    contrast_rows = []
    for m in CONTRAST_METRICS:
        fail_vals = [float(np.mean(per_seed_series[m][s][failure_bins])) for s in seeds]
        other_vals = [float(np.mean(per_seed_series[m][s][other_bins])) for s in seeds]
        diff = [a - b for a, b in zip(fail_vals, other_vals)]
        rho_seed = [spearman(fail_rate, per_seed_series[m][s]) for s in seeds]
        fs, os_, ds = seed_stats(fail_vals), seed_stats(other_vals), seed_stats(diff)
        contrast_rows.append(
            {
                "metric": m,
                "failure_bins": " ".join(map(str, failure_bins)),
                "failure_mean": fs["mean"],
                "failure_std": fs["std"],
                "other_mean": os_["mean"],
                "other_std": os_["std"],
                "failure_minus_other_mean": ds["mean"],
                "failure_minus_other_ci_lo": ds["ci_lo"],
                "failure_minus_other_ci_hi": ds["ci_hi"],
                "seeds_failure_lt_other": int(sum(d < 0 for d in diff)),
                "n_seeds": len(seeds),
                "spearman_failrate_vs_seedmean": spearman(fail_rate, np.array([agg[m][k]["mean"] for k in range(nb)])),
                "spearman_failrate_per_seed_mean": float(np.nanmean(rho_seed)),
                "spearman_failrate_per_seed_std": float(np.nanstd(rho_seed, ddof=1)) if len(seeds) > 1 else 0.0,
            }
        )
    write_csv(out / "failure_phase_contrast.csv", contrast_rows)

    checks = {
        "action_space_mode": sorted({c["action_space_mode"] for c in checkpoints}),
        "obs_normalization_applied": all(c["obs_normalization"] for c in checkpoints),
        "dataset_actions_within_actor_range": float(
            np.mean(np.abs(a_data) <= checkpoints[0]["action_scale_max"] + 1e-6)
        ),
        "dataset_abs_action_mean": float(np.abs(a_data).mean()),
        "policy_abs_action_mean": {f"seed{s}_{w}": v for (s, w), v in policy_abs.items()},
        "samples_per_bin": counts.tolist(),
        "n_high_per_bin": n_high.tolist(),
        "n_low_per_bin": n_low.tolist(),
        "manifest": str(args.manifest),
    }
    run_info = {
        "args": vars(args),
        "manifest_meta": meta,
        "failure_bins": failure_bins,
        "checkpoints": checkpoints,
        "checks": checks,
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2, default=float))
    plot_files = make_plots(out, nb, fail_rate, hazard, failure_bins, agg, per_seed_series)
    console = console_summary(seeds, named, agg, failure_bins, fail_rate, nb)

    # ---- markdown report ----
    by_ck = {(c["seed"], c["which"]): c for c in checkpoints}
    lines = [
        "# CQL Peak vs Final: same-state alignment with logged behaviour",
        "",
        f"Dataset `{h5.name}`, manifest `{args.manifest}` ({int(counts.sum())} fixed transitions, "
        f"{nb} phase bins x {meta['samples_per_phase']}, sampling seed {meta['eval_sampling_seed']}). "
        f"A^H from `{meta['sidecar_basename']}` (H={meta['horizon']}, gamma={meta['sidecar_gamma']}). "
        f"Utility groups: top {meta['high_quantile']:.0%} / bottom {meta['low_quantile']:.0%} "
        "of A^H within each phase.",
        "",
        "| seed | peak step | peak success % | final step | final success % |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in seeds:
        p, f = by_ck[(s, "peak")], by_ck[(s, "final")]
        lines.append(
            f"| {s} | {p['step']} | {fmt(p['success'], 1, False)} | {f['step']} | {fmt(f['success'], 1, False)} |"
        )
    lines += [
        "",
        "Signs: a negative distance change means increased dataset-action alignment from Peak to Final; "
        "G > 0 means the policy is closer to high- than to low-utility logged actions; a negative G change means "
        "reduced relative preference for high-utility behaviour. Differences are formed within each seed, then "
        "averaged (mean ± std over seeds). Descriptive only.",
        "",
        f"## Per phase (failure-prone bins marked *: {failure_bins})",
        "",
        "| bin | F^H % | D_P | dD | dD high | dD low | G_P | G_F | dG | seeds dG<0 | Peak-Final action dist |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for k in keys:
        mark = "*" if k in failure_bins else ""
        fr = fmt(100 * bin_info(k)["future_bad_tracking_rate"], 0, False) if k != "all" else "-"
        cells = [
            f"{fmt(agg[m][k]['mean'])} ± {fmt(agg[m][k]['std'], 3, False)}"
            for m in (
                "peak_distance_mean",
                "delta_distance",
                "delta_high_distance",
                "delta_low_distance",
                "peak_preference_gap",
                "final_preference_gap",
                "delta_preference_gap",
            )
        ]
        negs = sum(named[(s, k)]["delta_preference_gap"] < 0 for s in seeds)
        pf = fmt(agg["peak_final_action_distance"][k]["mean"], 3, False)
        lines.append(f"| {k}{mark} | {fr} | " + " | ".join(cells) + f" | {negs}/{len(seeds)} | {pf} |")
    lines += [
        "",
        "## Failure-prone vs other phases (per seed: mean over failure bins minus mean over the rest)",
        "",
        "| metric | failure bins | other bins | failure - other (95% CI over seeds) | seeds <0 | "
        "Spearman(F^H, seed mean) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in contrast_rows:
        lines.append(
            f"| {r['metric']} | {fmt(r['failure_mean'])} | {fmt(r['other_mean'])} | "
            f"{fmt(r['failure_minus_other_mean'])} [{fmt(r['failure_minus_other_ci_lo'])}, "
            f"{fmt(r['failure_minus_other_ci_hi'])}] | {r['seeds_failure_lt_other']}/{r['n_seeds']} | "
            f"{fmt(r['spearman_failrate_vs_seedmean'], 2)} |"
        )
    lines += [
        "",
        "Spearman correlations use only the phase bins as observations; treat them as descriptive.",
        "",
        "## Pipeline checks",
        "",
        f"- action space mode: {checks['action_space_mode']}; observation normaliser applied with update=False: "
        f"{checks['obs_normalization_applied']}",
        f"- dataset action entries inside the actor's tanh range: {checks['dataset_actions_within_actor_range']:.4f}",
        f"- mean |a|: dataset {checks['dataset_abs_action_mean']:.3f}; policies "
        + ", ".join(f"{k} {v:.3f}" for k, v in checks["policy_abs_action_mean"].items()),
        f"- samples per bin: {sorted(set(checks['samples_per_bin']))}; high {min(n_high)}-{max(n_high)}, "
        f"low {min(n_low)}-{max(n_low)} per bin",
        f"- episodes: episode_id blocks {meta['episodes_episode_id']}, AW dones|truncations bounds "
        f"{meta['episodes_aw_bounds']}",
        "",
        "## Console summary",
        "",
        "```text",
        console,
        "```",
        "",
        "Files: "
        + ", ".join(
            [f"sample_level_alignment_seed{s}.npz" for s in seeds]
            + [
                "phase_summary_per_seed.csv",
                "phase_summary_aggregated.csv",
                "utility_alignment_per_seed.csv",
                "utility_alignment_aggregated.csv",
                "utility_alignment_global_quantile_per_seed.csv",
                "failure_phase_contrast.csv",
                "run_info.json",
                *plot_files,
            ]
        )
        + f"; manifest {args.manifest}.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print()
    print(console)
    print(f"\nwritten: {out}/ ({', '.join(plot_files)})")


if __name__ == "__main__":
    main()
