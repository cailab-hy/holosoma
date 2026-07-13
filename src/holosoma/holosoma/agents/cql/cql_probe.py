"""Fixed-ruler Q probe across saved CQL checkpoints.

Motivation
==========
Training-time wandb curves measure a moving target with a moving ruler: each
point uses that step's minibatch and that step's actor, so when ``next_q``
sinks it is impossible to tell whether the critic drifted, the actor moved,
or twin-min bias amplified. This tool freezes the ruler:

1. A fixed probe set of transitions ``(s, a_D, s')`` is sampled ONCE from the
   offline dataset and stored RAW (unnormalized) in an npz file.
2. A reference checkpoint ("frozen", e.g. the best-eval step) provides a
   fixed next-action candidate ``a_frozen = pi_frozen(s')``.
3. Every checkpoint in the scan list is loaded in turn (env built once,
   ``algo.load()`` swaps actor/critic/normalizers) and evaluated on the same
   points: Q(s, a_D), Q(s', a_frozen), Q(s', a_current), target-net variants,
   Q1-Q2 twin gap, min vs avg, critic feature dot phi(s,a_D)*phi(s',a_cur),
   feature norms, srank, and an entropy-free deterministic TD residual.

Reading the scan table
======================
- Q(s', a_frozen) falls across checkpoints       -> critic deflation is real.
- Q(s', a_frozen) flat but Q(s', a_current) falls -> actor moved to low-Q land.
- Q(s, a_D) falls too                             -> global deflation.
- min(Q1,Q2) falls faster than (Q1+Q2)/2          -> twin-min amplification.
- cross-Q matrix Q_i(s', a_j): a column sinking for all critics i -> that
  actor's actions are genuinely rated bad; a row sinking for all action
  sources j -> that critic is deflated.

Gradient-leakage columns (--grad-leak, on by default)
=====================================================
Per checkpoint, the critic loss is split into components computed exactly as
in training on a fixed probe slice, and each component's first-order effect
on the probe-set mean data-Q is predicted as -lr * <grad qbar_data, grad L_c>:

- leak_pred_dqdata_cql < 0 with |cql| > |bellman|  -> "penalty leakage"
  confirmed: the CQL term pushes Q(s, a_D) down harder than Bellman restores
  it (the -Q(s,a_D) push-up inside the CQL loss is included, so this is the
  honest net effect of the conservative term).
- leak_pred_dqdata_net tracks the actual per-step drift of q_data_min_mean.
- ntk_featcos_pearson high (and hi-cos quartile leakage >> lo-cos quartile)
  -> the "cosine channel" is real: parameter-gradient leakage between
  Q(s, a_D) and Q(s, a_rand) rides on penultimate-feature similarity.
AdamW rescales per-parameter, so read signs/ratios, not absolute magnitudes.

Observations are stored raw because EmpiricalNormalization statistics evolve
during training; each checkpoint re-normalizes the raw obs with its own
restored normalizer state, reproducing exactly what that critic saw.

Usage
=====
Point at the run directory and select steps numerically — no per-checkpoint
paths needed:

    # every 10k-th checkpoint, anchored at the 50k actor
    python -m holosoma.agents.cql.cql_probe \\
        --checkpoint logs/WholeBodyTracking/<run> \\
        --step-stride 10000 --frozen-step 50000

    # or an explicit step list / range
    python -m holosoma.agents.cql.cql_probe \\
        --checkpoint logs/WholeBodyTracking/<run> \\
        --steps 30000,40000,50000,70000,90000 --frozen-step 50000
    python -m holosoma.agents.cql.cql_probe \\
        --checkpoint logs/WholeBodyTracking/<run> \\
        --step-min 30000 --step-max 90000 --step-stride 10000 --frozen-step 50000

``--checkpoint`` accepts a single .pt file, a glob (quote it), a run
directory, or a comma-separated list mixing local paths and
``wandb://entity/project/run/model_*.pt`` URIs (downloaded via the shared
checkpoint cache). ``--frozen-checkpoint`` (a path/URI) is still supported
and wins over ``--frozen-step``.
Extra CLI args are applied as overrides on top of the experiment config
embedded in the first checkpoint (same two-stage mechanism as eval_agent.py),
e.g. ``--training.eval-num-envs 1`` to keep the env small — the env is only
needed to construct the agent; the probe never steps it.

Outputs (default: ``<checkpoint_dir>/probe/``): ``probe_set.npz`` (reused
across scans), ``probe_scan.csv`` (one row per checkpoint), and
``probe_scan.npz`` (per-sample arrays, per-checkpoint actions, cross-Q matrix).

Both scan files are rewritten atomically after EVERY checkpoint, so an
interrupted scan keeps all rows measured so far. Re-running merges into the
existing CSV by step: re-measured steps replace their old row, other rows are
kept, and each row records the ``frozen_step`` anchor it was measured against
(a warning is logged when mixing anchors).
"""

from __future__ import annotations

import csv
import glob as _glob
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tyro
from loguru import logger

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG

PROBE_SET_SCHEMA_VERSION = 1
_FORWARD_CHUNK = 8192


@dataclass(frozen=True)
class QProbeCLI:
    checkpoint: str
    """Checkpoints to scan: a single model_*.pt, a glob (quote it), a run directory,
    or a comma-separated list mixing local paths and wandb://entity/project/run/model_*.pt URIs."""

    frozen_checkpoint: str | None = None
    """Reference checkpoint (local path or wandb:// URI) whose actor supplies a_frozen
    (e.g. the best-eval step). Defaults to the highest-step checkpoint in the scan list."""

    frozen_step: int | None = None
    """Pick the anchor by step number from the resolved checkpoint pool instead of a path
    (e.g. --frozen-step 50000). Ignored when --frozen-checkpoint is given."""

    steps: str | None = None
    """Explicit comma-separated steps to scan, e.g. "30000,40000,90000". Overrides the
    stride/range filters below."""

    step_min: int | None = None
    """Only scan checkpoints with step >= this."""

    step_max: int | None = None
    """Only scan checkpoints with step <= this."""

    step_stride: int = 0
    """Only scan steps divisible by this, e.g. 10000 keeps model_0030000/0040000/...
    out of a directory full of every-1k checkpoints. 0 scans everything matched."""

    probe_set: str | None = None
    """Path to the fixed probe-set npz. Built (and saved) on first run if missing.
    Defaults to <checkpoint_dir>/probe/probe_set.npz."""

    probe_size: int = 2048
    """Number of fixed transitions in the probe set (bootstrap rows only)."""

    probe_seed: int = 0
    """Seed for the one-time probe-set sampling."""

    output_dir: str | None = None
    """Where probe_scan.csv / probe_scan.npz are written. Defaults to <checkpoint_dir>/probe."""

    cross_q: bool = True
    """Also compute the full cross matrix Q_i(s', a_j) over scanned checkpoints (second load pass)."""

    grad_leak: bool = True
    """Per-checkpoint gradient-leakage diagnostic: split the critic loss into components
    (bellman / CQL conservative / dr3) and predict each component's first-order effect on
    the probe-set data-Q via -lr * <grad_theta q_data, grad_theta L_c>, plus a pairwise
    NTK-vs-feature-cosine correlation ("cosine channel" test)."""

    leak_rows: int = 512
    """Probe rows used for the component-gradient measurement (first K rows, fixed)."""

    leak_pairs: int = 128
    """Sample pairs for the per-sample NTK <grad Q(s,a_D), grad Q(s,a_rand)> vs feature-cosine test."""


# ---------------------------------------------------------------------------
# checkpoint resolution
# ---------------------------------------------------------------------------


_WANDB_PREFIX = "wandb://"


def _step_from_path(path: Path | str) -> int:
    match = re.search(r"model_(\d+)\.pt$", str(path))
    return int(match.group(1)) if match else -1


def _resolve_checkpoints(spec: str) -> list[str]:
    """Expand --checkpoint into an ordered list of references (local paths or wandb:// URIs)."""
    references: list[str] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if part.startswith(_WANDB_PREFIX):
            references.append(part)
            continue
        candidate = Path(part).expanduser()
        if candidate.is_file():
            matches = [candidate]
        elif candidate.is_dir():
            matches = list(candidate.glob("model_*.pt"))
        else:
            matches = [Path(p) for p in _glob.glob(str(candidate))]
        matched = [str(p) for p in matches if p.suffix == ".pt"]
        if not matched:
            raise FileNotFoundError(f"--checkpoint item {part!r} matched no .pt files")
        references.extend(matched)
    if not references:
        raise FileNotFoundError(f"No checkpoints matched --checkpoint {spec!r}")
    return sorted(set(references), key=lambda r: (_step_from_path(r), r))


def _parse_steps(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    steps = sorted({int(part) for part in re.split(r"[,\s]+", spec.strip()) if part})
    return steps or None


def _filter_references(
    references: list[str],
    steps: list[int] | None,
    step_min: int | None,
    step_max: int | None,
    step_stride: int,
) -> list[str]:
    """Narrow the resolved checkpoint pool by step; raises with the available steps on an empty result."""
    annotated = [(_step_from_path(reference), reference) for reference in references]
    available = sorted({step for step, _ in annotated if step >= 0})
    if steps is not None:
        wanted = set(steps)
        missing = wanted - {step for step, _ in annotated}
        if missing:
            raise FileNotFoundError(f"--steps {sorted(missing)} not found; available steps: {available}")
        return [reference for step, reference in annotated if step in wanted]
    picked = []
    for step, reference in annotated:
        if step_min is not None and step < step_min:
            continue
        if step_max is not None and step > step_max:
            continue
        if step_stride and (step < 0 or step % step_stride != 0):
            continue
        picked.append(reference)
    if not picked:
        raise FileNotFoundError(f"step filters removed every checkpoint; available steps: {available}")
    return picked


def _find_reference_by_step(references: list[str], step: int) -> str:
    matches = [reference for reference in references if _step_from_path(reference) == step]
    if not matches:
        available = sorted({_step_from_path(reference) for reference in references} - {-1})
        raise FileNotFoundError(f"--frozen-step {step} not among resolved checkpoints; available steps: {available}")
    return matches[0]


def _default_output_dir(explicit: str | None, first_reference: str) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    if first_reference.startswith(_WANDB_PREFIX):
        parts = first_reference[len(_WANDB_PREFIX) :].split("/")
        run_id = parts[2] if len(parts) > 2 else "wandb_run"
        return Path("logs") / "probe" / run_id
    return Path(first_reference).expanduser().parent / "probe"


# ---------------------------------------------------------------------------
# probe set (fixed ruler): sampled once, stored raw
# ---------------------------------------------------------------------------


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def build_probe_set(algo: Any, probe_size: int, seed: int, out_path: Path) -> dict[str, np.ndarray]:
    """Sample bootstrap transitions once with UNNORMALIZED observations and save them."""
    sample_offline_batch = getattr(algo, "_sample_offline_batch", None)
    if not callable(sample_offline_batch):
        raise RuntimeError(f"{type(algo).__name__} does not expose _sample_offline_batch; cannot build a probe set.")

    torch.manual_seed(seed)
    fields: dict[str, list[np.ndarray]] = {}
    collected = 0
    for draw in range(50):
        batch = sample_offline_batch(
            batch_size=probe_size,
            normalize_obs=_identity,
            normalize_critic_obs=_identity,
        )
        keep = batch["next"]["dones"].view(-1) == 0  # match training: bootstrap = ~dones
        if int(keep.sum().item()) == 0:
            continue
        chunk = {
            "observations": batch["observations"][keep],
            "actions": batch["actions"][keep],
            "critic_observations": batch["critic_observations"][keep],
            "next_observations": batch["next"]["observations"][keep],
            "next_critic_observations": batch["next"]["critic_observations"][keep],
            "rewards": batch["next"]["rewards"].view(-1)[keep],
            "effective_n_steps": batch["next"]["effective_n_steps"].view(-1)[keep],
            "truncations": batch["next"]["truncations"].view(-1)[keep],
        }
        for key, value in chunk.items():
            fields.setdefault(key, []).append(value.detach().float().cpu().numpy())
        collected += int(keep.sum().item())
        if collected >= probe_size:
            break
        logger.info("[QProbe] probe-set draw {}: {}/{} bootstrap rows collected", draw + 1, collected, probe_size)

    if collected < probe_size:
        logger.warning("[QProbe] only {}/{} bootstrap rows available; probing with what we have.", collected, probe_size)
    probe = {key: np.concatenate(chunks, axis=0)[:probe_size] for key, chunks in fields.items()}
    probe["schema_version"] = np.asarray(PROBE_SET_SCHEMA_VERSION)
    probe["probe_seed"] = np.asarray(seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **probe)
    logger.info("[QProbe] saved probe set with {} transitions to {}", int(probe["actions"].shape[0]), out_path)
    return probe


def load_probe_set(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        probe = {key: data[key] for key in data.files}
    version = int(probe.get("schema_version", np.asarray(-1)))
    if version != PROBE_SET_SCHEMA_VERSION:
        logger.warning("[QProbe] probe set schema_version={} != expected {}", version, PROBE_SET_SCHEMA_VERSION)
    logger.info("[QProbe] loaded probe set with {} transitions from {}", int(probe["actions"].shape[0]), path)
    return probe


def _probe_tensors(probe: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    keys = (
        "observations",
        "actions",
        "critic_observations",
        "next_observations",
        "next_critic_observations",
        "rewards",
        "effective_n_steps",
    )
    return {key: torch.from_numpy(probe[key]).float().to(device) for key in keys}


# ---------------------------------------------------------------------------
# per-checkpoint measurement
# ---------------------------------------------------------------------------


def _set_eval_mode(algo: Any) -> None:
    for name in ("actor", "qnet", "qnet_target", "obs_normalizer", "critic_obs_normalizer"):
        module = getattr(algo, name, None)
        if module is not None and hasattr(module, "eval"):
            module.eval()


def _to_critic_actions(algo: Any, actions: torch.Tensor) -> torch.Tensor:
    convert = getattr(algo, "_to_critic_actions", None)
    return convert(actions) if callable(convert) else actions


def _chunked_pair(
    fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    obs: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    firsts, seconds = [], []
    for start in range(0, obs.shape[0], _FORWARD_CHUNK):
        first, second = fn(obs[start : start + _FORWARD_CHUNK], actions[start : start + _FORWARD_CHUNK])
        firsts.append(first)
        seconds.append(second)
    return torch.cat(firsts), torch.cat(seconds)


def _critic_features(qnet: Any, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    if hasattr(qnet, "features"):
        return _chunked_pair(qnet.features, obs, actions)
    q1, q2 = getattr(qnet, "q1", None), getattr(qnet, "q2", None)
    if q1 is not None and hasattr(q1, "features") and q2 is not None and hasattr(q2, "features"):
        return _chunked_pair(lambda o, a: (q1.features(o, a), q2.features(o, a)), obs, actions)
    return None


def _srank(features: torch.Tensor, delta: float = 0.01) -> float:
    """srank_delta (Kumar et al., DR3): smallest k with sum-k singular values >= (1-delta) of total."""
    singular = torch.linalg.svdvals(features.detach().double().cpu())
    total = float(singular.sum().item())
    if total <= 0.0:
        return 0.0
    cumulative = torch.cumsum(singular, dim=0) / total
    return float(int((cumulative < (1.0 - delta)).sum().item()) + 1)


def _q_group(prefix: str, q1: torch.Tensor, q2: torch.Tensor, quantiles: bool = False) -> dict[str, float]:
    q_min = torch.minimum(q1, q2)
    row = {
        f"{prefix}_min_mean": float(q_min.mean().item()),
        f"{prefix}_avg_mean": float((0.5 * (q1 + q2)).mean().item()),
        f"{prefix}_q1_minus_q2_mean": float((q1 - q2).mean().item()),
        f"{prefix}_q1_minus_q2_abs_mean": float((q1 - q2).abs().mean().item()),
    }
    if quantiles:
        q = torch.quantile(q_min.float(), torch.tensor([0.05, 0.5, 0.95], device=q_min.device))
        row[f"{prefix}_min_p05"] = float(q[0].item())
        row[f"{prefix}_min_p50"] = float(q[1].item())
        row[f"{prefix}_min_p95"] = float(q[2].item())
    return row


@torch.no_grad()
def run_probe(
    algo: Any,
    tensors: dict[str, torch.Tensor],
    a_frozen: torch.Tensor | None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Measure one loaded checkpoint on the fixed probe set.

    Returns (scalar row for the CSV, per-sample arrays for the npz).
    """
    _set_eval_mode(algo)
    gamma = float(algo.config.gamma)
    reward_scale = float(getattr(algo, "reward_scale", 1.0))

    # Each checkpoint re-normalizes the RAW obs with its own restored normalizer.
    cobs = algo.critic_obs_normalizer(tensors["critic_observations"])
    ncobs = algo.critic_obs_normalizer(tensors["next_critic_observations"])
    nobs = algo.obs_normalizer(tensors["next_observations"])
    a_data = _to_critic_actions(algo, tensors["actions"])

    a_current = torch.cat(
        [algo.actor(nobs[start : start + _FORWARD_CHUNK])[0] for start in range(0, nobs.shape[0], _FORWARD_CHUNK)]
    ).detach()

    q1_data, q2_data = _chunked_pair(algo.qnet, cobs, a_data)
    q1_cur, q2_cur = _chunked_pair(algo.qnet, ncobs, a_current)
    qt1_cur, qt2_cur = _chunked_pair(algo.qnet_target, ncobs, a_current)

    row: dict[str, float] = {}
    row.update(_q_group("q_data", q1_data, q2_data, quantiles=True))
    row.update(_q_group("next_q_current", q1_cur, q2_cur, quantiles=True))
    row["next_q_current_target_min_mean"] = float(torch.minimum(qt1_cur, qt2_cur).mean().item())

    arrays: dict[str, np.ndarray] = {
        "q_data_min": torch.minimum(q1_data, q2_data).float().cpu().numpy(),
        "next_q_current_min": torch.minimum(q1_cur, q2_cur).float().cpu().numpy(),
        "a_current": a_current.float().cpu().numpy(),
    }

    if a_frozen is not None:
        q1_frz, q2_frz = _chunked_pair(algo.qnet, ncobs, a_frozen)
        qt1_frz, qt2_frz = _chunked_pair(algo.qnet_target, ncobs, a_frozen)
        row.update(_q_group("next_q_frozen", q1_frz, q2_frz))
        row["next_q_frozen_target_min_mean"] = float(torch.minimum(qt1_frz, qt2_frz).mean().item())
        row["next_q_current_minus_frozen_mean"] = row["next_q_current_min_mean"] - row["next_q_frozen_min_mean"]
        row["action_rms_current_vs_frozen"] = float((a_current - a_frozen).pow(2).mean().sqrt().item())
        arrays["next_q_frozen_min"] = torch.minimum(q1_frz, q2_frz).float().cpu().numpy()
    row["action_rms_current_vs_data"] = float((a_current - a_data).pow(2).mean().sqrt().item())

    # Entropy-free deterministic TD residual on the fixed set (bootstrap rows only).
    discount = gamma ** tensors["effective_n_steps"]
    td_target = reward_scale * tensors["rewards"] + discount * torch.minimum(qt1_cur, qt2_cur)
    residual = td_target - torch.minimum(q1_data, q2_data)
    row["td_target_det_mean"] = float(td_target.mean().item())
    row["bellman_residual_det_mean"] = float(residual.mean().item())
    row["bellman_residual_det_abs_mean"] = float(residual.abs().mean().item())

    features_data = _critic_features(algo.qnet, cobs, a_data)
    features_next = _critic_features(algo.qnet, ncobs, a_current)
    if features_data is not None and features_next is not None:
        f1_data, f2_data = features_data
        f1_next, f2_next = features_next
        dot = 0.5 * ((f1_data * f1_next).sum(dim=-1) + (f2_data * f2_next).sum(dim=-1))
        cos = 0.5 * (
            torch.nn.functional.cosine_similarity(f1_data, f1_next, dim=-1)
            + torch.nn.functional.cosine_similarity(f2_data, f2_next, dim=-1)
        )
        row["dr3_dot_mean"] = float(dot.mean().item())
        row["dr3_cosine_mean"] = float(cos.mean().item())
        row["feat_norm_data_mean"] = float(0.5 * (f1_data.norm(dim=-1) + f2_data.norm(dim=-1)).mean().item())
        row["feat_norm_next_current_mean"] = float(0.5 * (f1_next.norm(dim=-1) + f2_next.norm(dim=-1)).mean().item())
        row["srank_q1_data"] = _srank(f1_data)
        row["srank_q2_data"] = _srank(f2_data)
        row["srank_q1_next_current"] = _srank(f1_next)
        row["srank_q2_next_current"] = _srank(f2_next)

    log_alpha = getattr(algo, "log_alpha", None)
    if log_alpha is not None:
        row["alpha"] = float(log_alpha.exp().mean().item())
    return row, arrays


@torch.no_grad()
def compute_frozen_actions(algo: Any, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    """Deterministic pi_frozen(s'), normalized with the frozen checkpoint's own normalizer."""
    _set_eval_mode(algo)
    nobs = algo.obs_normalizer(tensors["next_observations"])
    return torch.cat(
        [algo.actor(nobs[start : start + _FORWARD_CHUNK])[0] for start in range(0, nobs.shape[0], _FORWARD_CHUNK)]
    ).detach()


@torch.no_grad()
def compute_cross_q(
    algo: Any,
    checkpoints: list[Path],
    tensors: dict[str, torch.Tensor],
    actions_by_checkpoint: list[np.ndarray],
) -> np.ndarray:
    """cross[i, j] = mean over probe set of min(Q1,Q2)_ckpt_i(s', a_ckpt_j(s'))."""
    device = tensors["next_critic_observations"].device
    action_sets = [torch.from_numpy(actions).float().to(device) for actions in actions_by_checkpoint]
    cross = np.full((len(checkpoints), len(action_sets)), np.nan, dtype=np.float64)
    for i, checkpoint_path in enumerate(checkpoints):
        algo.load(str(checkpoint_path))
        _set_eval_mode(algo)
        ncobs = algo.critic_obs_normalizer(tensors["next_critic_observations"])
        for j, actions in enumerate(action_sets):
            q1, q2 = _chunked_pair(algo.qnet, ncobs, actions)
            cross[i, j] = float(torch.minimum(q1, q2).mean().item())
    return cross


# ---------------------------------------------------------------------------
# gradient-leakage diagnostic: does the CQL penalty's parameter gradient push
# the data-Q down harder than the Bellman term restores it, and does the
# leakage travel through feature similarity (the "cosine channel")?
# ---------------------------------------------------------------------------


def _flat_dot(grads_a: tuple[torch.Tensor, ...], grads_b: tuple[torch.Tensor, ...]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for grad_a, grad_b in zip(grads_a, grads_b):
        total += (grad_a.double() * grad_b.double()).sum().cpu()
    return float(total)


def _grad_cosine(grads_a: tuple[torch.Tensor, ...], grads_b: tuple[torch.Tensor, ...]) -> float:
    dot = _flat_dot(grads_a, grads_b)
    norm_a = _flat_dot(grads_a, grads_a) ** 0.5
    norm_b = _flat_dot(grads_b, grads_b) ** 0.5
    if norm_a <= 0.0 or norm_b <= 0.0:
        return float("nan")
    return dot / (norm_a * norm_b)


@torch.enable_grad()
def run_grad_leak_probe(
    algo: Any,
    tensors: dict[str, torch.Tensor],
    num_rows: int,
    num_pairs: int,
    seed: int,
) -> dict[str, float]:
    """Component-wise first-order effect of the critic loss on the fixed data-Q.

    For each loss component L_c (bellman, weighted CQL conservative incl. its
    -Q(s,a_D) push-up part, dr3 when active) computed EXACTLY as in training on
    the first ``num_rows`` probe transitions:

        leak_pred_dqdata_<c> = -critic_lr * <grad_theta qbar_data, grad_theta L_c>

    i.e. the predicted change of the probe-set mean data-Q after one SGD step
    of that component alone (AdamW rescales per-parameter, so read signs and
    ratios, not absolute magnitudes). qbar_data = mean 0.5*(Q1+Q2)(s, a_D).

    The cosine-channel test computes, for ``num_pairs`` samples, the NTK entry
    k_i = <grad_theta Q1(s_i, a_D_i), grad_theta Q1(s_i, a_rand_i)> and its
    Pearson correlation with the penultimate-feature cosine of the same pair.
    Sampling noise (actor samples, random actions) is fixed by ``seed`` so
    every checkpoint is measured with the same ruler.
    """
    _set_eval_mode(algo)
    args = algo.config
    device = tensors["observations"].device
    torch.manual_seed(seed)

    rows = min(num_rows, int(tensors["observations"].shape[0]))
    observations = algo.obs_normalizer(tensors["observations"][:rows])
    next_observations = algo.obs_normalizer(tensors["next_observations"][:rows])
    critic_observations = algo.critic_obs_normalizer(tensors["critic_observations"][:rows])
    next_critic_observations = algo.critic_obs_normalizer(tensors["next_critic_observations"][:rows])
    dataset_actions = _to_critic_actions(algo, tensors["actions"][:rows]).detach()
    reward_scale = float(getattr(algo, "reward_scale", 1.0))
    rewards = reward_scale * tensors["rewards"][:rows]
    discount = float(args.gamma) ** tensors["effective_n_steps"][:rows]

    params = [p for p in algo.qnet.parameters() if p.requires_grad]

    def _qnet_grads(scalar: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(scalar, params, retain_graph=False, create_graph=False)

    # Direction of the quantity we care about: the probe-set mean data-Q.
    q1_data, q2_data = algo.qnet(critic_observations, dataset_actions)
    grads_data = _qnet_grads((0.5 * (q1_data + q2_data)).mean())

    # --- Bellman component, mirroring the training target exactly.
    with torch.no_grad():
        next_actions, next_log_probs = algo.actor.get_actions_and_log_probs(next_observations)
        next_q1_t, next_q2_t = algo.qnet_target(next_critic_observations, next_actions)
        next_v = torch.minimum(next_q1_t, next_q2_t).view(-1)
        if bool(getattr(args, "backup_entropy", False)):
            alpha = algo.log_alpha.exp().detach()
            next_v = next_v - alpha * next_log_probs.view(-1)
        q_target = (rewards.view(-1) + discount.view(-1) * next_v).clamp(min=-10000.0, max=10000.0)
    q1_b, q2_b = algo.qnet(critic_observations, dataset_actions)
    if getattr(args, "bellman_loss_type", "mse") == "huber":
        beta = float(getattr(args, "huber_beta", 1.0))
        bellman_loss = torch.nn.functional.smooth_l1_loss(q1_b, q_target, beta=beta) + torch.nn.functional.smooth_l1_loss(q2_b, q_target, beta=beta)
    else:
        bellman_loss = torch.nn.functional.mse_loss(q1_b, q_target) + torch.nn.functional.mse_loss(q2_b, q_target)
    grads_bellman = _qnet_grads(bellman_loss)

    learning_rate = float(getattr(args, "critic_learning_rate", 3e-4))
    row: dict[str, float] = {
        "leak_pred_dqdata_bellman": -learning_rate * _flat_dot(grads_data, grads_bellman),
        "leak_cos_bellman": _grad_cosine(grads_data, grads_bellman),
        "leak_gnorm_data": _flat_dot(grads_data, grads_data) ** 0.5,
        "leak_gnorm_bellman": _flat_dot(grads_bellman, grads_bellman) ** 0.5,
        "leak_rows": float(rows),
    }
    net_pred = row["leak_pred_dqdata_bellman"]

    # --- CQL conservative component (weighted, incl. the -Q(s,a_D) push-up part).
    cql_weight = float(getattr(algo, "_cql_weight", getattr(args, "cql_weight", 0.0)))
    if cql_weight > 0.0:
        if int(getattr(algo, "_num_near_actions", 0)) > 0:
            logger.warning("[QProbe][leak] cql_near_action_samples > 0 is not replicated; CQL component omits it.")
        num_repeat = int(getattr(algo, "_num_repeat_actions", getattr(args, "cql_num_action_samples", 10)))
        temperature = float(getattr(algo, "_temperature", getattr(args, "cql_temperature", 1.0)))
        batch = dataset_actions.shape[0]
        expanded_obs = observations[:, None, :].expand(batch, num_repeat, -1).reshape(batch * num_repeat, -1)
        expanded_cobs = critic_observations[:, None, :].expand(batch, num_repeat, -1).reshape(batch * num_repeat, -1)
        expanded_next_obs = next_observations[:, None, :].expand(batch, num_repeat, -1).reshape(batch * num_repeat, -1)
        with torch.no_grad():
            curr_actions, curr_logp = algo.actor.get_actions_and_log_probs(expanded_obs)
            next_actions_rep, next_logp = algo.actor.get_actions_and_log_probs(expanded_next_obs)
            action_scale = algo.actor.action_scale.to(device=device, dtype=dataset_actions.dtype)
            action_bias = algo.actor.action_bias.to(device=device, dtype=dataset_actions.dtype)
            rand_actions = torch.empty(batch * num_repeat, dataset_actions.shape[-1], device=device, dtype=dataset_actions.dtype).uniform_(-1.0, 1.0)
            rand_actions = rand_actions * action_scale + action_bias
            random_density = math.log(0.5) * dataset_actions.shape[-1] - torch.log(action_scale + 1e-6).sum()

        q1_c, q2_c = algo.qnet(critic_observations, dataset_actions)
        q1_rand, q2_rand = algo.qnet(expanded_cobs, rand_actions)
        q1_curr, q2_curr = algo.qnet(expanded_cobs, curr_actions)
        q1_next, q2_next = algo.qnet(expanded_cobs, next_actions_rep)
        cat_q1 = torch.cat(
            [
                (q1_rand - random_density).view(batch, num_repeat),
                q1_curr.view(batch, num_repeat) - curr_logp.view(batch, num_repeat),
                q1_next.view(batch, num_repeat) - next_logp.view(batch, num_repeat),
            ],
            dim=1,
        )
        cat_q2 = torch.cat(
            [
                (q2_rand - random_density).view(batch, num_repeat),
                q2_curr.view(batch, num_repeat) - curr_logp.view(batch, num_repeat),
                q2_next.view(batch, num_repeat) - next_logp.view(batch, num_repeat),
            ],
            dim=1,
        )
        cql1_loss = (torch.logsumexp(cat_q1 / temperature, dim=1) * temperature - q1_c).mean()
        cql2_loss = (torch.logsumexp(cat_q2 / temperature, dim=1) * temperature - q2_c).mean()
        conservative_loss = cql_weight * 0.5 * (cql1_loss + cql2_loss)
        log_cql_alpha = getattr(algo, "log_cql_alpha", None)
        if bool(getattr(args, "use_lagrange", False)) and log_cql_alpha is not None:
            conservative_loss = conservative_loss * log_cql_alpha.exp().detach().clamp(
                max=float(getattr(args, "cql_lagrange_max", 1e6))
            )
        grads_cql = _qnet_grads(conservative_loss)
        row["leak_pred_dqdata_cql"] = -learning_rate * _flat_dot(grads_data, grads_cql)
        row["leak_cos_cql"] = _grad_cosine(grads_data, grads_cql)
        row["leak_gnorm_cql"] = _flat_dot(grads_cql, grads_cql) ** 0.5
        row["leak_cql_vs_bellman"] = (
            row["leak_pred_dqdata_cql"] / abs(row["leak_pred_dqdata_bellman"])
            if row["leak_pred_dqdata_bellman"] != 0.0
            else float("nan")
        )
        net_pred += row["leak_pred_dqdata_cql"]
    else:
        row["leak_pred_dqdata_cql"] = float("nan")
        row["leak_cos_cql"] = float("nan")

    # --- DR3 component when active.
    if float(getattr(args, "dr3_weight", 0.0)) > 0.0:
        with torch.no_grad():
            dr3_next_actions, _ = algo.actor.get_actions_and_log_probs(next_observations)
        f1, f2 = algo.qnet.features(critic_observations, dataset_actions)
        nf1, nf2 = algo.qnet.features(next_critic_observations, dr3_next_actions)
        if bool(getattr(args, "dr3_normalize_features", False)):
            f1, f2 = torch.nn.functional.normalize(f1, dim=-1), torch.nn.functional.normalize(f2, dim=-1)
            nf1, nf2 = torch.nn.functional.normalize(nf1, dim=-1), torch.nn.functional.normalize(nf2, dim=-1)
        dr3_loss = float(args.dr3_weight) * (0.5 * ((f1 * nf1).sum(-1) + (f2 * nf2).sum(-1))).mean()
        grads_dr3 = _qnet_grads(dr3_loss)
        row["leak_pred_dqdata_dr3"] = -learning_rate * _flat_dot(grads_data, grads_dr3)
        net_pred += row["leak_pred_dqdata_dr3"]
    row["leak_pred_dqdata_net"] = net_pred

    # --- Cosine-channel test: per-sample NTK vs penultimate-feature cosine.
    q1_net = getattr(algo.qnet, "q1", None)
    if num_pairs > 0 and q1_net is not None and hasattr(q1_net, "features"):
        pairs = min(num_pairs, rows)
        with torch.no_grad():
            action_scale = algo.actor.action_scale.to(device=device, dtype=dataset_actions.dtype)
            action_bias = algo.actor.action_bias.to(device=device, dtype=dataset_actions.dtype)
            pair_rand = torch.empty(pairs, dataset_actions.shape[-1], device=device, dtype=dataset_actions.dtype).uniform_(-1.0, 1.0)
            pair_rand = pair_rand * action_scale + action_bias
            feat_data = q1_net.features(critic_observations[:pairs], dataset_actions[:pairs])
            feat_rand = q1_net.features(critic_observations[:pairs], pair_rand)
            feature_cos = torch.nn.functional.cosine_similarity(feat_data, feat_rand, dim=-1).cpu().numpy()
        q1_params = [p for p in q1_net.parameters() if p.requires_grad]
        ntk = np.zeros(pairs, dtype=np.float64)
        for i in range(pairs):
            grad_d = torch.autograd.grad(q1_net(critic_observations[i : i + 1], dataset_actions[i : i + 1]).squeeze(), q1_params)
            grad_r = torch.autograd.grad(q1_net(critic_observations[i : i + 1], pair_rand[i : i + 1]).squeeze(), q1_params)
            ntk[i] = _flat_dot(grad_d, grad_r)
        row["ntk_pair_dot_mean"] = float(ntk.mean())
        row["ntk_featcos_mean"] = float(feature_cos.mean())
        if pairs >= 8 and np.std(ntk) > 0 and np.std(feature_cos) > 0:
            row["ntk_featcos_pearson"] = float(np.corrcoef(feature_cos, ntk)[0, 1])
        else:
            row["ntk_featcos_pearson"] = float("nan")
        order = np.argsort(feature_cos)
        quartile = max(1, pairs // 4)
        row["ntk_dot_locos_mean"] = float(ntk[order[:quartile]].mean())
        row["ntk_dot_hicos_mean"] = float(ntk[order[-quartile:]].mean())
    return row


# ---------------------------------------------------------------------------
# incremental persistence: the CSV/npz are updated after EVERY checkpoint, so
# a killed scan keeps everything measured so far, and re-runs merge by step.
# ---------------------------------------------------------------------------


def _read_existing_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    try:
        with open(csv_path, newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        logger.warning("[QProbe] could not read existing {} ({}); starting a fresh table", csv_path, error)
        return []


def _row_step(row: dict[str, Any]) -> float:
    try:
        return float(row.get("step", "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _write_scan_csv(csv_path: Path, existing_rows: list[dict[str, str]], new_rows: list[dict[str, float]]) -> int:
    """Merge new rows over existing ones by step and atomically rewrite the CSV."""
    merged: dict[float, dict[str, Any]] = {}
    for row in existing_rows:
        merged[_row_step(row)] = dict(row)
    for row in new_rows:
        merged[_row_step(row)] = dict(row)
    ordered = sorted(merged.values(), key=_row_step)
    fieldnames = sorted({key for row in ordered for key in row}, key=lambda k: (k != "step", k != "frozen_step", k))
    tmp_path = csv_path.with_name(csv_path.name + ".tmp")
    with open(tmp_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(ordered)
    tmp_path.replace(csv_path)
    return len(ordered)


def _write_npz_atomic(npz_path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp_path = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    tmp_path.replace(npz_path)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run_probe_scan(probe_cli: QProbeCLI, tyro_config: ExperimentConfig) -> None:
    all_references = _resolve_checkpoints(probe_cli.checkpoint)
    references = _filter_references(
        all_references,
        _parse_steps(probe_cli.steps),
        probe_cli.step_min,
        probe_cli.step_max,
        probe_cli.step_stride,
    )
    if len(references) != len(all_references):
        logger.info("[QProbe] step filter selected {}/{} matched checkpoints", len(references), len(all_references))
    output_dir = _default_output_dir(probe_cli.output_dir, references[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_set_path = Path(probe_cli.probe_set).expanduser() if probe_cli.probe_set else output_dir / "probe_set.npz"

    # Materialize wandb:// references into local files (no-op for local paths).
    checkpoints = [Path(load_checkpoint(reference, str(output_dir))) for reference in references]
    if probe_cli.frozen_checkpoint is not None:
        if probe_cli.frozen_step is not None:
            logger.info("[QProbe] both --frozen-checkpoint and --frozen-step given; using --frozen-checkpoint")
        frozen_ref = probe_cli.frozen_checkpoint
        if not frozen_ref.startswith(_WANDB_PREFIX):
            frozen_ref = str(Path(frozen_ref).expanduser())
        frozen_path = Path(load_checkpoint(frozen_ref, str(output_dir)))
    elif probe_cli.frozen_step is not None:
        # Anchor is looked up in the FULL pool, so it does not have to be among the scanned steps.
        frozen_ref = _find_reference_by_step(all_references, probe_cli.frozen_step)
        frozen_path = Path(load_checkpoint(frozen_ref, str(output_dir)))
    else:
        frozen_path = checkpoints[-1]
        logger.warning(
            "[QProbe] no --frozen-checkpoint/--frozen-step given; using the last scanned checkpoint as anchor: {}",
            frozen_path,
        )

    # Fail fast on path typos BEFORE paying for simulator startup.
    missing = [str(p) for p in {*checkpoints, frozen_path} if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint file(s) not found: {missing} "
            "(check for a missing leading '/' or a typo in --checkpoint/--frozen-checkpoint)"
        )

    logger.info("[QProbe] scanning {} checkpoints; outputs -> {}", len(checkpoints), output_dir)
    for path in checkpoints:
        logger.info("[QProbe]   step={} {}", _step_from_path(path), path.name)

    env, device, simulation_app = setup_simulation_environment(tyro_config)
    try:
        algo_class = get_class(tyro_config.algo._target_)
        algo = algo_class(
            device=device,
            env=env,
            config=tyro_config.algo.config,
            log_dir=str(output_dir),
            multi_gpu_cfg=None,
        )
        algo.setup()

        if probe_set_path.exists():
            probe = load_probe_set(probe_set_path)
        else:
            probe = build_probe_set(algo, probe_cli.probe_size, probe_cli.probe_seed, probe_set_path)
        tensors = _probe_tensors(probe, device)

        logger.info("[QProbe] computing a_frozen (anchor) from {}", frozen_path)
        algo.load(str(frozen_path))
        a_frozen = compute_frozen_actions(algo, tensors)
        frozen_step = _step_from_path(frozen_path)

        csv_path = output_dir / "probe_scan.csv"
        npz_path = output_dir / "probe_scan.npz"
        existing_rows = _read_existing_rows(csv_path)
        if existing_rows:
            logger.info(
                "[QProbe] {} existing rows in {}; re-measured steps replace their old row, others are kept",
                len(existing_rows),
                csv_path,
            )
            previous_anchors = {row.get("frozen_step", "") for row in existing_rows} - {"", None}
            if previous_anchors - {str(float(frozen_step))}:
                logger.warning(
                    "[QProbe] existing rows were measured against a different anchor (frozen_step={}); "
                    "current anchor is {}. Rows are distinguishable via the frozen_step column.",
                    sorted(previous_anchors),
                    frozen_step,
                )

        def _flush_npz(actions: list[np.ndarray], arrays: dict[str, list[np.ndarray]], steps: list[int], cross: np.ndarray | None) -> None:
            payload: dict[str, np.ndarray] = {
                "steps": np.asarray(steps, dtype=np.int64),
                "a_frozen": a_frozen.float().cpu().numpy(),
                "a_current": np.stack(actions),
                "frozen_checkpoint": np.asarray(str(frozen_path)),
                "probe_set_path": np.asarray(str(probe_set_path)),
            }
            for key, values in arrays.items():
                payload[key] = np.stack(values)
            if cross is not None:
                payload["cross_q_min_mean"] = cross
            _write_npz_atomic(npz_path, payload)

        rows: list[dict[str, float]] = []
        actions_by_checkpoint: list[np.ndarray] = []
        per_step_arrays: dict[str, list[np.ndarray]] = {}
        scanned_steps: list[int] = []
        for index, checkpoint_path in enumerate(checkpoints):
            step = _step_from_path(checkpoint_path)
            algo.load(str(checkpoint_path))
            row, arrays = run_probe(algo, tensors, a_frozen)
            if probe_cli.grad_leak:
                row.update(
                    run_grad_leak_probe(
                        algo,
                        tensors,
                        num_rows=probe_cli.leak_rows,
                        num_pairs=probe_cli.leak_pairs,
                        seed=probe_cli.probe_seed,
                    )
                )
            row = {"step": float(step), "frozen_step": float(frozen_step), **row}
            rows.append(row)
            scanned_steps.append(step)
            actions_by_checkpoint.append(arrays.pop("a_current"))
            for key, value in arrays.items():
                per_step_arrays.setdefault(key, []).append(value)

            # Persist after every checkpoint: a killed scan keeps all rows so far.
            total_rows = _write_scan_csv(csv_path, existing_rows, rows)
            _flush_npz(actions_by_checkpoint, per_step_arrays, scanned_steps, cross=None)
            logger.info(
                "[QProbe] [{}/{}] step={} q_data_min={:.3f} next_q_frozen_min={:.3f} next_q_current_min={:.3f} "
                "cur_minus_frozen={:.3f} twin_gap={:.3f} drift_vs_frozen={:.4f} dr3_dot={:.3f} -> {} rows in {}",
                index + 1,
                len(checkpoints),
                step,
                row["q_data_min_mean"],
                row.get("next_q_frozen_min_mean", float("nan")),
                row["next_q_current_min_mean"],
                row.get("next_q_current_minus_frozen_mean", float("nan")),
                row["next_q_current_q1_minus_q2_abs_mean"],
                row.get("action_rms_current_vs_frozen", float("nan")),
                row.get("dr3_dot_mean", float("nan")),
                total_rows,
                csv_path.name,
            )
            if probe_cli.grad_leak:
                leak_cql = row.get("leak_pred_dqdata_cql", float("nan"))
                leak_bell = row.get("leak_pred_dqdata_bellman", float("nan"))
                verdict = ""
                if leak_cql == leak_cql and leak_cql < 0.0 and abs(leak_cql) > abs(leak_bell):
                    verdict = "  <-- CQL penalty leakage dominates the Bellman restoring force"
                logger.info(
                    "[QProbe][leak] step={} pred_dq_data/step: bellman={:+.5f} cql={:+.5f} net={:+.5f} "
                    "grad_cos(data,cql)={:.3f} ntk_featcos_pearson={:.3f} ntk hi/lo cos quartile={:.4g}/{:.4g}{}",
                    step,
                    leak_bell,
                    leak_cql,
                    row.get("leak_pred_dqdata_net", float("nan")),
                    row.get("leak_cos_cql", float("nan")),
                    row.get("ntk_featcos_pearson", float("nan")),
                    row.get("ntk_dot_hicos_mean", float("nan")),
                    row.get("ntk_dot_locos_mean", float("nan")),
                    verdict,
                )

        cross = None
        if probe_cli.cross_q and len(checkpoints) > 1:
            logger.info("[QProbe] computing cross-Q matrix over {} checkpoints", len(checkpoints))
            cross = compute_cross_q(algo, checkpoints, tensors, actions_by_checkpoint)
            logger.info("[QProbe] cross-Q rows=critic ckpt, cols=action ckpt, steps={}", scanned_steps)
            for i, step in enumerate(scanned_steps):
                logger.info("[QProbe]   critic@{:>7}: {}", step, np.array2string(cross[i], precision=3))
            _flush_npz(actions_by_checkpoint, per_step_arrays, scanned_steps, cross=cross)

        logger.info("[QProbe] done: {} and {}", csv_path, npz_path)
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    probe_cli, remaining_args = tyro.cli(QProbeCLI, return_unknown_args=True, add_help=False)

    first_checkpoint = _resolve_checkpoints(probe_cli.checkpoint)[0]
    saved_cfg, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(first_checkpoint)))
    eval_cfg = saved_cfg.get_eval_config()
    tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overrides applied on top of the experiment config embedded in the checkpoint.",
        config=TYRO_CONIFG,
    )
    run_probe_scan(probe_cli, tyro_config)


if __name__ == "__main__":
    main()
