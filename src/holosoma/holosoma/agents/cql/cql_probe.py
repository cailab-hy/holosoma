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

Distance-gate pre-scan columns (--gate-sim, on by default)
==========================================================
Per checkpoint, K policy samples per probe state (curr at s, next at s') get a
per-dim sigma-normalized RMS distance to a_D(s); the CSV records the distance
quantiles (does a far "shell" exist, and does it dry up as the policy
converges?) and, at candidate thresholds c (quantiles of the pooled distance
distribution), the CQL softmax mass split core/shell/rand plus the virtual
HARD-gate redistribution — "where the push-down fire would go" if core
samples were masked out — with the retained conservative gap
(gate_cql_gap_base -> gate_<c>_cql_gap_gated) and the shell-vs-data Q ordering
(gate_<c>_q_shell_minus_q_data > 0 marks the shell as overestimation targets).
Per-sample distance/Q matrices and fixed-bin histograms land in the npz for
figures. This is the no-training DG-CQL go/no-go + c calibration scan.

Refined-counterfactual gateway scan (--refine-sim, on by default)
=================================================================
K pi-samples per state are moved 2/5/10 steps of normalized action-space
ascent on J(a) = minQ(s,a) - beta * cos(phi(s,a), phi(s,a_D)) with a FROZEN
critic (gradients w.r.t. the action only), then the per-state best-Q reached
point is measured: Q vs Q(s,a_D), sigma-RMS distance (does it leave the pi
ring?), the feature-cos proxy the search used, and the TRUE full-parameter
grad-cos coupling on the first --refine-pairs states. beta=0 is the built-in
max-backup-like control (pure Q ascent -> expected to ride onto the data
ridge = maximum coupling). Go/no-go: refine_<beta>_s<steps>_quadrant_frac —
the fraction of states whose reached point has Q >= Q(s,a_D) AND grad-cos
below --refine-lowcos. refine_<beta>_featcos_gradcos_pearson answers the
"cos != NTK" worry inside the same scan; per-state scatter arrays (Q, d,
featcos, gradcos, reached actions at the final step) land in the npz.

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

    gate_sim: bool = True
    """Distance-gate pre-scan: per-state distributions of ||a_sample - a_D||_sigma over K
    policy samples (curr at s, next at s'), CQL softmax mass split into core/shell/rand at
    candidate thresholds c, and the virtual hard-gate redistribution ("where would the
    push-down fire go if core samples were gated out")."""

    gate_samples: int = 32
    """Policy samples per state (K) for the gate pre-scan, mirroring cql_num_action_samples."""

    gate_quantiles: str = "0.3,0.4,0.5"
    """Candidate gate thresholds c as quantiles of the pooled per-checkpoint distance
    distribution (comma separated, each in (0,1))."""

    refine_sim: bool = True
    """Refined-counterfactual gateway scan: move K pi-samples by action-space ascent on
    Q(s,a) - beta*cos(phi(s,a), phi(s,a_D)) (frozen critic, gradients w.r.t. the action
    only), then measure whether the reached points fill the high-Q / low-coupling quadrant
    (true full-parameter grad-cos, not just the feature-cos proxy)."""

    refine_samples: int = 32
    """Pi samples per probe state that get refined."""

    refine_steps: str = "2,5,10"
    """Ascent step counts at which reached points are measured (comma separated)."""

    refine_betas: str = "0,1,4"
    """Repulsion strengths beta; beta=0 is the pure-Q-ascent control (max-backup-like)."""

    refine_step_size: float = 0.1
    """Per-ascent-step movement in sigma-RMS units (normalized-gradient step)."""

    refine_pairs: int = 128
    """Probe states on which the true grad-cos coupling is measured per (beta, step)."""

    refine_lowcos: float = 0.25
    """Grad-cos threshold below which a reached point counts as "low coupling" for the
    quadrant fraction."""

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
# distance-gate pre-scan: does a "shell" of far-from-data policy samples exist,
# where does the CQL softmax aim today, and where WOULD it aim under a gate?
# ---------------------------------------------------------------------------

_GATE_HIST_BINS = 60
_GATE_HIST_MAX = 6.0  # sigma-RMS units; larger distances clamp into the last bin
_GATE_SEED_OFFSET = 104729


def _parse_gate_quantiles(spec: str) -> tuple[float, ...]:
    quantiles = tuple(float(part) for part in re.split(r"[,\s]+", spec.strip()) if part)
    for quantile in quantiles:
        if not (0.0 < quantile < 1.0):
            raise ValueError(f"--gate-quantiles entries must be in (0, 1), got {quantile}")
    return quantiles


def _gate_hist(distances: torch.Tensor) -> np.ndarray:
    clamped = distances.reshape(-1).clamp(0.0, _GATE_HIST_MAX - 1e-6).float().cpu()
    return torch.histc(clamped, bins=_GATE_HIST_BINS, min=0.0, max=_GATE_HIST_MAX).long().numpy()


@torch.no_grad()
def run_gate_probe(
    algo: Any,
    tensors: dict[str, torch.Tensor],
    num_samples: int,
    quantiles: tuple[float, ...],
    seed: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Per-state distance distributions of CQL's policy samples and a virtual-gate simulation.

    For every probe state, K curr samples (pi at s) and K next samples (pi at
    s') are drawn through the same path the training code uses, and their
    per-dim sigma-normalized RMS distance to a_D(s) is recorded (both blocks
    anchor to a_D(s) because that is the action the CQL term compares against).

    The CQL softmax over [rand - log rho, curr - log pi, next - log pi] / T is
    then decomposed — per critic, averaged over the twins — into mass on the
    core (policy samples with d < c), the shell (d >= c), and the rand block,
    at candidate thresholds c set at the requested quantiles of the pooled
    distance distribution. The virtual HARD gate masks core samples out of the
    logsumexp and reports the redistributed shell/rand masses and the retained
    conservative gap — "where the push-down fire would go" without training.
    Sampling noise is seed-fixed so every checkpoint uses the same ruler.
    """
    _set_eval_mode(algo)
    torch.manual_seed(seed + _GATE_SEED_OFFSET)
    temperature = float(getattr(algo, "_temperature", getattr(algo.config, "cql_temperature", 1.0)))

    observations = algo.obs_normalizer(tensors["observations"])
    next_observations = algo.obs_normalizer(tensors["next_observations"])
    critic_observations = algo.critic_obs_normalizer(tensors["critic_observations"])
    a_data = _to_critic_actions(algo, tensors["actions"]).detach()
    num_states, action_dim = a_data.shape
    sigma = a_data.std(dim=0, unbiased=False).clamp_min(1e-6)  # fixed given the fixed probe set

    def _expand(x: torch.Tensor) -> torch.Tensor:
        return x[:, None, :].expand(num_states, num_samples, -1).reshape(num_states * num_samples, -1)

    # RNG order fixed: curr, next, rand — mirrors the training block.
    curr_actions, curr_logp = algo.actor.get_actions_and_log_probs(_expand(observations))
    next_actions, next_logp = algo.actor.get_actions_and_log_probs(_expand(next_observations))
    action_scale = algo.actor.action_scale.to(device=a_data.device, dtype=a_data.dtype)
    action_bias = algo.actor.action_bias.to(device=a_data.device, dtype=a_data.dtype)
    rand_actions = torch.empty(num_states * num_samples, action_dim, device=a_data.device, dtype=a_data.dtype).uniform_(-1.0, 1.0)
    rand_actions = rand_actions * action_scale + action_bias
    if bool(getattr(algo.config, "use_tanh", True)):
        random_density = math.log(0.5) * action_dim - torch.log(action_scale + 1e-6).sum()
    else:
        random_density = math.log(0.5) * action_dim

    a_data_rep = _expand(a_data)
    dist_curr = (((curr_actions - a_data_rep) / sigma).pow(2).mean(dim=-1)).sqrt().view(num_states, num_samples)
    dist_next = (((next_actions - a_data_rep) / sigma).pow(2).mean(dim=-1)).sqrt().view(num_states, num_samples)
    dist_policy = torch.cat([dist_curr, dist_next], dim=1)  # [N, 2K]

    cobs_rep = _expand(critic_observations)
    q1_data, q2_data = _chunked_pair(algo.qnet, critic_observations, a_data)
    q1_curr, q2_curr = _chunked_pair(algo.qnet, cobs_rep, curr_actions)
    q1_next, q2_next = _chunked_pair(algo.qnet, cobs_rep, next_actions)
    q1_rand, q2_rand = _chunked_pair(algo.qnet, cobs_rep, rand_actions)
    q_policy_min = torch.minimum(
        torch.cat([q1_curr, q1_next]), torch.cat([q2_curr, q2_next])
    ).view(2, num_states, num_samples).permute(1, 0, 2).reshape(num_states, 2 * num_samples)
    q_data_min_mean = float(torch.minimum(q1_data, q2_data).mean().item())

    def _logits(q_rand: torch.Tensor, q_curr: torch.Tensor, q_next: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                (q_rand.view(num_states, num_samples) - random_density),
                q_curr.view(num_states, num_samples) - curr_logp.view(num_states, num_samples),
                q_next.view(num_states, num_samples) - next_logp.view(num_states, num_samples),
            ],
            dim=1,
        ) / temperature

    logits_twins = [_logits(q1_rand, q1_curr, q1_next), _logits(q2_rand, q2_curr, q2_next)]
    q_data_twins = [q1_data, q2_data]
    rand_block = slice(0, num_samples)
    policy_block = slice(num_samples, 3 * num_samples)

    row: dict[str, float] = {}
    for name, dist in (("curr", dist_curr), ("next", dist_next)):
        qs = torch.quantile(dist.reshape(-1).float(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=dist.device))
        for label, value in zip(("p10", "p25", "p50", "p75", "p90"), qs):
            row[f"dist_{name}_{label}"] = float(value.item())

    gap_base = 0.0
    for logits, q_data_twin in zip(logits_twins, q_data_twins):
        gap_base += float((temperature * torch.logsumexp(logits, dim=1) - q_data_twin).mean().item())
    row["gate_cql_gap_base"] = gap_base / 2.0

    for quantile in quantiles:
        tag = f"c{int(round(quantile * 100)):02d}"
        threshold = float(torch.quantile(dist_policy.reshape(-1).float(), quantile).item())
        shell_mask = dist_policy >= threshold  # [N, 2K] over the policy block
        p_core = p_shell = p_rand = 0.0
        gated_p_shell = gated_p_rand = 0.0
        gap_gated = 0.0
        for logits, q_data_twin in zip(logits_twins, q_data_twins):
            weights = torch.softmax(logits, dim=1)
            rand_mass = weights[:, rand_block].sum(dim=1)
            policy_weights = weights[:, policy_block]
            shell_mass = (policy_weights * shell_mask).sum(dim=1)
            core_mass = (policy_weights * (~shell_mask)).sum(dim=1)
            p_rand += float(rand_mass.mean().item())
            p_shell += float(shell_mass.mean().item())
            p_core += float(core_mass.mean().item())

            gated_logits = logits.clone()
            gated_logits[:, policy_block] = torch.where(
                shell_mask, gated_logits[:, policy_block], torch.full_like(gated_logits[:, policy_block], float("-inf"))
            )
            gated_weights = torch.softmax(gated_logits, dim=1)
            gated_p_rand += float(gated_weights[:, rand_block].sum(dim=1).mean().item())
            gated_p_shell += float(gated_weights[:, policy_block].sum(dim=1).mean().item())
            gap_gated += float((temperature * torch.logsumexp(gated_logits, dim=1) - q_data_twin).mean().item())

        row[f"gate_{tag}_threshold"] = threshold
        row[f"gate_{tag}_p_core"] = p_core / 2.0
        row[f"gate_{tag}_p_shell"] = p_shell / 2.0
        row[f"gate_{tag}_p_rand"] = p_rand / 2.0
        row[f"gate_{tag}_gated_p_shell"] = gated_p_shell / 2.0
        row[f"gate_{tag}_gated_p_rand"] = gated_p_rand / 2.0
        row[f"gate_{tag}_cql_gap_gated"] = gap_gated / 2.0
        row[f"gate_{tag}_shell_count_frac"] = float(shell_mask.float().mean().item())
        shell_q = q_policy_min[shell_mask]
        core_q = q_policy_min[~shell_mask]
        row[f"gate_{tag}_q_shell_minus_q_data"] = (
            float(shell_q.mean().item()) - q_data_min_mean if shell_q.numel() else float("nan")
        )
        row[f"gate_{tag}_q_core_minus_q_data"] = (
            float(core_q.mean().item()) - q_data_min_mean if core_q.numel() else float("nan")
        )

    arrays = {
        "gate_d_curr": dist_curr.float().cpu().numpy(),
        "gate_d_next": dist_next.float().cpu().numpy(),
        "gate_q_policy_min": q_policy_min.float().cpu().numpy(),
        "gate_hist_curr": _gate_hist(dist_curr),
        "gate_hist_next": _gate_hist(dist_next),
        "gate_sigma": sigma.float().cpu().numpy(),
        "gate_hist_edges": np.linspace(0.0, _GATE_HIST_MAX, _GATE_HIST_BINS + 1),
    }
    return row, arrays


# ---------------------------------------------------------------------------
# refined-counterfactual gateway scan: can action-space ascent MANUFACTURE
# high-Q / low-coupling targets that neither pi nor uniform sampling reaches?
# ---------------------------------------------------------------------------

_REFINE_SEED_OFFSET = 1299709


def _parse_refine_steps(spec: str) -> tuple[int, ...]:
    steps = sorted({int(part) for part in re.split(r"[,\s]+", spec.strip()) if part})
    if not steps or any(step <= 0 for step in steps):
        raise ValueError(f"--refine-steps must be positive ints, got {spec!r}")
    return tuple(steps)


def _parse_refine_betas(spec: str) -> tuple[float, ...]:
    betas = tuple(dict.fromkeys(float(part) for part in re.split(r"[,\s]+", spec.strip()) if part))
    if not betas or any(beta < 0 for beta in betas):
        raise ValueError(f"--refine-betas must be >= 0, got {spec!r}")
    return betas


def _beta_tag(beta: float) -> str:
    return "b" + f"{beta:g}".replace(".", "p")


def _twin_value_and_features(
    qnet: Any, cobs: torch.Tensor, actions: torch.Tensor, with_features: bool
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """min-twin Q plus per-twin penultimate features in a single forward per twin."""
    if with_features:
        f1 = qnet.q1.features(cobs, actions)
        f2 = qnet.q2.features(cobs, actions)
        q1 = qnet.q1.net[-1](f1).squeeze(-1)
        q2 = qnet.q2.net[-1](f2).squeeze(-1)
        return torch.minimum(q1, q2), f1, f2
    q1, q2 = qnet(cobs, actions)
    return torch.minimum(q1, q2), None, None


def _pair_gradcos(algo: Any, a_ref: torch.Tensor, a_data: torch.Tensor, cobs: torch.Tensor) -> np.ndarray:
    """True coupling per state: cos(grad_theta minQ(s, a_ref), grad_theta minQ(s, a_D))."""
    params = [p for p in algo.qnet.parameters() if p.requires_grad]
    out = np.zeros(a_ref.shape[0], dtype=np.float64)
    with torch.enable_grad():
        for i in range(a_ref.shape[0]):
            q1, q2 = algo.qnet(cobs[i : i + 1], a_ref[i : i + 1])
            g_ref = torch.autograd.grad(torch.minimum(q1, q2).squeeze(), params, allow_unused=True, materialize_grads=True)
            q1, q2 = algo.qnet(cobs[i : i + 1], a_data[i : i + 1])
            g_dat = torch.autograd.grad(torch.minimum(q1, q2).squeeze(), params, allow_unused=True, materialize_grads=True)
            out[i] = _grad_cosine(g_ref, g_dat)
    return out


def run_refine_probe(
    algo: Any,
    tensors: dict[str, torch.Tensor],
    num_samples: int,
    step_marks: tuple[int, ...],
    betas: tuple[float, ...],
    step_size: float,
    num_pairs: int,
    lowcos: float,
    seed: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Gateway scan for refined counterfactuals (no training, frozen critic).

    K pi-samples per probe state are moved by normalized action-space ascent on

        J(a) = minQ(s, a) - beta * 0.5*(cos(phi1(s,a), phi1(s,a_D)) + cos(phi2..))

    with gradients taken w.r.t. the ACTION only (autograd.grad — parameters are
    never touched, mirroring the detached-search defense of the training-time
    design). Each step moves exactly ``step_size`` sigma-RMS units before the
    action-bound clamp. At every requested step count the per-state BEST-Q
    reached point is measured: Q vs Q(s,a_D), distance (does it leave the
    ring?), feature-cos (the proxy the search used), and — on the first
    ``num_pairs`` states — the TRUE full-parameter grad-cos coupling, so the
    proxy-vs-truth question (cos != NTK) is answered in the same scan.
    beta=0 is the built-in max-backup-like control: pure Q ascent that is
    expected to converge onto the data ridge (maximum coupling, maximum
    self-harm). The go/no-go headline is quadrant_frac: the fraction of
    measured states whose reached point has Q >= Q(s,a_D) AND grad-cos below
    ``lowcos``.
    """
    _set_eval_mode(algo)
    torch.manual_seed(seed + _REFINE_SEED_OFFSET)

    observations = algo.obs_normalizer(tensors["observations"])
    critic_observations = algo.critic_obs_normalizer(tensors["critic_observations"])
    a_data = _to_critic_actions(algo, tensors["actions"]).detach()
    num_states, action_dim = a_data.shape
    device = a_data.device
    sigma = a_data.std(dim=0, unbiased=False).clamp_min(1e-6)
    action_scale = algo.actor.action_scale.to(device=device, dtype=a_data.dtype)
    action_bias = algo.actor.action_bias.to(device=device, dtype=a_data.dtype)
    bound_low, bound_high = action_bias - action_scale, action_bias + action_scale
    num_samples = max(1, num_samples)
    q1_net = getattr(algo.qnet, "q1", None)
    with_features = q1_net is not None and hasattr(q1_net, "features") and any(beta > 0 for beta in betas)
    if any(beta > 0 for beta in betas) and not with_features:
        logger.warning("[QProbe][refine] critic exposes no features(); running beta=0 only.")
        betas = (0.0,)

    with torch.no_grad():
        expanded_obs = observations[:, None, :].expand(num_states, num_samples, -1).reshape(num_states * num_samples, -1)
        start_actions = algo.actor.get_actions_and_log_probs(expanded_obs)[0].detach()
        q1_d, q2_d = _chunked_pair(algo.qnet, critic_observations, a_data)
        q_data_min = torch.minimum(q1_d, q2_d)

    max_steps = max(step_marks)
    combos = [("start", None, 0)] + [(f"{_beta_tag(beta)}_s{mark:02d}", beta, mark) for beta in betas for mark in step_marks]
    best = {
        tag: {
            "q": torch.full((num_states,), float("-inf"), device=device),
            "d": torch.zeros(num_states, device=device),
            "fc": torch.zeros(num_states, device=device),
            "act": torch.zeros(num_states, action_dim, device=device),
        }
        for tag, _, _ in combos
    }

    chunk_states = max(1, _FORWARD_CHUNK // num_samples)
    step_norm = step_size * math.sqrt(action_dim)  # L2 step in sigma-space == step_size RMS

    for start_idx in range(0, num_states, chunk_states):
        end_idx = min(start_idx + chunk_states, num_states)
        n_chunk = end_idx - start_idx
        rows_slice = slice(start_idx * num_samples, end_idx * num_samples)
        cobs_rep = (
            critic_observations[start_idx:end_idx, None, :]
            .expand(n_chunk, num_samples, -1)
            .reshape(n_chunk * num_samples, -1)
        )
        a_data_rep = a_data[start_idx:end_idx, None, :].expand(n_chunk, num_samples, -1).reshape(n_chunk * num_samples, -1)
        with torch.no_grad():
            if with_features:
                f1_data = algo.qnet.q1.features(cobs_rep, a_data_rep).detach()
                f2_data = algo.qnet.q2.features(cobs_rep, a_data_rep).detach()

        def _snapshot(tag: str, actions: torch.Tensor) -> None:
            with torch.no_grad():
                q_min, f1, f2 = _twin_value_and_features(algo.qnet, cobs_rep, actions, with_features)
                if with_features:
                    featcos = 0.5 * (
                        torch.nn.functional.cosine_similarity(f1, f1_data, dim=-1)
                        + torch.nn.functional.cosine_similarity(f2, f2_data, dim=-1)
                    )
                else:
                    featcos = torch.zeros_like(q_min)
                dist = (((actions - a_data_rep) / sigma).pow(2).mean(dim=-1)).sqrt()
                q_min = q_min.view(n_chunk, num_samples)
                pick = q_min.argmax(dim=1)
                gather = lambda t: t.view(n_chunk, num_samples).gather(1, pick[:, None]).squeeze(1)  # noqa: E731
                store = best[tag]
                store["q"][start_idx:end_idx] = q_min.gather(1, pick[:, None]).squeeze(1)
                store["d"][start_idx:end_idx] = gather(dist)
                store["fc"][start_idx:end_idx] = gather(featcos)
                store["act"][start_idx:end_idx] = (
                    actions.view(n_chunk, num_samples, -1).gather(1, pick[:, None, None].expand(-1, 1, action_dim)).squeeze(1)
                )

        _snapshot("start", start_actions[rows_slice])
        for beta in betas:
            actions = start_actions[rows_slice].clone()
            for step in range(1, max_steps + 1):
                actions = actions.detach().requires_grad_(True)
                objective, f1, f2 = _twin_value_and_features(algo.qnet, cobs_rep, actions, with_features)
                if with_features and beta > 0:
                    repel = 0.5 * (
                        torch.nn.functional.cosine_similarity(f1, f1_data, dim=-1)
                        + torch.nn.functional.cosine_similarity(f2, f2_data, dim=-1)
                    )
                    objective = objective - beta * repel
                grad_a = torch.autograd.grad(objective.sum(), actions)[0]
                with torch.no_grad():
                    grad_u = grad_a * sigma  # chain rule into sigma-space
                    unit = grad_u / grad_u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    actions = (actions.detach() + step_norm * unit * sigma).clamp(bound_low, bound_high)
                if step in step_marks:
                    _snapshot(f"{_beta_tag(beta)}_s{step:02d}", actions)

    # true coupling on the first num_pairs states, per combo
    pairs = min(num_pairs, num_states)
    gradcos: dict[str, np.ndarray] = {}
    for tag, _, _ in combos:
        gradcos[tag] = _pair_gradcos(algo, best[tag]["act"][:pairs], a_data[:pairs], critic_observations[:pairs])

    row: dict[str, float] = {}
    arrays: dict[str, np.ndarray] = {}
    start_q = best["start"]["q"]
    q_data_np = q_data_min.float().cpu().numpy()
    for tag, beta, mark in combos:
        store = best[tag]
        q_np = store["q"].float().cpu().numpy()
        d_np = store["d"].float().cpu().numpy()
        fc_np = store["fc"].float().cpu().numpy()
        gc = gradcos[tag]
        prefix = f"refine_{tag}"
        row[f"{prefix}_q_mean"] = float(q_np.mean())
        row[f"{prefix}_q_minus_qdata_mean"] = float((q_np - q_data_np).mean())
        row[f"{prefix}_q_above_data_frac"] = float((q_np >= q_data_np).mean())
        row[f"{prefix}_q_gain_mean"] = float((store["q"] - start_q).mean().item())
        row[f"{prefix}_d_p50"] = float(np.quantile(d_np, 0.5))
        row[f"{prefix}_d_p90"] = float(np.quantile(d_np, 0.9))
        row[f"{prefix}_featcos_mean"] = float(fc_np.mean())
        row[f"{prefix}_gradcos_mean"] = float(gc.mean())
        row[f"{prefix}_gradcos_p10"] = float(np.quantile(gc, 0.1))
        row[f"{prefix}_quadrant_frac"] = float(((gc < lowcos) & (q_np[:pairs] >= q_data_np[:pairs])).mean())
        arrays[f"{prefix}_gradcos"] = gc
        if mark == max_steps or tag == "start":
            arrays[f"{prefix}_q"] = q_np
            arrays[f"{prefix}_d"] = d_np
            arrays[f"{prefix}_featcos"] = fc_np
            arrays[f"{prefix}_actions"] = store["act"].float().cpu().numpy()
    for beta in betas:
        tag = f"{_beta_tag(beta)}_s{max_steps:02d}"
        fc_np = best[tag]["fc"].float().cpu().numpy()[:pairs]
        gc = gradcos[tag]
        if pairs >= 8 and np.std(fc_np) > 0 and np.std(gc) > 0:
            row[f"refine_{_beta_tag(beta)}_featcos_gradcos_pearson"] = float(np.corrcoef(fc_np, gc)[0, 1])
        else:
            row[f"refine_{_beta_tag(beta)}_featcos_gradcos_pearson"] = float("nan")
    return row, arrays


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
    gate_quantiles = _parse_gate_quantiles(probe_cli.gate_quantiles) if probe_cli.gate_sim else ()
    refine_steps = _parse_refine_steps(probe_cli.refine_steps) if probe_cli.refine_sim else ()
    refine_betas = _parse_refine_betas(probe_cli.refine_betas) if probe_cli.refine_sim else ()
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
            if probe_cli.gate_sim:
                gate_row, gate_arrays = run_gate_probe(
                    algo,
                    tensors,
                    num_samples=probe_cli.gate_samples,
                    quantiles=gate_quantiles,
                    seed=probe_cli.probe_seed,
                )
                row.update(gate_row)
                arrays.update(gate_arrays)
            if probe_cli.refine_sim:
                refine_row, refine_arrays = run_refine_probe(
                    algo,
                    tensors,
                    num_samples=probe_cli.refine_samples,
                    step_marks=refine_steps,
                    betas=refine_betas,
                    step_size=probe_cli.refine_step_size,
                    num_pairs=probe_cli.refine_pairs,
                    lowcos=probe_cli.refine_lowcos,
                    seed=probe_cli.probe_seed,
                )
                row.update(refine_row)
                arrays.update(refine_arrays)
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
            if probe_cli.gate_sim and gate_quantiles:
                tag = f"c{int(round(gate_quantiles[len(gate_quantiles) // 2] * 100)):02d}"
                logger.info(
                    "[QProbe][gate] step={} d_p50 curr/next={:.3f}/{:.3f} [{}] c={:.3f} "
                    "mass core/shell/rand={:.3f}/{:.3f}/{:.3f} gated shell/rand={:.3f}/{:.3f} "
                    "q_shell-q_data={:+.3f} gap base->gated={:.3f}->{:.3f}",
                    step,
                    row.get("dist_curr_p50", float("nan")),
                    row.get("dist_next_p50", float("nan")),
                    tag,
                    row.get(f"gate_{tag}_threshold", float("nan")),
                    row.get(f"gate_{tag}_p_core", float("nan")),
                    row.get(f"gate_{tag}_p_shell", float("nan")),
                    row.get(f"gate_{tag}_p_rand", float("nan")),
                    row.get(f"gate_{tag}_gated_p_shell", float("nan")),
                    row.get(f"gate_{tag}_gated_p_rand", float("nan")),
                    row.get(f"gate_{tag}_q_shell_minus_q_data", float("nan")),
                    row.get("gate_cql_gap_base", float("nan")),
                    row.get(f"gate_{tag}_cql_gap_gated", float("nan")),
                )
            if probe_cli.refine_sim and refine_steps and refine_betas:
                final_mark = max(refine_steps)
                repel_beta = max(refine_betas)
                base_tag = f"{_beta_tag(0.0)}_s{final_mark:02d}" if 0.0 in refine_betas else None
                repel_tag = f"{_beta_tag(repel_beta)}_s{final_mark:02d}"
                logger.info(
                    "[QProbe][refine] step={} start(Q>data,gradcos)={:.2f}/{:.3f} | beta=0 s{}: quadrant={:.3f} "
                    "gradcos={:.3f} | beta={} s{}: quadrant={:.3f} gradcos={:.3f} d_p50={:.2f} proxy_r={:.2f}",
                    step,
                    row.get("refine_start_q_above_data_frac", float("nan")),
                    row.get("refine_start_gradcos_mean", float("nan")),
                    final_mark,
                    row.get(f"refine_{base_tag}_quadrant_frac", float("nan")) if base_tag else float("nan"),
                    row.get(f"refine_{base_tag}_gradcos_mean", float("nan")) if base_tag else float("nan"),
                    repel_beta,
                    final_mark,
                    row.get(f"refine_{repel_tag}_quadrant_frac", float("nan")),
                    row.get(f"refine_{repel_tag}_gradcos_mean", float("nan")),
                    row.get(f"refine_{repel_tag}_d_p50", float("nan")),
                    row.get(f"refine_{_beta_tag(repel_beta)}_featcos_gradcos_pearson", float("nan")),
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
