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

Observations are stored raw because EmpiricalNormalization statistics evolve
during training; each checkpoint re-normalizes the raw obs with its own
restored normalizer state, reproducing exactly what that critic saw.

Usage
=====
    python -m holosoma.agents.cql.cql_probe \\
        --checkpoint "logs/WholeBodyTracking/<run>/model_00*.pt" \\
        --frozen-checkpoint logs/WholeBodyTracking/<run>/model_0050000.pt

``--checkpoint`` accepts a single .pt file, a glob, a run directory, or a
comma-separated list mixing local paths and ``wandb://entity/project/run/model_*.pt``
URIs (downloaded via the shared checkpoint cache).
Extra CLI args are applied as overrides on top of the experiment config
embedded in the first checkpoint (same two-stage mechanism as eval_agent.py),
e.g. ``--training.eval-num-envs 1`` to keep the env small — the env is only
needed to construct the agent; the probe never steps it.

Outputs (default: ``<checkpoint_dir>/probe/``): ``probe_set.npz`` (reused
across scans), ``probe_scan.csv`` (one row per checkpoint), and
``probe_scan.npz`` (per-sample arrays, per-checkpoint actions, cross-Q matrix).
"""

from __future__ import annotations

import csv
import glob as _glob
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
        references.extend(str(p) for p in matches if p.suffix == ".pt")
    if not references:
        raise FileNotFoundError(f"No checkpoints matched --checkpoint {spec!r}")
    return sorted(set(references), key=lambda r: (_step_from_path(r), r))


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
# entry point
# ---------------------------------------------------------------------------


def run_probe_scan(probe_cli: QProbeCLI, tyro_config: ExperimentConfig) -> None:
    references = _resolve_checkpoints(probe_cli.checkpoint)
    output_dir = _default_output_dir(probe_cli.output_dir, references[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_set_path = Path(probe_cli.probe_set).expanduser() if probe_cli.probe_set else output_dir / "probe_set.npz"

    # Materialize wandb:// references into local files (no-op for local paths).
    checkpoints = [Path(load_checkpoint(reference, str(output_dir))) for reference in references]
    if probe_cli.frozen_checkpoint is not None:
        frozen_path = Path(load_checkpoint(probe_cli.frozen_checkpoint, str(output_dir)))
    else:
        frozen_path = checkpoints[-1]
        logger.warning("[QProbe] --frozen-checkpoint not given; using the last scanned checkpoint: {}", frozen_path)

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

        logger.info("[QProbe] computing a_frozen from {}", frozen_path)
        algo.load(str(frozen_path))
        a_frozen = compute_frozen_actions(algo, tensors)

        rows: list[dict[str, float]] = []
        actions_by_checkpoint: list[np.ndarray] = []
        per_step_arrays: dict[str, list[np.ndarray]] = {}
        for checkpoint_path in checkpoints:
            step = _step_from_path(checkpoint_path)
            algo.load(str(checkpoint_path))
            row, arrays = run_probe(algo, tensors, a_frozen)
            row = {"step": float(step), **row}
            rows.append(row)
            actions_by_checkpoint.append(arrays.pop("a_current"))
            for key, value in arrays.items():
                per_step_arrays.setdefault(key, []).append(value)
            logger.info(
                "[QProbe] step={} q_data_min={:.3f} next_q_frozen_min={:.3f} next_q_current_min={:.3f} "
                "twin_gap={:.3f} drift_vs_frozen={:.4f} dr3_dot={:.3f}",
                step,
                row["q_data_min_mean"],
                row.get("next_q_frozen_min_mean", float("nan")),
                row["next_q_current_min_mean"],
                row["next_q_current_q1_minus_q2_abs_mean"],
                row.get("action_rms_current_vs_frozen", float("nan")),
                row.get("dr3_dot_mean", float("nan")),
            )

        cross = None
        if probe_cli.cross_q and len(checkpoints) > 1:
            logger.info("[QProbe] computing cross-Q matrix over {} checkpoints", len(checkpoints))
            cross = compute_cross_q(algo, checkpoints, tensors, actions_by_checkpoint)
            steps = [_step_from_path(p) for p in checkpoints]
            logger.info("[QProbe] cross-Q rows=critic ckpt, cols=action ckpt, steps={}", steps)
            for i, step in enumerate(steps):
                logger.info("[QProbe]   critic@{:>7}: {}", step, np.array2string(cross[i], precision=3))

        csv_path = output_dir / "probe_scan.csv"
        fieldnames = sorted({key for row in rows for key in row}, key=lambda k: (k != "step", k))
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("[QProbe] wrote {}", csv_path)

        npz_payload: dict[str, np.ndarray] = {
            "steps": np.asarray([_step_from_path(p) for p in checkpoints], dtype=np.int64),
            "a_frozen": a_frozen.float().cpu().numpy(),
            "a_current": np.stack(actions_by_checkpoint),
            "frozen_checkpoint": np.asarray(str(frozen_path)),
            "probe_set_path": np.asarray(str(probe_set_path)),
        }
        for key, values in per_step_arrays.items():
            npz_payload[key] = np.stack(values)
        if cross is not None:
            npz_payload["cross_q_min_mean"] = cross
        npz_path = output_dir / "probe_scan.npz"
        np.savez_compressed(npz_path, **npz_payload)
        logger.info("[QProbe] wrote {}", npz_path)
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
