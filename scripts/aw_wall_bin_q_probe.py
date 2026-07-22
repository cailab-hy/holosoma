#!/usr/bin/env python3
"""Probe wall-bin Q separation and actor drift across offline-RL checkpoints.

Rows in phase bin ``b`` use the exact Measurement-C episode label:

* FAIL: the row's episode terminates from bad tracking in ``[b, b + wall_span]``.
* SURV: every other episode represented by a row in bin ``b``.

For each checkpoint, the script restores the checkpoint observation normalizers,
deterministic actor, and online twin critic without launching a simulator. It
reports twin-min dataset-action Q separation, a global Q span, normalized delta,
and deterministic actor-to-dataset action drift for bins 4 and 5 by default.

Example:

    python scripts/aw_wall_bin_q_probe.py DATA.h5 \
      --manifest checkpoints.csv --output wall_bin_q_probe.csv

Manifest columns are ``run,ckpt,path``. Alternatively repeat
``--checkpoint RUN CKPT PATH`` for every checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "holosoma"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from holosoma.agents.cql.cql import Actor, DoubleQCritic  # noqa: E402
from holosoma.utils.safe_torch_import import torch  # noqa: E402


@dataclass(frozen=True)
class CheckpointSpec:
    run: str
    ckpt: str
    path: Path


@dataclass(frozen=True)
class ProbeModel:
    actor: Actor
    critic: DoubleQCritic
    obs_normalizer_state: Mapping[str, torch.Tensor] | None
    critic_obs_normalizer_state: Mapping[str, torch.Tensor] | None
    obs_normalization: bool
    actor_obs_dim: int
    critic_obs_dim: int
    action_dim: int


@dataclass(frozen=True)
class CellArrays:
    observations: np.ndarray
    critic_observations: np.ndarray
    actions: np.ndarray


def _resolve_h5_key(h5_file: h5py.File, candidates: Sequence[str]) -> str:
    for key in candidates:
        if key in h5_file:
            return key
    raise KeyError(f"None of the H5 keys exist: {list(candidates)}")


def _flat_bool(dataset: h5py.Dataset, num_rows: int) -> np.ndarray:
    return np.asarray(dataset[:num_rows]).reshape(-1).astype(bool, copy=False)


def _reward_hash(dataset: h5py.Dataset, num_rows: int) -> str:
    edge = min(1000, num_rows)
    first = np.asarray(dataset[:edge]).reshape(-1).astype(np.float64, copy=False)
    last = np.asarray(dataset[num_rows - edge : num_rows]).reshape(-1).astype(np.float64, copy=False)
    return hashlib.sha256(
        np.ascontiguousarray(first).tobytes() + np.ascontiguousarray(last).tobytes()
    ).hexdigest()[:16]


def measurement_c_episode_labels(
    phase_bins: np.ndarray,
    dones: np.ndarray,
    truncations: np.ndarray,
    bad_tracking: np.ndarray,
    bin_index: int,
    wall_span: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return FAIL and SURV row indices using Measurement-C's wall label.

    For a row in phase bin ``b``, its episode is FAIL iff the episode terminal
    row is marked bad-tracking and its terminal phase bin lies in
    ``[b, b + wall_span]``. This is the same ``fail_ep[episode_id[row]]`` rule
    used by :mod:`scripts.aw_measurement_c`.
    """

    num_rows = int(phase_bins.shape[0])
    if not (dones.shape[0] == truncations.shape[0] == bad_tracking.shape[0] == num_rows):
        raise ValueError("phase/reason arrays must have identical lengths")
    if num_rows == 0:
        raise ValueError("dataset is empty")

    episode_end = np.logical_or(dones, truncations).copy()
    episode_end[-1] = True
    ends = np.flatnonzero(episode_end)
    starts = np.concatenate((np.array([0], dtype=np.int64), ends[:-1] + 1))
    episode_id = np.empty(num_rows, dtype=np.int64)
    for episode_index, (start, end) in enumerate(zip(starts, ends)):
        episode_id[start : end + 1] = episode_index

    terminal_bin = phase_bins[ends]
    terminal_bad = bad_tracking[ends]
    fail_episode = terminal_bad & (terminal_bin >= bin_index) & (terminal_bin <= bin_index + wall_span)

    rows = np.flatnonzero(phase_bins == bin_index)
    is_fail = fail_episode[episode_id[rows]]
    return rows[is_fail], rows[~is_fail]


def _sample_without_replacement(
    rng: np.random.Generator,
    candidates: np.ndarray,
    count: int,
    label: str,
) -> np.ndarray:
    if candidates.shape[0] < count:
        raise ValueError(
            f"{label} has only {candidates.shape[0]:,} rows, fewer than the required {count:,}. "
            "Use a larger dataset or explicitly lower --cell-size."
        )
    return rng.choice(candidates, size=count, replace=False).astype(np.int64, copy=False)


def _read_rows(dataset: h5py.Dataset, row_indices: np.ndarray) -> np.ndarray:
    order = np.argsort(row_indices)
    sorted_indices = row_indices[order]
    sorted_values = np.asarray(dataset[sorted_indices], dtype=np.float32)
    values = np.empty_like(sorted_values)
    values[order] = sorted_values
    return values


def _strip_state_dict_prefixes(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "_orig_mod.")
    cleaned: dict[str, torch.Tensor] = {}
    for original_key, value in state_dict.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        cleaned[key] = value
    return cleaned


def _checkpoint_args(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args", {})
    if isinstance(args, Mapping):
        return dict(args)
    return vars(args)


def _single_obs_layout(size: int, key: str) -> dict[str, dict[str, int]]:
    return {key: {"start": 0, "end": size, "size": size}}


def load_probe_model(checkpoint_path: Path, device: torch.device) -> ProbeModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in checkpoint or "qnet_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint lacks actor_state_dict or qnet_state_dict: {checkpoint_path}")

    args = _checkpoint_args(checkpoint)
    if bool(args.get("use_cnn_encoder", False)):
        raise NotImplementedError("The wall-bin probe currently supports the MLP actor used by WBT, not CNNActor.")

    actor_state = _strip_state_dict_prefixes(checkpoint["actor_state_dict"])
    critic_state = _strip_state_dict_prefixes(checkpoint["qnet_state_dict"])
    required_actor_keys = ("net.0.weight", "fc_mu.0.weight")
    if any(key not in actor_state for key in required_actor_keys):
        raise ValueError(
            f"Unsupported actor layout in {checkpoint_path}; expected scalar CQL/AW-CQL/IQL Actor keys "
            f"{required_actor_keys}."
        )
    if "q1.net.0.weight" not in critic_state:
        raise ValueError(f"Unsupported critic layout in {checkpoint_path}; expected twin scalar Q networks.")

    actor_obs_dim = int(actor_state["net.0.weight"].shape[1])
    actor_hidden_dim = int(actor_state["net.0.weight"].shape[0])
    action_dim = int(actor_state["fc_mu.0.weight"].shape[0])
    critic_hidden_dim = int(critic_state["q1.net.0.weight"].shape[0])
    critic_obs_dim = int(critic_state["q1.net.0.weight"].shape[1]) - action_dim
    use_layer_norm = bool(args.get("use_layer_norm", "net.1.weight" in actor_state))
    use_tanh = bool(args.get("use_tanh", True))

    actor = Actor(
        obs_indices=_single_obs_layout(actor_obs_dim, "actor_obs"),
        obs_keys=["actor_obs"],
        n_act=action_dim,
        num_envs=1,
        hidden_dim=actor_hidden_dim,
        log_std_max=float(args.get("log_std_max", 0.0)),
        log_std_min=float(args.get("log_std_min", -5.0)),
        use_tanh=use_tanh,
        use_layer_norm=use_layer_norm,
        device=device,
    )
    critic = DoubleQCritic(
        obs_indices=_single_obs_layout(critic_obs_dim, "critic_obs"),
        obs_keys=["critic_obs"],
        n_act=action_dim,
        hidden_dim=critic_hidden_dim,
        use_layer_norm=use_layer_norm,
        device=device,
    )
    actor.load_state_dict(actor_state, strict=True)
    critic.load_state_dict(critic_state, strict=True)
    actor.eval()
    critic.eval()

    return ProbeModel(
        actor=actor,
        critic=critic,
        obs_normalizer_state=checkpoint.get("obs_normalizer_state"),
        critic_obs_normalizer_state=checkpoint.get("critic_obs_normalizer_state"),
        obs_normalization=bool(args.get("obs_normalization", True)),
        actor_obs_dim=actor_obs_dim,
        critic_obs_dim=critic_obs_dim,
        action_dim=action_dim,
    )


def _normalize_from_checkpoint(
    values: torch.Tensor,
    state: Mapping[str, torch.Tensor] | None,
    enabled: bool,
    label: str,
) -> torch.Tensor:
    if not enabled:
        return values
    if state is None or "_mean" not in state or "_std" not in state:
        raise ValueError(f"Checkpoint enables observation normalization but has no valid {label} state")
    mean = state["_mean"].to(device=values.device, dtype=values.dtype)
    std = state["_std"].to(device=values.device, dtype=values.dtype)
    if tuple(mean.shape[1:]) != tuple(values.shape[1:]):
        raise ValueError(f"{label} shape {tuple(mean.shape[1:])} does not match data {tuple(values.shape[1:])}")
    return (values - mean) / (std + 1e-2)


@torch.inference_mode()
def infer_q_and_drift(
    model: ProbeModel,
    arrays: CellArrays,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if arrays.observations.shape[1] != model.actor_obs_dim:
        raise ValueError(
            f"Actor observation mismatch: H5={arrays.observations.shape[1]}, checkpoint={model.actor_obs_dim}"
        )
    if arrays.critic_observations.shape[1] != model.critic_obs_dim:
        raise ValueError(
            "Critic observation mismatch: "
            f"H5={arrays.critic_observations.shape[1]}, checkpoint={model.critic_obs_dim}"
        )
    if arrays.actions.shape[1] != model.action_dim:
        raise ValueError(f"Action mismatch: H5={arrays.actions.shape[1]}, checkpoint={model.action_dim}")

    q_chunks: list[torch.Tensor] = []
    drift_chunks: list[torch.Tensor] = []
    for start in range(0, arrays.actions.shape[0], batch_size):
        end = min(start + batch_size, arrays.actions.shape[0])
        observations = torch.as_tensor(arrays.observations[start:end], device=device)
        critic_observations = torch.as_tensor(arrays.critic_observations[start:end], device=device)
        actions = torch.as_tensor(arrays.actions[start:end], device=device)
        observations = _normalize_from_checkpoint(
            observations,
            model.obs_normalizer_state,
            model.obs_normalization,
            "actor observation normalizer",
        )
        critic_observations = _normalize_from_checkpoint(
            critic_observations,
            model.critic_obs_normalizer_state,
            model.obs_normalization,
            "critic observation normalizer",
        )
        policy_actions = model.actor(observations)[0]
        q1, q2 = model.critic(critic_observations, actions)
        q_chunks.append(torch.minimum(q1, q2).float().cpu())
        drift_chunks.append(torch.linalg.vector_norm(policy_actions - actions, dim=-1).float().cpu())
    return torch.cat(q_chunks).numpy(), torch.cat(drift_chunks).numpy()


@torch.inference_mode()
def infer_q(
    model: ProbeModel,
    critic_observations_array: np.ndarray,
    actions_array: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    q_chunks: list[torch.Tensor] = []
    for start in range(0, actions_array.shape[0], batch_size):
        end = min(start + batch_size, actions_array.shape[0])
        critic_observations = torch.as_tensor(critic_observations_array[start:end], device=device)
        actions = torch.as_tensor(actions_array[start:end], device=device)
        critic_observations = _normalize_from_checkpoint(
            critic_observations,
            model.critic_obs_normalizer_state,
            model.obs_normalization,
            "critic observation normalizer",
        )
        q1, q2 = model.critic(critic_observations, actions)
        q_chunks.append(torch.minimum(q1, q2).float().cpu())
    return torch.cat(q_chunks).numpy()


def _load_manifest(path: Path) -> list[CheckpointSpec]:
    specs: list[CheckpointSpec] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"run", "ckpt", "path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain columns {sorted(required)}")
        for row in reader:
            specs.append(CheckpointSpec(row["run"], row["ckpt"], Path(row["path"]).expanduser()))
    return specs


def _parse_checkpoint_specs(args: argparse.Namespace) -> list[CheckpointSpec]:
    specs = _load_manifest(args.manifest) if args.manifest is not None else []
    for run, ckpt, path in args.checkpoint or []:
        specs.append(CheckpointSpec(run, ckpt, Path(path).expanduser()))
    if not specs:
        raise ValueError("Provide --manifest or at least one --checkpoint RUN CKPT PATH")
    for spec in specs:
        if not spec.path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {spec.path}")
    return specs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path)
    parser.add_argument("--sidecar", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=3,
        metavar=("RUN", "CKPT", "PATH"),
        help="Repeatable checkpoint specification.",
    )
    parser.add_argument("--output", type=Path, default=Path("wall_bin_q_probe.csv"))
    parser.add_argument("--bins", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--wall-span", type=int, default=1)
    parser.add_argument("--cell-size", type=int, default=2000)
    parser.add_argument("--span-sample-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    h5_path = args.h5.expanduser()
    sidecar_path = (args.sidecar or Path(f"{h5_path}.aw_weights.npz")).expanduser()
    if not h5_path.is_file():
        raise FileNotFoundError(f"H5 dataset not found: {h5_path}")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"AW sidecar not found: {sidecar_path}")
    specs = _parse_checkpoint_specs(args)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    with np.load(sidecar_path) as sidecar, h5py.File(h5_path, "r") as h5_file:
        if "phase_bin" not in sidecar:
            raise KeyError(f"Sidecar has no phase_bin array: {sidecar_path}")
        phase_bins = np.asarray(sidecar["phase_bin"], dtype=np.int64)
        num_rows = int(h5_file.attrs.get("num_samples", phase_bins.shape[0]))
        if phase_bins.shape[0] != num_rows or int(sidecar["n"]) != num_rows:
            raise ValueError("H5 and sidecar row counts do not match")

        observation_key = _resolve_h5_key(h5_file, ("observations",))
        critic_observation_key = _resolve_h5_key(h5_file, ("critic_observations",))
        action_key = _resolve_h5_key(h5_file, ("actions",))
        reward_key = _resolve_h5_key(h5_file, ("rewards", "next_rewards", "next.rewards", "next/rewards"))
        done_key = _resolve_h5_key(h5_file, ("dones", "next_dones", "next.dones", "next/dones"))
        truncation_key = _resolve_h5_key(
            h5_file,
            ("truncations", "next_truncations", "next.truncations", "next/truncations"),
        )
        bad_key = _resolve_h5_key(
            h5_file,
            (
                "next_done_bad_tracking",
                "done_bad_tracking",
                "next.done_bad_tracking",
                "next/done_bad_tracking",
            ),
        )

        reward_hash = _reward_hash(h5_file[reward_key], num_rows)
        if "rhash" in sidecar and str(sidecar["rhash"]) != reward_hash:
            raise ValueError("rhash mismatch: sidecar belongs to a different H5 build")

        dones = _flat_bool(h5_file[done_key], num_rows)
        truncations = _flat_bool(h5_file[truncation_key], num_rows)
        bad_tracking = _flat_bool(h5_file[bad_key], num_rows)
        cell_indices: dict[tuple[int, str], np.ndarray] = {}
        for bin_index in args.bins:
            fail_rows, surv_rows = measurement_c_episode_labels(
                phase_bins,
                dones,
                truncations,
                bad_tracking,
                bin_index,
                wall_span=args.wall_span,
            )
            cell_indices[(bin_index, "FAIL")] = _sample_without_replacement(
                rng, fail_rows, args.cell_size, f"bin {bin_index} FAIL"
            )
            cell_indices[(bin_index, "SURV")] = _sample_without_replacement(
                rng, surv_rows, args.cell_size, f"bin {bin_index} SURV"
            )
            print(
                f"[selection] bin={bin_index} available_surv={surv_rows.shape[0]:,} "
                f"available_fail={fail_rows.shape[0]:,} sampled_each={args.cell_size:,}"
            )

        span_rows = _sample_without_replacement(
            rng,
            np.arange(num_rows, dtype=np.int64),
            args.span_sample_size,
            "global span sample",
        )
        cell_arrays = {
            key: CellArrays(
                observations=_read_rows(h5_file[observation_key], indices),
                critic_observations=_read_rows(h5_file[critic_observation_key], indices),
                actions=_read_rows(h5_file[action_key], indices),
            )
            for key, indices in cell_indices.items()
        }
        span_critic_observations = _read_rows(h5_file[critic_observation_key], span_rows)
        span_actions = _read_rows(h5_file[action_key], span_rows)

    output_rows: list[dict[str, Any]] = []
    for spec in specs:
        print(f"[checkpoint] run={spec.run} ckpt={spec.ckpt} path={spec.path}")
        model = load_probe_model(spec.path, device)
        span_q = infer_q(model, span_critic_observations, span_actions, device, args.batch_size)
        q_span = float(np.quantile(span_q, 0.99) - np.quantile(span_q, 0.01))
        if not np.isfinite(q_span) or q_span <= 0.0:
            raise ValueError(f"Non-positive Q span for {spec.run}/{spec.ckpt}: {q_span}")

        for bin_index in args.bins:
            q_surv, d_surv_values = infer_q_and_drift(
                model, cell_arrays[(bin_index, "SURV")], device, args.batch_size
            )
            q_fail, d_fail_values = infer_q_and_drift(
                model, cell_arrays[(bin_index, "FAIL")], device, args.batch_size
            )
            q_surv_mean = float(q_surv.mean())
            q_fail_mean = float(q_fail.mean())
            delta = q_surv_mean - q_fail_mean
            d_surv = float(d_surv_values.mean())
            d_fail = float(d_fail_values.mean())
            d_ratio = d_fail / max(d_surv, 1e-12)
            row = {
                "run": spec.run,
                "ckpt": spec.ckpt,
                "bin": bin_index,
                "n_surv": int(q_surv.shape[0]),
                "n_fail": int(q_fail.shape[0]),
                "Q_surv": q_surv_mean,
                "Q_fail": q_fail_mean,
                "Δ": delta,
                "span": q_span,
                "Δ̂": delta / q_span,
                "d_surv": d_surv,
                "d_fail": d_fail,
                "d_ratio": d_ratio,
            }
            output_rows.append(row)
            print(
                f"[probe] run={spec.run} ckpt={spec.ckpt} bin={bin_index} "
                f"delta_hat={row['Δ̂']:.6f} Q_surv={q_surv_mean:.6f} Q_fail={q_fail_mean:.6f} "
                f"span={q_span:.6f} d_ratio={d_ratio:.6f}"
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "ckpt",
        "bin",
        "n_surv",
        "n_fail",
        "Q_surv",
        "Q_fail",
        "Δ",
        "span",
        "Δ̂",
        "d_surv",
        "d_fail",
        "d_ratio",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[done] wrote {len(output_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
