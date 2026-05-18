"""Dataset / normalisation primitives — canonical offline-RL common.

The removed ``offline_cql`` package no longer provides dataset re-exports;
callers should import these helpers from ``offline_rl.common.datasets``.
"""

from __future__ import annotations

import os
import textwrap
from typing import Any

import h5py
import numpy as np
import torch
from loguru import logger
from tensordict import TensorDict
from torch import nn
from torch.utils.data import Dataset

from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization


def create_frozen_normalizer(
    mean: torch.Tensor,
    std: torch.Tensor,
    count: int,
    device: torch.device | str,
    eps: float = 1e-2,
) -> EmpiricalNormalization:
    """Create an ``EmpiricalNormalization`` pre-initialised from dataset stats.

    The returned normalizer is frozen via ``eval()`` and ``until=count``.
    """
    obs_dim = mean.shape[-1]

    norm = EmpiricalNormalization(
        shape=obs_dim,
        device=device,
        eps=eps,
        until=count,
    )

    mean_dev = mean.to(device)
    std_dev = std.to(device)

    norm._mean.copy_(mean_dev)
    norm._std.copy_(std_dev)
    norm._var.copy_(std_dev.pow(2))
    norm.count.copy_(torch.tensor(count, dtype=torch.long, device=device))

    norm.eval()
    return norm


@torch.no_grad()
def validate_normalization(
    normalizer: nn.Module,
    raw_data: torch.Tensor,
    label: str = "obs",
    atol_mean: float = 0.15,
    atol_std: float = 0.35,
) -> dict[str, Any]:
    """Compare raw vs. normalised batch statistics for auditing."""
    is_identity = isinstance(normalizer, nn.Identity)

    raw_mean = raw_data.mean(dim=0)
    raw_std = raw_data.std(dim=0)

    if is_identity:
        norm_data = normalizer(raw_data)
    else:
        norm_data = normalizer(raw_data, update=False)

    norm_mean = norm_data.mean(dim=0)
    norm_std = norm_data.std(dim=0)

    obs_dim = raw_data.shape[-1]
    const_mask = raw_std < 1e-5
    num_const = int(const_mask.sum().item())
    non_const_mask = ~const_mask

    if is_identity:
        mean_ok = True
        std_ok = True
    else:
        mean_close = norm_mean.abs() < atol_mean
        mean_ok = bool(mean_close.float().mean() >= 0.95)

        if non_const_mask.any():
            std_close = (norm_std[non_const_mask] - 1.0).abs() < atol_std
            std_ok = bool(std_close.float().mean() >= 0.95)
        else:
            std_ok = True

    lines = [
        f"[{label}] Normalisation audit  ({raw_data.shape[0]:,} samples, {obs_dim} features)",
        f"  Raw   — mean: [{raw_mean.min().item():.4f}, {raw_mean.max().item():.4f}]  "
        f"std: [{raw_std.min().item():.6f}, {raw_std.max().item():.4f}]",
        f"  Norm  — mean: [{norm_mean.min().item():.4f}, {norm_mean.max().item():.4f}]  "
        f"std: [{norm_std.min().item():.6f}, {norm_std.max().item():.4f}]",
    ]
    if num_const > 0:
        lines.append(f"  Constant features: {num_const}/{obs_dim}")
    if is_identity:
        lines.append("  Normalizer: nn.Identity (no-op) — skipping mean/std checks.")
    else:
        lines.append(f"  Mean ≈ 0: {'PASS' if mean_ok else 'FAIL'}  (atol={atol_mean})")
        lines.append(
            f"  Std  ≈ 1: {'PASS' if std_ok else 'FAIL'}  "
            f"(atol={atol_std}, excl. {num_const} const features)"
        )
    report = "\n".join(lines)

    return {
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "mean_close_to_zero": mean_ok,
        "std_close_to_one": std_ok,
        "num_const_features": num_const,
        "report": report,
    }


_REQUIRED_H5_KEYS: tuple[str, ...] = (
    "actor_obs",
    "critic_obs",
    "actions",
    "rewards",
    "next_actor_obs",
    "next_critic_obs",
    "dones",
    "truncations",
)

_H5_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "actor_obs": ("observations",),
    "critic_obs": ("critic_observations",),
    "next_actor_obs": ("next_observations",),
    "next_critic_obs": ("next_critic_observations",),
    "actions": (),
    "rewards": (),
    "dones": (),
    "truncations": (),
}


def _resolve_h5_keys(h5_keys: set[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    missing: list[str] = []
    alias_used: list[str] = []

    for canonical in _REQUIRED_H5_KEYS:
        if canonical in h5_keys:
            resolved[canonical] = canonical
        else:
            found = False
            for alias in _H5_KEY_ALIASES.get(canonical, ()): 
                if alias in h5_keys:
                    resolved[canonical] = alias
                    alias_used.append(f"  {alias} → {canonical}")
                    found = True
                    break
            if not found:
                missing.append(canonical)

    if alias_used:
        logger.info("H5 key aliasing applied:\n" + "\n".join(alias_used))

    if missing:
        raise KeyError(
            f"H5 file is missing required datasets (after alias resolution): "
            f"{missing}. Available keys: {sorted(h5_keys)}"
        )

    return resolved


_EXPECTED_DTYPES: dict[str, str] = {
    "actor_obs": "float32",
    "critic_obs": "float32",
    "actions": "float32",
    "rewards": "float32",
    "next_actor_obs": "float32",
    "next_critic_obs": "float32",
    "dones": "int64",
    "truncations": "int64",
}


class OfflineDataset(Dataset):
    """GPU-resident static dataset for offline RL, loaded from an HDF5 file."""

    def __init__(
        self,
        path: str,
        device: torch.device | str = "cpu",
        *,
        expected_actor_obs_dim: int | None = None,
        expected_critic_obs_dim: int | None = None,
        expected_act_dim: int | None = None,
    ):
        super().__init__()

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Offline dataset not found: {path}")

        logger.info(f"Loading offline dataset from {path} ...")

        with h5py.File(path, "r") as f:
            key_map = _resolve_h5_keys(set(f.keys()))

            self.metadata: dict[str, Any] = {}
            for attr in (
                "task_name",
                "num_envs",
                "total_steps",
                "actor_obs_keys",
                "critic_obs_keys",
            ):
                if attr in f.attrs:
                    self.metadata[attr] = f.attrs[attr]

            raw: dict[str, np.ndarray] = {}
            for canonical in _REQUIRED_H5_KEYS:
                raw[canonical] = f[key_map[canonical]][()]

        n = raw["actor_obs"].shape[0]
        for key in _REQUIRED_H5_KEYS:
            if raw[key].shape[0] != n:
                raise ValueError(
                    f"Length mismatch: '{key}' has {raw[key].shape[0]} rows "
                    f"but 'actor_obs' has {n}. All datasets must share the "
                    f"first dimension."
                )

        if raw["actor_obs"].ndim != 2:
            raise ValueError(
                f"'actor_obs' must be 2-D [N, obs_dim], got shape {raw['actor_obs'].shape}"
            )
        if raw["critic_obs"].ndim != 2:
            raise ValueError(
                f"'critic_obs' must be 2-D [N, obs_dim], got shape {raw['critic_obs'].shape}"
            )
        if raw["actions"].ndim != 2:
            raise ValueError(
                f"'actions' must be 2-D [N, act_dim], got shape {raw['actions'].shape}"
            )
        for scalar_key in ("rewards", "dones", "truncations"):
            if raw[scalar_key].ndim != 1:
                raise ValueError(
                    f"'{scalar_key}' must be 1-D [N], got shape {raw[scalar_key].shape}"
                )

        actor_obs_dim = raw["actor_obs"].shape[1]
        critic_obs_dim = raw["critic_obs"].shape[1]
        act_dim = raw["actions"].shape[1]

        if raw["next_actor_obs"].shape[1] != actor_obs_dim:
            raise ValueError(
                f"'next_actor_obs' dim {raw['next_actor_obs'].shape[1]} != "
                f"'actor_obs' dim {actor_obs_dim}"
            )
        if raw["next_critic_obs"].shape[1] != critic_obs_dim:
            raise ValueError(
                f"'next_critic_obs' dim {raw['next_critic_obs'].shape[1]} != "
                f"'critic_obs' dim {critic_obs_dim}"
            )

        if expected_actor_obs_dim is not None and actor_obs_dim != expected_actor_obs_dim:
            raise ValueError(
                f"Expected actor_obs_dim={expected_actor_obs_dim}, "
                f"got {actor_obs_dim} from dataset"
            )
        if expected_critic_obs_dim is not None and critic_obs_dim != expected_critic_obs_dim:
            raise ValueError(
                f"Expected critic_obs_dim={expected_critic_obs_dim}, "
                f"got {critic_obs_dim} from dataset"
            )
        if expected_act_dim is not None and act_dim != expected_act_dim:
            raise ValueError(
                f"Expected act_dim={expected_act_dim}, "
                f"got {act_dim} from dataset"
            )

        for key in _REQUIRED_H5_KEYS:
            expected_dt = _EXPECTED_DTYPES[key]
            actual_dt = str(raw[key].dtype)
            if actual_dt != expected_dt:
                logger.warning(
                    f"Dataset key '{key}' has dtype {actual_dt}, "
                    f"expected {expected_dt}. Will cast."
                )

        trunc_np = raw["truncations"].astype(np.int64)
        dones_np = raw["dones"].astype(np.int64)
        invalid_trunc = int(np.sum((trunc_np == 1) & (dones_np == 0)))
        if invalid_trunc > 0:
            logger.warning(
                f"Found {invalid_trunc} transitions where truncation=1 but "
                f"done=0. This violates the invariant truncation ⊂ dones. "
                f"These may cause incorrect TD-target bootstrapping."
            )

        self.actor_obs = torch.from_numpy(raw["actor_obs"].astype(np.float32)).to(device)
        self.critic_obs = torch.from_numpy(raw["critic_obs"].astype(np.float32)).to(device)
        self.actions = torch.from_numpy(raw["actions"].astype(np.float32)).to(device)
        self.rewards = torch.from_numpy(raw["rewards"].astype(np.float32)).to(device)
        self.next_actor_obs = torch.from_numpy(raw["next_actor_obs"].astype(np.float32)).to(device)
        self.next_critic_obs = torch.from_numpy(raw["next_critic_obs"].astype(np.float32)).to(device)
        self.dones = torch.from_numpy(raw["dones"].astype(np.int64)).to(device)
        self.truncations = torch.from_numpy(raw["truncations"].astype(np.int64)).to(device)

        self.size: int = n
        self.actor_obs_dim: int = actor_obs_dim
        self.critic_obs_dim: int = critic_obs_dim
        self.act_dim: int = act_dim

        self._actor_obs_stats: tuple[torch.Tensor, torch.Tensor] | None = None
        self._critic_obs_stats: tuple[torch.Tensor, torch.Tensor] | None = None

        logger.info(
            f"Loaded {self.size:,} transitions  "
            f"(actor_obs={actor_obs_dim}, critic_obs={critic_obs_dim}, "
            f"act={act_dim})"
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "actor_obs": self.actor_obs[idx],
            "critic_obs": self.critic_obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_actor_obs": self.next_actor_obs[idx],
            "next_critic_obs": self.next_critic_obs[idx],
            "dones": self.dones[idx],
            "truncations": self.truncations[idx],
        }

    @torch.no_grad()
    def sample(self, batch_size: int) -> TensorDict:
        if batch_size > self.size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds dataset size ({self.size})"
            )

        idx = torch.randint(0, self.size, (batch_size,), device=self.actor_obs.device)

        out = TensorDict(
            {
                "observations": self.actor_obs[idx],
                "actions": self.actions[idx],
                "next": {
                    "rewards": self.rewards[idx],
                    "dones": self.dones[idx],
                    "truncations": self.truncations[idx],
                    "observations": self.next_actor_obs[idx],
                    "effective_n_steps": torch.ones(
                        batch_size, device=self.actor_obs.device, dtype=torch.long
                    ),
                },
            },
            batch_size=batch_size,
        )
        out["critic_observations"] = self.critic_obs[idx]
        out["next"]["critic_observations"] = self.next_critic_obs[idx]

        return out

    @torch.no_grad()
    def compute_obs_statistics(
        self,
        obs_type: str = "actor",
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if obs_type == "actor":
            if self._actor_obs_stats is not None:
                return self._actor_obs_stats
            data = self.actor_obs
        elif obs_type == "critic":
            if self._critic_obs_stats is not None:
                return self._critic_obs_stats
            data = self.critic_obs
        else:
            raise ValueError(f"obs_type must be 'actor' or 'critic', got '{obs_type}'")

        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True).clamp(min=eps)

        if obs_type == "actor":
            self._actor_obs_stats = (mean, std)
        else:
            self._critic_obs_stats = (mean, std)

        return mean, std

    def summary(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║            Offline CQL Dataset Summary                  ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"  Transitions   : {self.size:>12,}",
            f"  actor_obs_dim : {self.actor_obs_dim:>12}",
            f"  critic_obs_dim: {self.critic_obs_dim:>12}",
            f"  act_dim       : {self.act_dim:>12}",
            "──────────────────────────────────────────────────────────",
        ]

        for name, tensor in [
            ("actor_obs      ", self.actor_obs),
            ("critic_obs     ", self.critic_obs),
            ("actions        ", self.actions),
            ("rewards        ", self.rewards),
            ("next_actor_obs ", self.next_actor_obs),
            ("next_critic_obs", self.next_critic_obs),
            ("dones          ", self.dones),
            ("truncations    ", self.truncations),
        ]:
            t = tensor.float()
            lines.append(
                f"  {name}  "
                f"shape={str(list(tensor.shape)):>18}  "
                f"min={t.min().item():>10.4f}  "
                f"max={t.max().item():>10.4f}  "
                f"mean={t.mean().item():>10.4f}"
            )

        n_dones = int(self.dones.sum().item())
        n_truncations = int(self.truncations.sum().item())
        n_terminals = n_dones - n_truncations
        lines.extend([
            "──────────────────────────────────────────────────────────",
            f"  Total episodes (dones)     : {n_dones:>8,}",
            f"    of which terminal (death): {n_terminals:>8,}",
            f"    of which truncated       : {n_truncations:>8,}",
        ])

        if n_dones > 0:
            mean_ep_len = self.size / n_dones
            lines.append(f"  Approx mean episode length : {mean_ep_len:>8.1f}")

        if self.metadata:
            lines.append("──────────────────────────────────────────────────────────")
            lines.append("  Metadata:")
            for k, v in self.metadata.items():
                lines.append(f"    {k}: {v}")

        lines.append("╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def print_summary(self) -> None:
        logger.info("\n" + self.summary())


def validate_dataset_dry_run(
    path: str,
    device: torch.device | str = "cpu",
    sample_batch_size: int = 64,
) -> OfflineDataset:
    """Load a dataset, print a summary, and run basic sanity checks."""
    logger.info("=== Offline Dataset Dry-Run Validation ===")
    logger.info(f"File: {path}")

    ds = OfflineDataset(path, device=device)
    ds.print_summary()

    logger.info("Running shape / semantics checks ...")

    assert ds.actor_obs.shape == (ds.size, ds.actor_obs_dim), (
        f"actor_obs shape {ds.actor_obs.shape} != ({ds.size}, {ds.actor_obs_dim})"
    )
    assert ds.critic_obs.shape == (ds.size, ds.critic_obs_dim), (
        f"critic_obs shape {ds.critic_obs.shape} != ({ds.size}, {ds.critic_obs_dim})"
    )
    assert ds.next_actor_obs.shape == ds.actor_obs.shape, (
        f"next_actor_obs shape {ds.next_actor_obs.shape} != actor_obs {ds.actor_obs.shape}"
    )
    assert ds.next_critic_obs.shape == ds.critic_obs.shape, (
        f"next_critic_obs shape {ds.next_critic_obs.shape} != critic_obs {ds.critic_obs.shape}"
    )
    assert ds.actions.shape == (ds.size, ds.act_dim)
    assert ds.rewards.shape == (ds.size,)
    assert ds.dones.shape == (ds.size,)
    assert ds.truncations.shape == (ds.size,)

    act_min = ds.actions.min().item()
    act_max = ds.actions.max().item()
    if act_min < -10.0 or act_max > 10.0:
        logger.warning(
            f"Action range [{act_min:.3f}, {act_max:.3f}] is very wide. "
            f"Verify these are truly post-scaled actor outputs."
        )
    else:
        logger.info(f"  Actions range: [{act_min:.4f}, {act_max:.4f}]  ✓")

    dones_unique = ds.dones.unique().tolist()
    trunc_unique = ds.truncations.unique().tolist()
    assert set(dones_unique).issubset({0, 1}), (
        f"dones has non-binary values: {dones_unique}"
    )
    assert set(trunc_unique).issubset({0, 1}), (
        f"truncations has non-binary values: {trunc_unique}"
    )
    logger.info(f"  dones values: {dones_unique}  ✓")
    logger.info(f"  truncations values: {trunc_unique}  ✓")

    actor_mean, actor_std = ds.compute_obs_statistics("actor")
    critic_mean, critic_std = ds.compute_obs_statistics("critic")
    logger.info(
        f"  Actor obs stats:  mean range [{actor_mean.min().item():.4f}, "
        f"{actor_mean.max().item():.4f}],  "
        f"std range [{actor_std.min().item():.6f}, {actor_std.max().item():.4f}]"
    )
    logger.info(
        f"  Critic obs stats: mean range [{critic_mean.min().item():.4f}, "
        f"{critic_mean.max().item():.4f}],  "
        f"std range [{critic_std.min().item():.6f}, {critic_std.max().item():.4f}]"
    )

    actor_const = int((actor_std.squeeze() < 1e-5).sum().item())
    critic_const = int((critic_std.squeeze() < 1e-5).sum().item())
    if actor_const > 0:
        logger.warning(
            f"  {actor_const}/{ds.actor_obs_dim} actor obs features have "
            f"near-zero variance (constant columns)."
        )
    if critic_const > 0:
        logger.warning(
            f"  {critic_const}/{ds.critic_obs_dim} critic obs features have "
            f"near-zero variance (constant columns)."
        )

    effective_batch = min(sample_batch_size, ds.size)
    batch = ds.sample(effective_batch)
    logger.info(f"  Test batch (size={effective_batch}):")
    logger.info(f"    observations       : {list(batch['observations'].shape)}")
    logger.info(f"    actions            : {list(batch['actions'].shape)}")
    logger.info(f"    critic_observations: {list(batch['critic_observations'].shape)}")
    logger.info(f"    next/observations  : {list(batch['next']['observations'].shape)}")
    logger.info(f"    next/rewards       : {list(batch['next']['rewards'].shape)}")
    logger.info(f"    next/dones         : {list(batch['next']['dones'].shape)}")
    logger.info(f"    next/truncations   : {list(batch['next']['truncations'].shape)}")
    logger.info(
        f"    next/effective_n_steps: {list(batch['next']['effective_n_steps'].shape)} "
        f"(all ones: {bool((batch['next']['effective_n_steps'] == 1).all())})"
    )
    logger.info(
        f"    next/critic_observations: "
        f"{list(batch['next']['critic_observations'].shape)}"
    )

    item = ds[0]
    logger.info(f"  __getitem__(0) keys: {sorted(item.keys())}")

    logger.info("=== Dry-run validation PASSED ===")
    return ds


__all__ = [
    "OfflineDataset",
    "create_frozen_normalizer",
    "validate_dataset_dry_run",
    "validate_normalization",
]
