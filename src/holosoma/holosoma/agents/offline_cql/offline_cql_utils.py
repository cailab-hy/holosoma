"""Utilities for offline CQL: dataset loading, normalizer initialisation, and
checkpoint helpers.

Parallels ``fast_sac_utils.py`` which provides ``SimpleReplayBuffer``,
``EmpiricalNormalization``, and ``save_params``.

``EmpiricalNormalization`` is reused directly from fast_sac_utils — it is
NOT duplicated here.  The factory function ``create_frozen_normalizer``
initialises an ``EmpiricalNormalization`` from pre-computed dataset
statistics and locks it against future updates, providing a safe,
audit-friendly normalizer for offline training.
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
from torch.amp import GradScaler
from torch.utils.data import Dataset

from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization, cpu_state


# ── Normalizer load helper ────────────────────────────────────────────

def _load_normalizer_safe(
    module: nn.Module,
    saved_state: dict[str, Any],
    label: str,
) -> None:
    """Load a normalizer state dict with a clear error on type mismatch.

    ``nn.Identity`` has an empty state_dict (0 keys).
    ``EmpiricalNormalization`` has ``_mean``, ``_var``, ``_std``, ``count``.

    Loading one into the other produces a confusing PyTorch RuntimeError
    about missing/unexpected keys.  This helper detects the mismatch and
    raises a human-readable error instead.
    """
    live_keys = set(module.state_dict().keys())
    saved_keys = set(saved_state.keys())

    if saved_keys and not live_keys:
        raise RuntimeError(
            f"{label}: checkpoint has a trained EmpiricalNormalization "
            f"(keys: {sorted(saved_keys)}) but the live module is "
            f"nn.Identity (no parameters). Set obs_normalization=True "
            f"in the config to match the checkpoint."
        )
    if live_keys and not saved_keys:
        logger.warning(
            f"{label}: checkpoint has an empty normalizer state (nn.Identity) "
            f"but the live module expects {sorted(live_keys)}. "
            f"Keeping the initialised normalizer — statistics were not saved."
        )
        return

    module.load_state_dict(saved_state)


# ── Frozen normalizer factory ──────────────────────────────────────────


def create_frozen_normalizer(
    mean: torch.Tensor,
    std: torch.Tensor,
    count: int,
    device: torch.device | str,
    eps: float = 1e-2,
) -> EmpiricalNormalization:
    """Create an ``EmpiricalNormalization`` pre-initialised from dataset stats.

    Why a factory instead of using ``EmpiricalNormalization`` directly?
    ----------------------------------------------------------------
    ``EmpiricalNormalization`` is an *online* running-statistics tracker.
    By default it starts with (mean=0, std=1) and drifts its buffers on
    every ``forward()`` call while ``self.training`` is True.  In offline
    RL the dataset is fixed, so the normalizer statistics should be
    computed once from the full dataset and then **frozen**.

    This factory provides a **double safety net** against accidental
    drift:

    1. **``eval()`` mode** — ``forward()`` only calls ``update()`` when
       ``self.training`` is True, so eval mode prevents updates.
    2. **``until=count``** — even if someone accidentally calls
       ``.train()``, the ``update()`` method short-circuits because
       ``self.count >= self.until``.

    Checkpoint compatibility
    -----------------------
    The returned module's ``state_dict()`` has exactly the same keys as
    any ``EmpiricalNormalization``:
    ``{'_mean', '_var', '_std', 'count'}``.
    This means ``save_cql_params`` / ``load_cql_params`` and FastSAC's
    ``save_params`` / ``load`` all handle it identically under the
    ``obs_normalizer_state`` / ``critic_obs_normalizer_state`` keys.

    Parameters
    ----------
    mean:
        Dataset mean of shape ``[1, obs_dim]`` (from
        ``OfflineDataset.compute_obs_statistics``).
    std:
        Dataset standard deviation of shape ``[1, obs_dim]``.
    count:
        Number of transitions the statistics were computed over
        (``OfflineDataset.size``).  Stored in the ``count`` buffer and
        used as the ``until`` cap.
    device:
        Device for the normalizer buffers.
    eps:
        Stability constant for normalisation (denominator is
        ``std + eps``).  Default matches ``EmpiricalNormalization``.

    Returns
    -------
    EmpiricalNormalization
        Ready-to-use normalizer in ``eval()`` mode.

    Example
    -------
    ::

        ds = OfflineDataset("data.h5", device="cuda")
        actor_mean, actor_std = ds.compute_obs_statistics("actor")
        obs_normalizer = create_frozen_normalizer(
            mean=actor_mean,
            std=actor_std,
            count=ds.size,
            device="cuda",
        )
        # normalizer is now frozen — forward() will never update stats
    """
    obs_dim = mean.shape[-1]

    norm = EmpiricalNormalization(
        shape=obs_dim,
        device=device,
        eps=eps,
        until=count,  # safety net: blocks update() even if .train() is called
    )

    # Overwrite default (zeros / ones) with dataset statistics.
    mean_dev = mean.to(device)
    std_dev = std.to(device)

    norm._mean.copy_(mean_dev)
    norm._std.copy_(std_dev)
    norm._var.copy_(std_dev.pow(2))
    norm.count.copy_(torch.tensor(count, dtype=torch.long, device=device))

    norm.eval()  # primary guard: forward() won't call update()
    return norm


# ── Normalisation validation / audit ──────────────────────────────────


@torch.no_grad()
def validate_normalization(
    normalizer: nn.Module,
    raw_data: torch.Tensor,
    label: str = "obs",
    atol_mean: float = 0.15,
    atol_std: float = 0.35,
) -> dict[str, Any]:
    """Compare raw vs. normalised batch statistics for auditing.

    Intended for manual inspection or automated smoke-tests during
    ``setup()`` — call it on a representative slice of the offline
    dataset (e.g. the first 10 000 transitions) to verify that the
    normalizer produces approximately zero-mean, unit-variance output.

    Parameters
    ----------
    normalizer:
        An ``EmpiricalNormalization`` (or ``nn.Identity`` to skip).
    raw_data:
        A batch of **raw** (unnormalized) observations, shape
        ``[N, obs_dim]``.
    label:
        Human-readable name for log messages (e.g. ``"actor_obs"``).
    atol_mean:
        Absolute tolerance for the "mean ≈ 0" check.
    atol_std:
        Absolute tolerance for the "std ≈ 1" check.

    Returns
    -------
    dict with keys:
        ``raw_mean``         — ``[obs_dim]`` mean of raw data.
        ``raw_std``          — ``[obs_dim]`` std of raw data.
        ``norm_mean``        — ``[obs_dim]`` mean of normalised data.
        ``norm_std``         — ``[obs_dim]`` std of normalised data.
        ``mean_close_to_zero`` — bool, ``|norm_mean| < atol_mean`` for
            ≥ 95 % of features.
        ``std_close_to_one`` — bool, ``|norm_std − 1| < atol_std`` for
            ≥ 95 % of non-constant features.
        ``num_const_features`` — int, features with raw std < 1e-5.
        ``report``           — human-readable multi-line string.
    """
    is_identity = isinstance(normalizer, nn.Identity)

    raw_mean = raw_data.mean(dim=0)
    raw_std = raw_data.std(dim=0)

    if is_identity:
        norm_data = normalizer(raw_data)
    else:
        # Force update=False regardless of training mode
        norm_data = normalizer(raw_data, update=False)

    norm_mean = norm_data.mean(dim=0)
    norm_std = norm_data.std(dim=0)

    obs_dim = raw_data.shape[-1]
    const_mask = raw_std < 1e-5
    num_const = int(const_mask.sum().item())
    non_const_mask = ~const_mask

    # Mean check: fraction of features where |norm_mean| < atol_mean
    if is_identity:
        mean_ok = True  # no expectation for identity
        std_ok = True
    else:
        mean_close = norm_mean.abs() < atol_mean
        mean_ok = bool(mean_close.float().mean() >= 0.95)

        # Std check: only on non-constant features
        if non_const_mask.any():
            std_close = (norm_std[non_const_mask] - 1.0).abs() < atol_std
            std_ok = bool(std_close.float().mean() >= 0.95)
        else:
            std_ok = True  # all constant — nothing to check

    # ── Report ────────────────────────────────────────────────────────
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
        lines.append(f"  Std  ≈ 1: {'PASS' if std_ok else 'FAIL'}  (atol={atol_std}, excl. {num_const} const features)")
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


# ── H5 dataset key contract ───────────────────────────────────────────
#
# The H5 file must contain the following top-level datasets.  All arrays
# share the first dimension N (total transitions).  The dataset is
# assumed to have been collected by FastSAC on a task with separate
# actor_obs and critic_obs observation groups.
#
# ┌──────────────────────┬───────────────────┬────────┬────────────────────────┐
# │ H5 key               │ Shape             │ Dtype  │ Semantics              │
# ├──────────────────────┼───────────────────┼────────┼────────────────────────┤
# │ actor_obs            │ [N, actor_obs_dim]│ float32│ Raw (unnormalized)     │
# │ critic_obs           │ [N, crit_obs_dim] │ float32│ Raw (unnormalized)     │
# │ actions              │ [N, act_dim]      │ float32│ Post-scaled (tanh out) │
# │ rewards              │ [N]               │ float32│ Per-step scalar reward │
# │ next_actor_obs       │ [N, actor_obs_dim]│ float32│ True next obs (pre-    │
# │                      │                   │        │ reset for truncated)   │
# │ next_critic_obs      │ [N, crit_obs_dim] │ float32│ Same as above          │
# │ dones                │ [N]               │ int64  │ Terminal OR time-out   │
# │ truncations          │ [N]               │ int64  │ Time-out only          │
# └──────────────────────┴───────────────────┴────────┴────────────────────────┘
#
# Optional metadata attributes on the root group:
#   "task_name"      (str)  — e.g. "g1-29dof-wbt-fast-sac-w-object"
#   "num_envs"       (int)  — number of parallel envs used for collection
#   "total_steps"    (int)  — total training steps at time of dump
#   "actor_obs_keys" (str)  — JSON-encoded list of obs group names
#   "critic_obs_keys"(str)  — JSON-encoded list of obs group names
#
# ═══════════════════════════════════════════════════════════════════════

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

# ── Key aliasing ──────────────────────────────────────────────────────
#
# H5 files produced by different collectors (FastSAC, PPO, external
# pipelines) may use different key names for the same logical dataset.
# This mapping lets OfflineDataset accept common alternatives
# transparently.  Each entry maps a *canonical* CQL key to a tuple of
# aliases to try in priority order.  The first alias found in the H5
# file wins.  If neither the canonical key nor any alias is present,
# the file is rejected with a clear error message.

_H5_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "actor_obs":        ("observations",),
    "critic_obs":       ("critic_observations",),
    "next_actor_obs":   ("next_observations",),
    "next_critic_obs":  ("next_critic_observations",),
    # These are universally named — no aliases needed, but the dict is
    # kept exhaustive so that future formats can be handled in one place.
    "actions":          (),
    "rewards":          (),
    "dones":            (),
    "truncations":      (),
}


def _resolve_h5_keys(h5_keys: set[str]) -> dict[str, str]:
    """Return a mapping ``{canonical_key: actual_h5_key}`` for all required keys.

    For each canonical key, the function first checks if it exists
    literally in *h5_keys*.  If not, it falls back to the aliases
    defined in :data:`_H5_KEY_ALIASES`.  If neither is found the key is
    recorded as missing and a ``KeyError`` is raised at the end with a
    single, aggregated error message.
    """
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
        logger.info(
            "H5 key aliasing applied:\n" + "\n".join(alias_used)
        )

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
    """GPU-resident static dataset for offline RL, loaded from an HDF5 file.

    Loads *all* transitions into GPU tensors at construction time, then
    provides:

    * ``sample(batch_size)`` — uniform random batch returning a
      ``TensorDict`` with the same nested structure as
      ``SimpleReplayBuffer.sample()`` (1-step variant).
    * ``__len__`` / ``__getitem__`` — PyTorch ``Dataset`` interface for
      use with ``DataLoader`` (returns individual transitions).
    * ``compute_obs_statistics()`` — dataset-wide (mean, std) for actor
      and critic observations.

    The dataset stores *post-scaled* actions (same space as the actor's
    tanh-squashed output).  Observations are raw (unnormalized); the
    caller is responsible for applying normalization.
    """

    # ── construction ──────────────────────────────────────────────────

    def __init__(
        self,
        path: str,
        device: torch.device | str = "cpu",
        *,
        expected_actor_obs_dim: int | None = None,
        expected_critic_obs_dim: int | None = None,
        expected_act_dim: int | None = None,
    ):
        """Load an H5 offline dataset into GPU-resident tensors.

        Parameters
        ----------
        path:
            Path to a ``.h5`` file.
        device:
            Target device for all tensors (typically ``"cuda:0"``).
        expected_actor_obs_dim:
            If provided, assert that the actor obs dimension matches.
        expected_critic_obs_dim:
            If provided, assert that the critic obs dimension matches.
        expected_act_dim:
            If provided, assert that the action dimension matches.
        """
        super().__init__()

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Offline dataset not found: {path}")

        logger.info(f"Loading offline dataset from {path} ...")

        # ── Read H5 ──────────────────────────────────────────────────
        with h5py.File(path, "r") as f:
            # Resolve key names (canonical ↔ alias)
            key_map = _resolve_h5_keys(set(f.keys()))

            # Read metadata attributes (optional)
            self.metadata: dict[str, Any] = {}
            for attr in ("task_name", "num_envs", "total_steps",
                         "actor_obs_keys", "critic_obs_keys"):
                if attr in f.attrs:
                    self.metadata[attr] = f.attrs[attr]

            # Load into numpy first (h5py reads are fastest to numpy)
            # key_map maps canonical → actual H5 key name
            raw: dict[str, np.ndarray] = {}
            for canonical in _REQUIRED_H5_KEYS:
                raw[canonical] = f[key_map[canonical]][()]  # type: ignore[index]

        # ── Validate shapes ──────────────────────────────────────────
        n = raw["actor_obs"].shape[0]
        for key in _REQUIRED_H5_KEYS:
            if raw[key].shape[0] != n:
                raise ValueError(
                    f"Length mismatch: '{key}' has {raw[key].shape[0]} rows "
                    f"but 'actor_obs' has {n}. All datasets must share the "
                    f"first dimension."
                )

        # Validate dimensionality
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

        # Validate observation consistency
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

        # Optional expected-dim assertions
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

        # Validate dtypes (warn, don't crash — we cast below)
        for key in _REQUIRED_H5_KEYS:
            expected_dt = _EXPECTED_DTYPES[key]
            actual_dt = str(raw[key].dtype)
            if actual_dt != expected_dt:
                logger.warning(
                    f"Dataset key '{key}' has dtype {actual_dt}, "
                    f"expected {expected_dt}. Will cast."
                )

        # ── Validate logical invariants ──────────────────────────────
        # truncation ⊂ dones  (every truncation must also be a done)
        trunc_np = raw["truncations"].astype(np.int64)
        dones_np = raw["dones"].astype(np.int64)
        invalid_trunc = int(np.sum((trunc_np == 1) & (dones_np == 0)))
        if invalid_trunc > 0:
            logger.warning(
                f"Found {invalid_trunc} transitions where truncation=1 but "
                f"done=0. This violates the invariant truncation ⊂ dones. "
                f"These may cause incorrect TD-target bootstrapping."
            )

        # ── Convert to GPU tensors ───────────────────────────────────
        self.actor_obs = torch.from_numpy(
            raw["actor_obs"].astype(np.float32)
        ).to(device)
        self.critic_obs = torch.from_numpy(
            raw["critic_obs"].astype(np.float32)
        ).to(device)
        self.actions = torch.from_numpy(
            raw["actions"].astype(np.float32)
        ).to(device)
        self.rewards = torch.from_numpy(
            raw["rewards"].astype(np.float32)
        ).to(device)
        self.next_actor_obs = torch.from_numpy(
            raw["next_actor_obs"].astype(np.float32)
        ).to(device)
        self.next_critic_obs = torch.from_numpy(
            raw["next_critic_obs"].astype(np.float32)
        ).to(device)
        self.dones = torch.from_numpy(
            raw["dones"].astype(np.int64)
        ).to(device)
        self.truncations = torch.from_numpy(
            raw["truncations"].astype(np.int64)
        ).to(device)

        # ── Store dimensions ─────────────────────────────────────────
        self.size: int = n
        self.actor_obs_dim: int = actor_obs_dim
        self.critic_obs_dim: int = critic_obs_dim
        self.act_dim: int = act_dim

        # ── Pre-compute normalization statistics (cached) ────────────
        self._actor_obs_stats: tuple[torch.Tensor, torch.Tensor] | None = None
        self._critic_obs_stats: tuple[torch.Tensor, torch.Tensor] | None = None

        logger.info(
            f"Loaded {self.size:,} transitions  "
            f"(actor_obs={actor_obs_dim}, critic_obs={critic_obs_dim}, "
            f"act={act_dim})"
        )

    # ── PyTorch Dataset interface ─────────────────────────────────────

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single transition as a flat dict (for DataLoader use)."""
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

    # ── Uniform random batch (replay-buffer compatible) ───────────────

    @torch.no_grad()
    def sample(self, batch_size: int) -> TensorDict:
        """Uniformly sample a batch of transitions.

        Returns a ``TensorDict`` with the **same nested structure** as
        ``SimpleReplayBuffer.sample()`` (1-step variant)::

            TensorDict({
                "observations":       [B, actor_obs_dim],
                "actions":            [B, act_dim],
                "critic_observations":[B, critic_obs_dim],
                "next": {
                    "observations":       [B, actor_obs_dim],
                    "rewards":            [B],
                    "dones":              [B],
                    "truncations":        [B],
                    "effective_n_steps":  [B],   # always 1 for offline data
                    "critic_observations":[B, critic_obs_dim],
                },
            }, batch_size=B)

        Key name mapping (H5 → TensorDict):
            actor_obs       → "observations"
            critic_obs      → "critic_observations"
            next_actor_obs  → "next"/"observations"
            next_critic_obs → "next"/"critic_observations"

        This matches the SimpleReplayBuffer contract where
        "observations" always means the actor observation.
        """
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

    # ── Normalization statistics ──────────────────────────────────────

    @torch.no_grad()
    def compute_obs_statistics(
        self,
        obs_type: str = "actor",
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (mean, std) over all observations in the dataset.

        Parameters
        ----------
        obs_type:
            ``"actor"`` or ``"critic"`` — which observation tensor to use.
        eps:
            Minimum standard deviation (avoids division by zero).

        Returns
        -------
        (mean, std) : tuple[Tensor, Tensor]
            Both of shape ``[1, obs_dim]``, matching the layout expected
            by ``EmpiricalNormalization._mean`` / ``_std``.
        """
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

        mean = data.mean(dim=0, keepdim=True)        # [1, obs_dim]
        std = data.std(dim=0, keepdim=True).clamp(min=eps)  # [1, obs_dim]

        if obs_type == "actor":
            self._actor_obs_stats = (mean, std)
        else:
            self._critic_obs_stats = (mean, std)

        return mean, std

    # ── Summary / validation helpers ──────────────────────────────────

    def summary(self) -> str:
        """Return a concise human-readable report of the dataset."""
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

        # Per-field statistics
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

        # Episode statistics
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
            # Approximate mean episode length
            mean_ep_len = self.size / n_dones
            lines.append(f"  Approx mean episode length : {mean_ep_len:>8.1f}")

        # Metadata
        if self.metadata:
            lines.append("──────────────────────────────────────────────────────────")
            lines.append("  Metadata:")
            for k, v in self.metadata.items():
                lines.append(f"    {k}: {v}")

        lines.append("╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Log the dataset summary report."""
        logger.info("\n" + self.summary())


def validate_dataset_dry_run(
    path: str,
    device: torch.device | str = "cpu",
    sample_batch_size: int = 64,
) -> OfflineDataset:
    """Load a dataset, print a full summary, and run basic sanity checks.

    This is intended as a standalone diagnostic — call it from a script or
    notebook to verify a dataset file before training.

    Parameters
    ----------
    path:
        Path to the ``.h5`` file.
    device:
        Device for tensors (use ``"cpu"`` for validation to avoid GPU OOM).
    sample_batch_size:
        Size of a test batch to draw for shape verification.

    Returns
    -------
    OfflineDataset
        The loaded dataset (so callers can inspect further).
    """
    logger.info(f"=== Offline Dataset Dry-Run Validation ===")
    logger.info(f"File: {path}")

    ds = OfflineDataset(path, device=device)
    ds.print_summary()

    # ── Shape checks ──────────────────────────────────────────────────
    logger.info("Running shape / semantics checks ...")

    assert ds.actor_obs.shape == (ds.size, ds.actor_obs_dim), \
        f"actor_obs shape {ds.actor_obs.shape} != ({ds.size}, {ds.actor_obs_dim})"
    assert ds.critic_obs.shape == (ds.size, ds.critic_obs_dim), \
        f"critic_obs shape {ds.critic_obs.shape} != ({ds.size}, {ds.critic_obs_dim})"
    assert ds.next_actor_obs.shape == ds.actor_obs.shape, \
        f"next_actor_obs shape {ds.next_actor_obs.shape} != actor_obs {ds.actor_obs.shape}"
    assert ds.next_critic_obs.shape == ds.critic_obs.shape, \
        f"next_critic_obs shape {ds.next_critic_obs.shape} != critic_obs {ds.critic_obs.shape}"
    assert ds.actions.shape == (ds.size, ds.act_dim)
    assert ds.rewards.shape == (ds.size,)
    assert ds.dones.shape == (ds.size,)
    assert ds.truncations.shape == (ds.size,)

    # ── Value range checks ────────────────────────────────────────────
    # Actions should be in [-1, 1] if post-tanh (but action_scale widens)
    act_min = ds.actions.min().item()
    act_max = ds.actions.max().item()
    if act_min < -10.0 or act_max > 10.0:
        logger.warning(
            f"Action range [{act_min:.3f}, {act_max:.3f}] is very wide. "
            f"Verify these are truly post-scaled actor outputs."
        )
    else:
        logger.info(f"  Actions range: [{act_min:.4f}, {act_max:.4f}]  ✓")

    # Dones and truncations should be binary
    dones_unique = ds.dones.unique().tolist()
    trunc_unique = ds.truncations.unique().tolist()
    assert set(dones_unique).issubset({0, 1}), \
        f"dones has non-binary values: {dones_unique}"
    assert set(trunc_unique).issubset({0, 1}), \
        f"truncations has non-binary values: {trunc_unique}"
    logger.info(f"  dones values: {dones_unique}  ✓")
    logger.info(f"  truncations values: {trunc_unique}  ✓")

    # ── Normalization statistics ──────────────────────────────────────
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

    # Check for zero-variance features (constant columns)
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

    # ── Test batch sample ─────────────────────────────────────────────
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

    # ── DataLoader interface test ─────────────────────────────────────
    item = ds[0]
    logger.info(f"  __getitem__(0) keys: {sorted(item.keys())}")

    logger.info("=== Dry-run validation PASSED ===")
    return ds


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint schema
# ═══════════════════════════════════════════════════════════════════════
#
# Design goal: a CQL checkpoint must be loadable by ``eval_agent.py``
# through the same ``experiment_config → algo._target_ → class resolution
# → setup() → load()`` flow that FastSAC uses.  Actor-only consumers
# (ONNX export, inference) must work identically.
#
# ┌─────────────────────────────────┬──────────┬────────────────────────┐
# │ Key                             │ Origin   │ Notes                  │
# ├─────────────────────────────────┼──────────┼────────────────────────┤
# │ actor_state_dict                │ FastSAC  │ IDENTICAL — same Actor │
# │                                 │          │ class, same buffers    │
# │                                 │          │ (action_scale, etc.)   │
# │ qnet_state_dict                 │ FastSAC  │ SAME KEY, different    │
# │                                 │          │ tensor shapes: scalar  │
# │                                 │          │ Q final layer is [H/4, │
# │                                 │          │ 1] not [H/4, atoms].   │
# │ qnet_target_state_dict          │ FastSAC  │ Same as qnet.          │
# │ log_alpha                       │ FastSAC  │ IDENTICAL — SAC entropy│
# │                                 │          │ temperature scalar.    │
# │ obs_normalizer_state            │ FastSAC  │ IDENTICAL —            │
# │                                 │          │ EmpiricalNormalization  │
# │                                 │          │ state dict (_mean,     │
# │                                 │          │ _var, _std, count).    │
# │ critic_obs_normalizer_state     │ FastSAC  │ IDENTICAL.             │
# │ actor_optimizer_state_dict      │ FastSAC  │ IDENTICAL — same param │
# │                                 │          │ groups.                │
# │ q_optimizer_state_dict          │ FastSAC  │ SAME KEY, different    │
# │                                 │          │ shapes (scalar critic).│
# │ alpha_optimizer_state_dict      │ FastSAC  │ IDENTICAL.             │
# │ grad_scaler_state_dict          │ FastSAC  │ IDENTICAL.             │
# │ args                            │ FastSAC  │ IDENTICAL — dict of    │
# │                                 │          │ config fields. Will    │
# │                                 │          │ contain CQL-specific   │
# │                                 │          │ fields not present in  │
# │                                 │          │ FastSAC checkpoints.   │
# │ global_step                     │ FastSAC  │ IDENTICAL.             │
# │ env_state                       │ FastSAC  │ IDENTICAL — optional,  │
# │                                 │          │ curriculum state.      │
# │ experiment_config               │ FastSAC  │ IDENTICAL — written by │
# │                                 │          │ BaseAlgo._checkpoint_  │
# │                                 │          │ metadata().            │
# │ wandb_run_path                  │ FastSAC  │ IDENTICAL — optional.  │
# │ iteration                       │ FastSAC  │ IDENTICAL.             │
# ├─────────────────────────────────┼──────────┼────────────────────────┤
# │ log_alpha_cql                   │ CQL-ONLY │ Conservative penalty   │
# │                                 │          │ temperature. Absent in │
# │                                 │          │ FastSAC checkpoints.   │
# │ alpha_cql_optimizer_state_dict  │ CQL-ONLY │ Optimizer state for    │
# │                                 │          │ Lagrangian α_cql.      │
# │                                 │          │ Absent in FastSAC.     │
# └─────────────────────────────────┴──────────┴────────────────────────┘
#
# Backward compatibility (loading a FastSAC checkpoint into CQL):
#   NOT supported for full resume — qnet shapes differ.  However,
#   ``actor_state_dict`` and ``obs_normalizer_state`` CAN be loaded
#   to warm-start the CQL actor from a FastSAC policy.  The load()
#   method provides ``actor_only=True`` for this purpose.
#
# Forward compatibility (loading a CQL checkpoint into FastSAC):
#   NOT supported — FastSAC's distributional Critic cannot consume
#   scalar-Q state dicts.  Actor-only loading into any consumer that
#   uses the same Actor class will work because ``actor_state_dict``
#   and ``obs_normalizer_state`` are identical.
#
# ═══════════════════════════════════════════════════════════════════════


def save_cql_params(
    global_step: int,
    actor: nn.Module,
    qnet: nn.Module,
    qnet_target: nn.Module,
    log_alpha: torch.Tensor,
    obs_normalizer: nn.Module,
    critic_obs_normalizer: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    alpha_optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: Any,  # TODO: replace with OfflineCQLConfig once the dataclass is added
    save_path: str,
    save_fn: Any = torch.save,
    metadata: dict[str, Any] | None = None,
    env_state: dict[str, torch.Tensor | float] | None = None,
    log_alpha_cql: torch.Tensor | None = None,
    alpha_cql_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Save CQL training state to disk.

    The checkpoint dict is a strict superset of the one produced by
    ``fast_sac_utils.save_params``.  Every key that FastSAC writes is
    present here under the **exact same name** so that actor-only
    consumers (ONNX export, inference helpers, ``eval_agent.py``)
    work without any special-casing.

    CQL-only additions (2 keys):

    * ``log_alpha_cql`` — conservative-penalty Lagrange multiplier.
    * ``alpha_cql_optimizer_state_dict`` — its optimiser state.

    These are absent from FastSAC checkpoints; ``load_cql_params``
    handles their absence gracefully when warm-starting from FastSAC.
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    save_dict: dict[str, Any] = {
        # ── Checkpoint type marker ────────────────────────────────────
        "algo_type": "offline_cql",
        # ── Keys identical to FastSAC (same name, same semantics) ─────
        "actor_state_dict": cpu_state(actor.state_dict()),
        "qnet_state_dict": cpu_state(qnet.state_dict()),
        "qnet_target_state_dict": cpu_state(qnet_target.state_dict()),
        "log_alpha": log_alpha.detach().cpu(),
        "obs_normalizer_state": (
            cpu_state(obs_normalizer.state_dict()) if hasattr(obs_normalizer, "state_dict") else None
        ),
        "critic_obs_normalizer_state": (
            cpu_state(critic_obs_normalizer.state_dict()) if hasattr(critic_obs_normalizer, "state_dict") else None
        ),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "alpha_optimizer_state_dict": alpha_optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": vars(args) if hasattr(args, "__dict__") else dict(args),
        "global_step": global_step,
        # ── CQL-only keys ─────────────────────────────────────────────
        "log_alpha_cql": log_alpha_cql.detach().cpu() if log_alpha_cql is not None else None,
        "alpha_cql_optimizer_state_dict": (
            alpha_cql_optimizer.state_dict() if alpha_cql_optimizer is not None else None
        ),
    }
    # env_state is optional — identical to FastSAC's handling
    if env_state:
        save_dict["env_state"] = env_state

    if metadata is None:
        raise ValueError("Checkpoint metadata is required when saving CQL parameters.")
    # metadata supplies: experiment_config, wandb_run_path, iteration
    save_dict.update(metadata)
    save_fn(save_dict, save_path)
    logger.info(f"Saved CQL parameters to {save_path}")


def load_cql_params(
    ckpt_path: str,
    device: torch.device | str,
    *,
    actor: nn.Module,
    qnet: nn.Module,
    qnet_target: nn.Module,
    log_alpha: torch.Tensor,
    obs_normalizer: nn.Module,
    critic_obs_normalizer: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    alpha_optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    log_alpha_cql: torch.Tensor | None = None,
    alpha_cql_optimizer: torch.optim.Optimizer | None = None,
    actor_only: bool = False,
) -> dict[str, Any]:
    """Load a CQL (or FastSAC actor-only) checkpoint into live objects.

    Parameters
    ----------
    ckpt_path:
        Path to the ``.pt`` checkpoint file.
    device:
        Device to map tensors onto.
    actor_only:
        If ``True``, load **only** ``actor_state_dict`` and
        ``obs_normalizer_state``.  This enables warm-starting a CQL
        actor from a FastSAC checkpoint without touching the critic
        (whose shapes are incompatible).
    *remaining args*:
        Live module / optimizer / tensor references whose state will
        be overwritten in-place.

    Returns
    -------
    dict[str, Any]
        The raw checkpoint dict, so callers can inspect ``global_step``,
        ``env_state``, ``args``, etc.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── Checkpoint type detection ─────────────────────────────────────
    # CQL checkpoints include an 'algo_type' marker.  FastSAC checkpoints
    # lack it.  When doing a full load (actor_only=False) from a non-CQL
    # checkpoint, warn early — the critic shapes will mismatch and the
    # load_state_dict call below will fail with a RuntimeError anyway,
    # but this log line explains *why*.
    ckpt_algo_type = ckpt.get("algo_type")
    if not actor_only and ckpt_algo_type is not None and ckpt_algo_type != "offline_cql":
        logger.warning(
            f"Checkpoint was saved by algo_type='{ckpt_algo_type}', "
            f"but this is an OfflineCQL load with actor_only=False. "
            f"Critic shapes will almost certainly mismatch."
        )
    if not actor_only and ckpt_algo_type is None:
        logger.warning(
            "Checkpoint has no 'algo_type' marker — it was likely saved by "
            "FastSAC.  Full CQL resume requires a CQL checkpoint; critic "
            "shapes differ.  Use actor_only=True for warm-starting."
        )

    # ── Actor (always loaded — identical between FastSAC and CQL) ─────
    actor.load_state_dict(ckpt["actor_state_dict"])

    # Normalizer guard: detect type mismatch between checkpoint and live module.
    # nn.Identity has an empty state_dict; EmpiricalNormalization has _mean, _var, etc.
    # Loading one into the other produces a confusing RuntimeError, so we catch it early.
    saved_obs_norm = ckpt.get("obs_normalizer_state")
    if saved_obs_norm is not None:
        _load_normalizer_safe(obs_normalizer, saved_obs_norm, "obs_normalizer")

    if actor_only:
        logger.info(f"Loaded actor-only state from {ckpt_path}")
        return ckpt

    # ── Critic & target (shapes must match — CQL↔CQL only) ───────────
    try:
        qnet.load_state_dict(ckpt["qnet_state_dict"])
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load qnet_state_dict — this usually means the checkpoint "
            f"was saved by a different algo (e.g. FastSAC distributional critic "
            f"vs CQL scalar critic).  Use actor_only=True to warm-start "
            f"only the actor from a cross-algo checkpoint.\n"
            f"Original error: {e}"
        ) from e
    try:
        qnet_target.load_state_dict(ckpt["qnet_target_state_dict"])
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load qnet_target_state_dict — likely a cross-algo "
            f"checkpoint (see qnet error above for details).\n"
            f"Original error: {e}"
        ) from e
    saved_critic_obs_norm = ckpt.get("critic_obs_normalizer_state")
    if saved_critic_obs_norm is not None:
        _load_normalizer_safe(critic_obs_normalizer, saved_critic_obs_norm, "critic_obs_normalizer")

    # ── Entropy temperature ───────────────────────────────────────────
    log_alpha.data.copy_(ckpt["log_alpha"].to(device))

    # ── Optimiser states ──────────────────────────────────────────────
    actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
    q_optimizer.load_state_dict(ckpt["q_optimizer_state_dict"])
    alpha_optimizer.load_state_dict(ckpt["alpha_optimizer_state_dict"])

    if ckpt.get("grad_scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["grad_scaler_state_dict"])

    # ── CQL-specific (gracefully absent in FastSAC checkpoints) ───────
    if log_alpha_cql is not None and ckpt.get("log_alpha_cql") is not None:
        log_alpha_cql.data.copy_(ckpt["log_alpha_cql"].to(device))
    elif log_alpha_cql is not None:
        logger.warning("Checkpoint has no log_alpha_cql — keeping initialised value.")

    if alpha_cql_optimizer is not None and ckpt.get("alpha_cql_optimizer_state_dict") is not None:
        alpha_cql_optimizer.load_state_dict(ckpt["alpha_cql_optimizer_state_dict"])
    elif alpha_cql_optimizer is not None:
        logger.warning("Checkpoint has no alpha_cql_optimizer — keeping initialised state.")

    logger.info(f"Loaded full CQL state from {ckpt_path} (global_step={ckpt.get('global_step', '?')})")
    return ckpt
