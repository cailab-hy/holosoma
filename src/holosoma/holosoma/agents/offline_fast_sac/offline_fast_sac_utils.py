from __future__ import annotations

import h5py
import torch
from loguru import logger
from tensordict import TensorDict

from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization

# Required dataset keys and their minimum rank (excluding batch axis).
_REQUIRED_KEYS = {
    "observations": 1,
    "critic_observations": 1,
    "actions": 1,
    "rewards": 0,
    "next_observations": 1,
    "next_critic_observations": 1,
    "dones": 0,
    "truncations": 0,
}


def validate_offline_dataset(path: str) -> dict[str, tuple[int, ...]]:
    """Open *path* and verify that it contains all required keys with consistent N.

    Returns a dict mapping each key to its shape (for informational purposes).
    Raises ``ValueError`` on any mismatch.
    """
    with h5py.File(path, "r") as f:
        shapes: dict[str, tuple[int, ...]] = {}
        for key, min_extra_dims in _REQUIRED_KEYS.items():
            if key not in f:
                raise ValueError(f"Offline dataset '{path}' is missing required key '{key}'")
            shape = f[key].shape
            if len(shape) < 1 + min_extra_dims:
                raise ValueError(
                    f"Key '{key}' has shape {shape} but expected at least {1 + min_extra_dims} dimensions"
                )
            shapes[key] = shape

        # All keys must share the same leading dimension N.
        n_values = {k: s[0] for k, s in shapes.items()}
        unique_ns = set(n_values.values())
        if len(unique_ns) != 1:
            raise ValueError(f"Inconsistent dataset sizes across keys: {n_values}")

        n = unique_ns.pop()
        if n == 0:
            raise ValueError("Dataset is empty (N=0)")

        logger.info(f"Offline dataset validated: N={n}, path='{path}'")
        for key, shape in shapes.items():
            logger.info(f"  {key}: {shape}")

    return shapes


class OfflineReplayBuffer:
    """A simple offline replay buffer that loads an HDF5 dataset into GPU memory.

    Designed as a drop-in replacement for ``SimpleReplayBuffer.sample()`` in the
    offline training loop.  The returned ``TensorDict`` has the same structure
    that ``_update_main`` / ``_update_pol`` expect.
    """

    def __init__(self, dataset_path: str, device: torch.device | str = "cuda"):
        shapes = validate_offline_dataset(dataset_path)

        with h5py.File(dataset_path, "r") as f:
            self.observations = torch.as_tensor(f["observations"][:], dtype=torch.float, device=device)
            self.critic_observations = torch.as_tensor(
                f["critic_observations"][:], dtype=torch.float, device=device
            )
            self.actions = torch.as_tensor(f["actions"][:], dtype=torch.float, device=device)
            self.next_observations = torch.as_tensor(f["next_observations"][:], dtype=torch.float, device=device)
            self.next_critic_observations = torch.as_tensor(
                f["next_critic_observations"][:], dtype=torch.float, device=device
            )

            # Squeeze trailing dim if present: [N, 1] -> [N]
            rewards_raw = torch.as_tensor(f["rewards"][:], dtype=torch.float, device=device)
            dones_raw = torch.as_tensor(f["dones"][:], dtype=torch.long, device=device)
            truncations_raw = torch.as_tensor(f["truncations"][:], dtype=torch.long, device=device)

        self.rewards = rewards_raw.squeeze(-1) if rewards_raw.ndim > 1 else rewards_raw
        self.dones = dones_raw.squeeze(-1) if dones_raw.ndim > 1 else dones_raw
        self.truncations = truncations_raw.squeeze(-1) if truncations_raw.ndim > 1 else truncations_raw

        self.size = self.observations.shape[0]
        self.n_obs = self.observations.shape[1]
        self.n_critic_obs = self.critic_observations.shape[1]
        self.n_act = self.actions.shape[1]
        self.device = device

        # Validate dimensional consistency across related keys.
        assert self.next_observations.shape == self.observations.shape, (
            f"next_observations shape {self.next_observations.shape} != observations {self.observations.shape}"
        )
        assert self.next_critic_observations.shape == self.critic_observations.shape, (
            f"next_critic_observations shape {self.next_critic_observations.shape} "
            f"!= critic_observations {self.critic_observations.shape}"
        )
        assert self.rewards.shape == (self.size,), f"rewards shape {self.rewards.shape} != ({self.size},)"
        assert self.dones.shape == (self.size,), f"dones shape {self.dones.shape} != ({self.size},)"
        assert self.truncations.shape == (self.size,), (
            f"truncations shape {self.truncations.shape} != ({self.size},)"
        )

        logger.info(
            f"OfflineReplayBuffer loaded: {self.size} transitions, "
            f"obs={self.n_obs}, critic_obs={self.n_critic_obs}, act={self.n_act}"
        )

    def __len__(self) -> int:
        return self.size

    @torch.no_grad()
    def sample(self, batch_size: int) -> TensorDict:
        """Sample a batch of transitions uniformly at random.

        Returns a ``TensorDict`` with the same nested structure that
        ``FastSACAgent._update_main`` and ``_update_pol`` expect.
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        out = TensorDict(
            {
                "observations": self.observations[indices],
                "actions": self.actions[indices],
                "next": {
                    "rewards": self.rewards[indices],
                    "dones": self.dones[indices],
                    "truncations": self.truncations[indices],
                    "observations": self.next_observations[indices],
                    "effective_n_steps": torch.ones(batch_size, device=self.device, dtype=torch.long),
                },
            },
            batch_size=batch_size,
        )
        out["critic_observations"] = self.critic_observations[indices]
        out["next"]["critic_observations"] = self.next_critic_observations[indices]
        return out

    def compute_obs_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std of ``observations`` over the full dataset.

        Returns (mean, std) each of shape ``(n_obs,)``.
        """
        mean = self.observations.mean(dim=0)
        std = self.observations.std(dim=0, correction=0)
        return mean, std

    def compute_critic_obs_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std of ``critic_observations`` over the full dataset.

        Returns (mean, std) each of shape ``(n_critic_obs,)``.
        """
        mean = self.critic_observations.mean(dim=0)
        std = self.critic_observations.std(dim=0, correction=0)
        return mean, std


def init_normalizer_from_dataset(
    normalizer: EmpiricalNormalization,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    """Set an ``EmpiricalNormalization`` module's buffers from precomputed statistics
    and freeze it by switching to eval mode.

    This avoids running the online Welford update on replay data, which would
    bias the running statistics toward the offline distribution.
    """
    normalizer._mean.copy_(mean.unsqueeze(0))
    var = std.pow(2)
    normalizer._var.copy_(var.unsqueeze(0))
    normalizer._std.copy_(std.unsqueeze(0))
    # Set count to a large value so that any accidental update() call is a no-op
    # (when ``until`` is set) and to signal that statistics are initialized.
    normalizer.count.fill_(mean.shape[0])
    normalizer.eval()
