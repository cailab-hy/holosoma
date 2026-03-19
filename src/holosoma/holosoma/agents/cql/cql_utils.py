from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.amp import GradScaler

from holosoma.config_types.algo import CQLConfig


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, device, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, center: bool = True, update: bool = True) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}")

        if self.training and update:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        if self.until is not None and self.count >= self.until:
            return

        if dist.is_available() and dist.is_initialized():
            local_batch_size = x.shape[0]
            world_size = dist.get_world_size()
            global_batch_size = world_size * local_batch_size

            x_shifted = x - self._mean
            local_sum_shifted = torch.sum(x_shifted, dim=0, keepdim=True)
            local_sum_sq_shifted = torch.sum(x_shifted.pow(2), dim=0, keepdim=True)

            stats_to_sync = torch.cat([local_sum_shifted, local_sum_sq_shifted], dim=0)
            dist.all_reduce(stats_to_sync, op=dist.ReduceOp.SUM)
            global_sum_shifted, global_sum_sq_shifted = stats_to_sync

            batch_mean_shifted = global_sum_shifted / global_batch_size
            batch_var = global_sum_sq_shifted / global_batch_size - batch_mean_shifted.pow(2)
            batch_mean = batch_mean_shifted + self._mean
        else:
            global_batch_size = x.shape[0]
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)

        new_count = self.count + global_batch_size

        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (global_batch_size / new_count))

        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * global_batch_size
        M2 = m_a + m_b + delta2.pow(2) * (self.count * global_batch_size / new_count)
        self._var.copy_(M2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


def cpu_state(sd):
    return {k: v.detach().to("cpu", non_blocking=True) for k, v in sd.items()}


def save_params(
    global_step: int,
    actor: nn.Module,
    qnet: nn.Module,
    qnet_target: nn.Module,
    obs_normalizer: nn.Module,
    critic_obs_normalizer: nn.Module,
    log_alpha: torch.Tensor,
    actor_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    alpha_optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: CQLConfig,
    save_path: str,
    save_fn=torch.save,
    metadata: dict[str, Any] | None = None,
    env_state: dict[str, torch.Tensor | float] | None = None,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {
        "actor_state_dict": cpu_state(actor.state_dict()),
        "qnet_state_dict": cpu_state(qnet.state_dict()),
        "qnet_target_state_dict": cpu_state(qnet_target.state_dict()),
        "obs_normalizer_state": (
            cpu_state(obs_normalizer.state_dict()) if hasattr(obs_normalizer, "state_dict") else None
        ),
        "critic_obs_normalizer_state": (
            cpu_state(critic_obs_normalizer.state_dict()) if hasattr(critic_obs_normalizer, "state_dict") else None
        ),
        "log_alpha": log_alpha.detach().cpu(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "alpha_optimizer_state_dict": alpha_optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "global_step": global_step,
    }
    if env_state:
        save_dict["env_state"] = env_state
    if metadata is None:
        raise ValueError("Checkpoint metadata is required when saving CQL parameters.")
    save_dict.update(metadata)
    save_fn(save_dict, save_path)
    print(f"Saved parameters and configuration to {save_path}")
