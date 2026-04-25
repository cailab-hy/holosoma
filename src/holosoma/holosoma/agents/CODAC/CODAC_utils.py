from __future__ import annotations

import os
from typing import Any

from torch import nn
from torch.amp import GradScaler

from holosoma.agents.offline_sac.offline_sac_utils import EmpiricalNormalization, cpu_state
from holosoma.config_types.algo import CODACConfig
from holosoma.utils.safe_torch_import import torch


def save_params(
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
    args: CODACConfig,
    save_path: str,
    save_fn=torch.save,
    metadata: dict[str, Any] | None = None,
    env_state: dict[str, torch.Tensor | float] | None = None,
    extra_state: dict[str, Any] | None = None,
):
    """Save CODAC parameters and training configuration to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {
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
        "args": vars(args),
        "global_step": global_step,
    }
    if env_state:
        save_dict["env_state"] = env_state
    if extra_state:
        save_dict.update(extra_state)
    if metadata is None:
        raise ValueError("Checkpoint metadata is required when saving CODAC parameters.")
    save_dict.update(metadata)
    save_fn(save_dict, save_path)
    print(f"Saved parameters and configuration to {save_path}")


__all__ = ["EmpiricalNormalization", "save_params"]
