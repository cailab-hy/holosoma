"""Checkpoint I/O — canonical implementation for offline-RL common.

The legacy ``offline_cql`` package has been removed, but the checkpoint key
schema is unchanged. Old checkpoint/config target metadata is handled by
``offline_rl.common.target_compat`` rather than by importing legacy modules.
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger
from torch import nn

from holosoma.agents.fast_sac.fast_sac_utils import cpu_state
from holosoma.utils.safe_torch_import import GradScaler, torch


def _load_normalizer_safe(
    module: nn.Module,
    saved_state: dict[str, Any],
    label: str,
) -> None:
    """Load a normalizer state dict with a clear error on type mismatch."""
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


def save_offline_rl_params(
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
    args: Any,
    save_path: str,
    save_fn: Any = torch.save,
    metadata: dict[str, Any] | None = None,
    env_state: dict[str, torch.Tensor | float] | None = None,
    log_alpha_cql: torch.Tensor | None = None,
    alpha_cql_optimizer: torch.optim.Optimizer | None = None,
    value_net: nn.Module | None = None,
    value_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Save CQL/offline-RL training state to disk.

    Schema and key names are intentionally unchanged from the legacy helper.
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    save_dict: dict[str, Any] = {
        "algo_type": "offline_cql",
        "actor_state_dict": cpu_state(actor.state_dict()),
        "qnet_state_dict": cpu_state(qnet.state_dict()),
        "qnet_target_state_dict": cpu_state(qnet_target.state_dict()),
        "log_alpha": log_alpha.detach().cpu(),
        "obs_normalizer_state": (
            cpu_state(obs_normalizer.state_dict())
            if hasattr(obs_normalizer, "state_dict")
            else None
        ),
        "critic_obs_normalizer_state": (
            cpu_state(critic_obs_normalizer.state_dict())
            if hasattr(critic_obs_normalizer, "state_dict")
            else None
        ),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "alpha_optimizer_state_dict": alpha_optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": vars(args) if hasattr(args, "__dict__") else dict(args),
        "global_step": global_step,
        "log_alpha_cql": (
            log_alpha_cql.detach().cpu() if log_alpha_cql is not None else None
        ),
        "alpha_cql_optimizer_state_dict": (
            alpha_cql_optimizer.state_dict() if alpha_cql_optimizer is not None else None
        ),
        "value_net_state_dict": (
            cpu_state(value_net.state_dict()) if value_net is not None else None
        ),
        "value_optimizer_state_dict": (
            value_optimizer.state_dict() if value_optimizer is not None else None
        ),
    }
    if env_state:
        save_dict["env_state"] = env_state

    if metadata is None:
        raise ValueError("Checkpoint metadata is required when saving CQL parameters.")
    save_dict.update(metadata)
    save_fn(save_dict, save_path)
    logger.info(f"Saved CQL parameters to {save_path}")


def load_offline_rl_params(
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
    value_net: nn.Module | None = None,
    value_optimizer: torch.optim.Optimizer | None = None,
    actor_only: bool = False,
) -> dict[str, Any]:
    """Load a CQL/offline-RL checkpoint into live objects."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

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

    actor.load_state_dict(ckpt["actor_state_dict"])

    saved_obs_norm = ckpt.get("obs_normalizer_state")
    if saved_obs_norm is not None:
        _load_normalizer_safe(obs_normalizer, saved_obs_norm, "obs_normalizer")

    if actor_only:
        logger.info(f"Loaded actor-only state from {ckpt_path}")
        return ckpt

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
        _load_normalizer_safe(
            critic_obs_normalizer,
            saved_critic_obs_norm,
            "critic_obs_normalizer",
        )

    log_alpha.data.copy_(ckpt["log_alpha"].to(device))

    actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
    q_optimizer.load_state_dict(ckpt["q_optimizer_state_dict"])
    alpha_optimizer.load_state_dict(ckpt["alpha_optimizer_state_dict"])

    if ckpt.get("grad_scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["grad_scaler_state_dict"])

    if log_alpha_cql is not None and ckpt.get("log_alpha_cql") is not None:
        log_alpha_cql.data.copy_(ckpt["log_alpha_cql"].to(device))
    elif log_alpha_cql is not None:
        logger.warning(
            "Checkpoint has no log_alpha_cql — keeping initialised value."
        )

    if (
        alpha_cql_optimizer is not None
        and ckpt.get("alpha_cql_optimizer_state_dict") is not None
    ):
        alpha_cql_optimizer.load_state_dict(ckpt["alpha_cql_optimizer_state_dict"])
    elif alpha_cql_optimizer is not None:
        logger.warning(
            "Checkpoint has no alpha_cql_optimizer — keeping initialised state."
        )

    if value_net is not None and ckpt.get("value_net_state_dict") is not None:
        value_net.load_state_dict(ckpt["value_net_state_dict"])
    elif value_net is not None:
        logger.warning(
            "Checkpoint has no value_net_state_dict — keeping initialised weights."
        )

    if (
        value_optimizer is not None
        and ckpt.get("value_optimizer_state_dict") is not None
    ):
        value_optimizer.load_state_dict(ckpt["value_optimizer_state_dict"])
    elif value_optimizer is not None:
        logger.warning(
            "Checkpoint has no value_optimizer_state_dict — keeping initialised state."
        )

    logger.info(
        f"Loaded full CQL state from {ckpt_path} "
        f"(global_step={ckpt.get('global_step', '?')})"
    )
    return ckpt


# Backward-compatible names (legacy API surface)
save_cql_params = save_offline_rl_params
load_cql_params = load_offline_rl_params

__all__ = [
    "save_cql_params",
    "load_cql_params",
    "save_offline_rl_params",
    "load_offline_rl_params",
]
