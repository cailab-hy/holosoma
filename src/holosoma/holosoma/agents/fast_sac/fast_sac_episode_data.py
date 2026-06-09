from __future__ import annotations

from typing import Any

from loguru import logger

from holosoma.agents.fast_sac.fast_sac_agent import (
    FastSACAgent,
    close_transition_saver,
    init_transition_saver,
    save_transition,
)
from holosoma.utils.safe_torch_import import TensorDict, torch


class FastSACEpisodeDataAgent(FastSACAgent):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._episode_recording_mask: torch.Tensor | None = None
        self._episode_buffers: list[list[TensorDict]] = []
        self._episode_ids: torch.Tensor | None = None
        self._next_episode_id = 0
        self._completed_episode_count = 0

    def _target_active_envs(self) -> int:
        return min(max(int(self.config.episode_data_active_envs), 0), self.env.num_envs)

    def _init_transition_exporter(self) -> bool:
        if not self.is_main_process:
            return False

        init_transition_saver(self.config.offline_dataset_path, flush_every=1)
        self._episode_recording_mask = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        self._episode_buffers = [[] for _ in range(self.env.num_envs)]
        self._episode_ids = torch.full((self.env.num_envs,), -1, device=self.device, dtype=torch.long)
        self._next_episode_id = 0
        self._completed_episode_count = 0

        self._activate_episode_recorders(torch.arange(self.env.num_envs, device=self.device))
        logger.info(
            "FastSAC episode-data export enabled: "
            f"active_envs={self._target_active_envs()}, path={self.config.offline_dataset_path}"
        )
        return True

    def _export_transition_batch(self, transition_to_save: TensorDict, *, dones: torch.Tensor, infos: dict[str, Any]) -> None:
        if self._episode_recording_mask is None:
            return

        active_envs = self._episode_recording_mask.nonzero(as_tuple=False).flatten()
        for env_id_tensor in active_envs:
            env_id = int(env_id_tensor.item())
            self._episode_buffers[env_id].append(transition_to_save[env_id : env_id + 1].clone().cpu())

        finished_envs = self._finished_recording_envs(dones=dones, infos=infos)
        for env_id_tensor in finished_envs:
            env_id = int(env_id_tensor.item())
            self._flush_episode(env_id)
            self._episode_recording_mask[env_id] = False
            self._episode_buffers[env_id] = []
            if self._episode_ids is not None:
                self._episode_ids[env_id] = -1

        self._activate_episode_recorders(finished_envs)

    def _finished_recording_envs(self, *, dones: torch.Tensor, infos: dict[str, Any]) -> torch.Tensor:
        if self._episode_recording_mask is None:
            return torch.empty(0, device=self.device, dtype=torch.long)

        done_flags = dones.to(device=self.device, dtype=torch.bool).flatten()
        timeout_flags = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        time_outs = infos.get("time_outs")
        if isinstance(time_outs, torch.Tensor):
            timeout_flags = time_outs.to(device=self.device, dtype=torch.bool).flatten()

        finished_mask = self._episode_recording_mask & (done_flags | timeout_flags)
        return finished_mask.nonzero(as_tuple=False).flatten()

    def _activate_episode_recorders(self, candidate_env_ids: torch.Tensor) -> None:
        if self._episode_recording_mask is None or self._episode_ids is None:
            return

        slots = self._target_active_envs() - int(self._episode_recording_mask.sum().item())
        if slots <= 0:
            return

        candidate_env_ids = candidate_env_ids.to(device=self.device, dtype=torch.long).flatten()
        if candidate_env_ids.numel() == 0:
            return

        candidate_env_ids = candidate_env_ids[~self._episode_recording_mask[candidate_env_ids]]
        if candidate_env_ids.numel() == 0:
            return

        perm = torch.randperm(candidate_env_ids.numel(), device=self.device)
        selected_env_ids = candidate_env_ids[perm[:slots]]
        for env_id_tensor in selected_env_ids:
            env_id = int(env_id_tensor.item())
            self._episode_recording_mask[env_id] = True
            self._episode_buffers[env_id] = []
            self._episode_ids[env_id] = self._next_episode_id
            self._next_episode_id += 1

    def _flush_episode(self, env_id: int) -> None:
        if self._episode_ids is None:
            return

        episode_steps = self._episode_buffers[env_id]
        if not episode_steps:
            return

        episode = torch.cat(episode_steps, dim=0)
        rewards = episode["next"]["rewards"].to(dtype=torch.float32).reshape(-1)
        mc_returns = self._compute_mc_returns(rewards)
        num_steps = int(rewards.numel())
        episode_id = int(self._episode_ids[env_id].item())

        episode["mc_return"] = mc_returns
        episode["episode_id"] = torch.full((num_steps,), episode_id, dtype=torch.long)
        episode["episode_return"] = torch.full((num_steps,), float(rewards.sum().item()), dtype=torch.float32)
        episode["episode_length"] = torch.full((num_steps,), num_steps, dtype=torch.long)
        episode["episode_data_complete"] = torch.ones((num_steps,), dtype=torch.uint8)

        save_transition(episode)
        self._completed_episode_count += 1

    def _compute_mc_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        gamma = self.config.episode_data_mc_gamma
        discount = float(self.config.gamma if gamma is None else gamma)
        mc_returns = torch.empty_like(rewards)
        running_return = torch.zeros((), dtype=rewards.dtype, device=rewards.device)

        for step_idx in range(rewards.numel() - 1, -1, -1):
            running_return = rewards[step_idx] + discount * running_return
            mc_returns[step_idx] = running_return
        return mc_returns

    def _close_transition_exporter(self) -> None:
        active_count = 0
        if self._episode_recording_mask is not None:
            active_count = int(self._episode_recording_mask.sum().item())
        if active_count > 0:
            logger.info(f"Dropping {active_count} partial active episodes to keep MC returns complete")
        logger.info(f"Saved {self._completed_episode_count} complete episodes")
        close_transition_saver()
