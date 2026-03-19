from __future__ import annotations

import copy
import dataclasses
import itertools
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import h5py
import tqdm
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.cql.cql import Actor, CNNActor, CNNCritic, Critic
from holosoma.agents.cql.cql_utils import (
    EmpiricalNormalization,
    SimpleReplayBuffer,
    save_params,
)
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.config_types.algo import CQLConfig
from holosoma.config_types.env import get_tyro_env_config
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.average_meters import TensorAverageMeterDict
from holosoma.utils.helpers import get_class
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_motion_and_policy_as_onnx,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)
from holosoma.utils.safe_torch_import import (
    F,
    GradScaler,
    TensorboardSummaryWriter,
    TensorDict,
    autocast,
    nn,
    optim,
    torch,
)
import numpy as np

torch.set_float32_matmul_precision("high")


class CQLEnv:
    def __init__(
        self,
        env: BaseTask,
        actor_obs_keys: Sequence[str],
        critic_obs_keys: Sequence[str],
    ):
        self._env = env
        self._actor_obs_keys = actor_obs_keys
        self._critic_obs_keys = critic_obs_keys

        # Initialize per-joint action boundaries for proper tanh scaling
        self._action_boundaries = self._compute_action_boundaries()

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self._env, name)

    def reset(self) -> torch.Tensor:
        obs_dict = self._env.reset_all()
        return torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)

    def reset_with_critic_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs_dict = self._env.reset_all()
        actor_obs = torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)
        critic_obs = torch.cat([obs_dict[k] for k in self._critic_obs_keys], dim=1)
        return actor_obs, critic_obs

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        # Actions are now already scaled by the actor, so pass them directly to the environment
        obs_dict, rew_buf, reset_buf, info_dict = self._env.step({"actions": actions})  # type: ignore[attr-defined]
        actor_obs = torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)
        critic_obs = torch.cat([obs_dict[k] for k in self._critic_obs_keys], dim=1)
        if "final_observations" in info_dict:
            # Use true final observations when available
            final_actor_obs = torch.cat([info_dict["final_observations"][k] for k in self._actor_obs_keys], dim=1)
            final_critic_obs = torch.cat([info_dict["final_observations"][k] for k in self._critic_obs_keys], dim=1)
        else:
            final_actor_obs = actor_obs
            final_critic_obs = critic_obs
        extras = {
            "time_outs": info_dict["time_outs"],
            "observations": {
                "critic": critic_obs,
                "final": {
                    "actor_obs": final_actor_obs,
                    "critic_obs": final_critic_obs,
                },
            },
            "episode": info_dict["episode"],
            "episode_all": info_dict["episode_all"],
            "raw_episode": info_dict.get("raw_episode", {}),
            "raw_episode_all": info_dict.get("raw_episode_all", {}),
            "to_log": info_dict["to_log"],
        }
        return actor_obs, rew_buf, reset_buf, extras

    def _compute_action_boundaries(self) -> torch.Tensor:
        """
        Compute per-joint action scaling factors based on robot configuration.
        Returns tensor of shape (num_dof,) containing the scaling factor for each joint.

        The scaling factor is the maximum difference between default and joint limits,
        ensuring that action=0 corresponds to default position and action=±1 reaches
        the furthest limit from default.
        """
        robot_config = self._env.robot_config

        # Get joint limits and default positions
        dof_pos_lower_limits = torch.tensor(robot_config.dof_pos_lower_limit_list, device=self._env.device)
        dof_pos_upper_limits = torch.tensor(robot_config.dof_pos_upper_limit_list, device=self._env.device)

        # Get default joint angles
        default_joint_angles = torch.zeros(len(robot_config.dof_names), device=self._env.device)
        for i, joint_name in enumerate(robot_config.dof_names):
            if joint_name in robot_config.init_state.default_joint_angles:
                default_joint_angles[i] = robot_config.init_state.default_joint_angles[joint_name]

        # Get action scale from robot config
        action_scale = robot_config.control.action_scale

        # Compute maximum range from default to either limit for each joint
        # This ensures symmetric scaling where action=0 -> default position
        range_to_lower = torch.abs(dof_pos_lower_limits - default_joint_angles)
        range_to_upper = torch.abs(dof_pos_upper_limits - default_joint_angles)
        max_range = torch.maximum(range_to_lower, range_to_upper)

        # Account for action_scale: the environment applies actions_scaled = actions * action_scale
        # So our scaling factor should be: max_range / action_scale
        action_scaling_factors = max_range / action_scale

        logger.info(f"Computed action scaling factors for {len(robot_config.dof_names)} DOFs")
        logger.info(f"Action scale: {action_scale}")
        logger.info(f"Scaling: {action_scaling_factors}")

        return action_scaling_factors




class CQLAgent(BaseAlgo):
    """
    FastSAC is an efficient variant of Soft Actor-Critic (SAC) tuned for
    large-scale training with massively parallel simulation.
    See https://arxiv.org/abs/2505.22642 for more details about FastTD3.
    Detailed technical report for FastSAC will be available soon.
    """

    config: CQLConfig
    env: CQLEnv  # type: ignore[assignment]
    actor: Actor
    qnet: Critic

    def __init__(
        self, env: BaseTask, config: CQLConfig, device: str, log_dir: str, multi_gpu_cfg: dict | None = None
    ):
        wrapped_env = CQLEnv(env, config.actor_obs_keys, config.critic_obs_keys)

        super().__init__(wrapped_env, config, device, multi_gpu_cfg)  # type: ignore[arg-type]
        self.unwrapped_env = env
        self.log_dir = log_dir
        self.global_step = 0
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            device=self.device,
            num_envs=self.env.num_envs,
            num_steps_per_env=config.logging_interval,
            num_learning_iterations=config.num_learning_iterations,
            is_main_process=self.is_main_process,
            num_gpus=self.gpu_world_size,
        )

        self.training_metrics = TensorAverageMeterDict()

        self._num_repeat_actions = 10
        self._temperature =1.0
        self._cql_weight = 5
        self.eval_step = 1000
        self._offline_dataset_path = Path("offline_data/fastsac_dataset.h5")
        self._offline_dataset_cache: dict[str, torch.Tensor] | None = None
        self._offline_num_samples = 0
        self._eval_envs: dict[int, CQLEnv] = {}



    def setup(self) -> None:
        logger.info("Setting up CQL")

        # Log curriculum synchronization status for multi-GPU training
        if self.is_multi_gpu:
            if self.has_curricula_enabled():
                logger.info(f"Multi-GPU curriculum synchronization enabled across {self.gpu_world_size} GPUs")

        args = self.config
        device = self.device
        env = self.env

        algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()

        algo_history_length_dict: Dict[str, int] = {}

        for group_cfg in self.env.observation_manager.cfg.groups.values():
            history_len = getattr(group_cfg, "history_length", 1)
            for term_name in group_cfg.terms:
                algo_history_length_dict[term_name] = history_len

        actor_obs_keys = self.config.actor_obs_keys
        critic_obs_keys = self.config.critic_obs_keys

        n_act = self.env.robot_config.actions_dim

        # Compute actor observation dimensions and store indices
        actor_obs_dim = 0
        self.actor_obs_indices = {}
        for obs_key in actor_obs_keys:
            history_len = algo_history_length_dict.get(obs_key, 1)
            obs_size = algo_obs_dim_dict[obs_key] * history_len

            # Store start and end indices for this observation key
            self.actor_obs_indices[obs_key] = {
                "start": actor_obs_dim,
                "end": actor_obs_dim + obs_size,
                "size": obs_size,
            }
            actor_obs_dim += obs_size

        self.actor_obs_dim = actor_obs_dim

        # Compute critic observation dimensions and store indices
        critic_obs_dim = 0
        self.critic_obs_indices = {}
        for obs_key in critic_obs_keys:
            history_len = algo_history_length_dict.get(obs_key, 1)
            obs_size = algo_obs_dim_dict[obs_key] * history_len

            # Store start and end indices for this observation key
            self.critic_obs_indices[obs_key] = {
                "start": critic_obs_dim,
                "end": critic_obs_dim + obs_size,
                "size": obs_size,
            }
            critic_obs_dim += obs_size

        self.scaler = GradScaler(enabled=args.amp)

        self.obs_normalization = args.obs_normalization
        if args.obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(shape=actor_obs_dim, device=device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(shape=critic_obs_dim, device=device)
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        # Get action scaling parameters from the environment
        action_scale = env._action_boundaries if args.use_tanh else torch.ones(n_act, device=device)
        action_bias = torch.zeros(n_act, device=device)  # Assuming zero bias for now
        print(f"action_scale: {action_scale}")
        print(f"action_bias: {action_bias}")
        # Handle CNN actor/critic
        if args.use_cnn_encoder:
            # We assume that MLP doesn't take raw encoder observations
            actor_mlp_obs_keys = [k for k in actor_obs_keys if k != args.encoder_obs_key]
            critic_mlp_obs_keys = [k for k in critic_obs_keys if k != args.encoder_obs_key]
        else:
            actor_mlp_obs_keys = list(actor_obs_keys)
            critic_mlp_obs_keys = list(critic_obs_keys)
        actor_cls, critic_cls = (CNNActor, CNNCritic) if args.use_cnn_encoder else (Actor, Critic)

        self.actor = actor_cls(
            obs_indices=self.actor_obs_indices,
            obs_keys=actor_mlp_obs_keys,
            n_act=n_act,
            num_envs=env.num_envs,
            device=device,
            hidden_dim=args.actor_hidden_dim,
            log_std_max=args.log_std_max,
            log_std_min=args.log_std_min,
            use_tanh=args.use_tanh,
            use_layer_norm=args.use_layer_norm,
            action_scale=action_scale,
            action_bias=action_bias,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )
        self.qnet = critic_cls(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_mlp_obs_keys,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            device=device,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )
        print("SELF.ACTOR")
        print(self.actor)
        print("SELF.QMET")
        print(self.qnet)
        self.log_alpha = torch.tensor([math.log(args.alpha_init)], requires_grad=True, device=device)
        self.policy = self.actor.explore

        self.qnet_target = critic_cls(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_mlp_obs_keys,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            device=device,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.q_optimizer = optim.AdamW(
            list(self.qnet.parameters()),
            lr=args.critic_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.actor_optimizer = optim.AdamW(
            list(self.actor.parameters()),
            lr=args.actor_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )

        self.target_entropy = -n_act * args.target_entropy_ratio
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=args.alpha_learning_rate, fused=True, betas=(0.9, 0.95))

        logger.info(f"actor_obs_dim: {actor_obs_dim}, critic_obs_dim: {critic_obs_dim}")

        self.rb = SimpleReplayBuffer(
            n_env=env.num_envs,
            buffer_size=args.buffer_size,
            n_obs=actor_obs_dim,
            n_act=n_act,
            n_critic_obs=critic_obs_dim,
            n_steps=args.num_steps,
            gamma=args.gamma,
            device=device,
        )

        if args.use_symmetry:
            # using env._env is not really ideal..
            self.symmetry_utils = SymmetryUtils(env._env)

        # Synchronize model parameters across GPUs for consistent initialization
        if self.is_multi_gpu:
            self._synchronize_model_parameters()

    @contextmanager
    def _maybe_amp(self):
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.config.amp):
            yield

    def _synchronize_model_parameters(self):
        """Synchronize actor, qnet, and log_alpha parameters across all GPUs."""
        # Broadcast actor weights from rank 0 to all other ranks
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast qnet weights from rank 0 to all other ranks
        for param in self.qnet.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast log_alpha parameter from rank 0 to all other ranks
        torch.distributed.broadcast(self.log_alpha.data, src=0)

        # Load qnet_target weights from synced qnet
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        logger.info(f"Synchronized model parameters across {self.gpu_world_size} GPUs")

    def _all_reduce_model_grads(self, model: nn.Module) -> None:
        """Batches and all-reduces gradients across GPUs to reduce NCCL call count.

        This flattens all existing parameter gradients into a single contiguous
        tensor, performs one all_reduce, averages by world size, and then
        scatters the reduced values back into the original gradient tensors.
        """
        if not self.is_multi_gpu:
            return
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for p in model.parameters():
            if p.grad is not None:
                n = p.numel()
                p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
                offset += n

    def _update_main(
        self, data: TensorDict
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        args = self.config

        scaler = self.scaler
        actor = self.actor
        qnet = self.qnet
        qnet_target = self.qnet_target
        q_optimizer = self.q_optimizer
        alpha_optimizer = self.alpha_optimizer

        with self._maybe_amp():
            observations =data["observations"]
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()

            with torch.no_grad():
                next_state_actions, next_state_log_probs = actor.get_actions_and_log_probs(next_observations)
                discount = args.gamma ** data["next"]["effective_n_steps"]

                target_distributions = qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    rewards - discount * bootstrap * self.log_alpha.exp() * next_state_log_probs,
                    bootstrap,
                    discount,
                )
                target_values = qnet_target.get_value(target_distributions)
                target_value_max = target_values.max()
                target_value_min = target_values.min()

            q_outputs = qnet(critic_observations, actions)
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            bellman_loss = critic_losses.mean(dim=1).sum(dim=0)
            #================================ ADDTION CQL LOSS TERM ===================================
            B = actions.shape[0]
            R = self._num_repeat_actions

            
            expanded_obs = observations[:, None, :].expand(B, R, -1).reshape(B * R, -1)
            expanded_critic_obs = critic_observations[:, None, :].expand(B, R, -1).reshape(B * R, -1)
            expanded_next_obs = next_observations[:, None, :].expand(B, R, -1).reshape(B * R, -1)
            expanded_next_critic_obs = next_critic_observations[:, None, :].expand(B, R, -1).reshape(B * R, -1)


            with torch.no_grad():
                curr_actions, curr_logp = actor.get_actions_and_log_probs(expanded_obs)
                next_actions, next_logp = actor.get_actions_and_log_probs(expanded_next_obs)

            # Sample random actions in the same scaled action space as the policy/data.
            # action = tanh(raw_action) * action_scale + action_bias
            action_scale = actor.action_scale
            action_bias = actor.action_bias
            rand_actions = (
                torch.empty(B * R, actions.shape[-1], device=self.device).uniform_(-1.0, 1.0) * action_scale
                + action_bias
            )

            q_data = qnet.get_value(F.softmax(q_outputs, dim=-1))              # [num_q, B]

            q_rand = qnet.get_value(F.softmax(qnet(expanded_critic_obs, rand_actions), dim=-1))
            q_curr = qnet.get_value(F.softmax(qnet(expanded_critic_obs, curr_actions), dim=-1))
            q_next = qnet.get_value(F.softmax(qnet(expanded_next_critic_obs, next_actions), dim=-1))

            q_rand = q_rand.view(self.config.num_q_networks, B, R)
            q_curr = q_curr.view(self.config.num_q_networks, B, R)
            q_next = q_next.view(self.config.num_q_networks, B, R)

            curr_logp = curr_logp.squeeze(-1).reshape(B, R)
            next_logp = next_logp.squeeze(-1).reshape(B, R)
            # log p(a) for uniform over per-dim bounds [bias - scale, bias + scale]
            random_density = -torch.log(2.0 * action_scale + 1e-6).sum()

            cat_q = torch.cat([
                q_rand - random_density,
                q_curr - curr_logp.unsqueeze(0),
                q_next - next_logp.unsqueeze(0),
            ], dim=-1)

            log_sum_exp = torch.logsumexp(cat_q / self._temperature, dim=-1) * self._temperature
            cql_gap = (log_sum_exp - q_data).mean()
            conservative_loss = self._cql_weight * (log_sum_exp - q_data).mean(dim=-1).sum()

             #================================ ADDTION CQL LOSS TERM ===================================

            qf_loss = bellman_loss + conservative_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(qnet)

        scaler.unscale_(q_optimizer)
        if args.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(q_optimizer)
        scaler.update()
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.config.use_autotune:
            alpha_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                alpha_loss = (-self.log_alpha.exp() * (next_state_log_probs.detach() + self.target_entropy)).mean()

            scaler.scale(alpha_loss).backward()

            if self.is_multi_gpu:
                if self.log_alpha.grad is not None:
                    torch.distributed.all_reduce(self.log_alpha.grad.data, op=torch.distributed.ReduceOp.SUM)
                    self.log_alpha.grad.data.copy_(self.log_alpha.grad.data / self.gpu_world_size)

            scaler.unscale_(alpha_optimizer)

            scaler.step(alpha_optimizer)
            scaler.update()

        return (
            rewards.mean(),
            critic_grad_norm.detach(),
            qf_loss.detach(),
            target_value_max.detach(),
            target_value_min.detach(),
            alpha_loss.detach(),
            conservative_loss.detach(),
            bellman_loss.detach(),
            cql_gap.detach(),
            q_data.mean().detach(),
        )

    def _update_pol(self, data: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor = self.actor
        qnet = self.qnet
        actor_optimizer = self.actor_optimizer
        scaler = self.scaler
        args = self.config

        with self._maybe_amp():
            critic_observations = data["critic_observations"]

            actions, log_probs = actor.get_actions_and_log_probs(data["observations"])
            # For logging, this is a bit wasteful though, but could be useful
            with torch.no_grad():
                _, _, log_std = actor(data["observations"])
                action_std = log_std.exp().mean()
                # Compute policy entropy (negative log probability)
                policy_entropy = -log_probs.mean()

            q_outputs = qnet(critic_observations, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = qnet.get_value(q_probs)
            qf_value = q_values.mean(dim=0)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(actor)

        scaler.unscale_(actor_optimizer)

        if args.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(actor_optimizer)
        scaler.update()
        return (
            actor_grad_norm.detach(),
            actor_loss.detach(),
            policy_entropy.detach(),
            action_std.detach(),
        )

    def _sample_and_prepare_batches(
        self, batch_size: int, num_updates: int, normalize_obs, normalize_critic_obs
    ) -> list[TensorDict]:
        """
        Sample a large batch once and split it into smaller batches for each update.
        This reduces sampling overhead by `num_updates` and normalization overhead by `num_updates`.
        """
        # Sample a large batch (batch_size * num_updates)
        large_batch_size = batch_size * num_updates
        large_data = self.rb.sample(large_batch_size)
        samples_per_update = batch_size * self.env.num_envs

        if self.config.use_symmetry:
            samples_per_update *= 2

            augmented_large_data: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {"next": {}}

            augmented_large_data["observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["observations"],
                env=self.env,
                obs_list=self.config.actor_obs_keys,
            )
            augmented_large_data["actions"] = self.symmetry_utils.augment_actions(actions=large_data["actions"])
            assert isinstance(augmented_large_data["next"], dict)
            augmented_large_data["next"]["observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["next"]["observations"],
                env=self.env,
                obs_list=self.config.actor_obs_keys,
            )
            augmented_large_data["critic_observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["critic_observations"],
                env=self.env,
                obs_list=self.config.critic_obs_keys,
            )
            augmented_large_data["next"]["critic_observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["next"]["critic_observations"],
                env=self.env,
                obs_list=self.config.critic_obs_keys,
            )

            # Calculate augmentation factor and repeat non-augmented data
            observations_tensor = augmented_large_data["observations"]
            assert isinstance(observations_tensor, torch.Tensor), (
                "observations should be a Tensor after data augmentation"
            )
            num_aug = int(observations_tensor.shape[0] / large_data["next"]["rewards"].shape[0])
            augmented_large_data["next"]["rewards"] = large_data["next"]["rewards"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["dones"] = large_data["next"]["dones"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["truncations"] = large_data["next"]["truncations"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["effective_n_steps"] = large_data["next"]["effective_n_steps"].repeat(num_aug)  # type: ignore[index]

            # Override large_data
            large_data = augmented_large_data

        # Normalize all data once
        large_data["observations"] = normalize_obs(large_data["observations"])
        large_data["next"]["observations"] = normalize_obs(large_data["next"]["observations"])
        large_data["critic_observations"] = normalize_critic_obs(large_data["critic_observations"])
        large_data["next"]["critic_observations"] = normalize_critic_obs(large_data["next"]["critic_observations"])

        # Split into smaller batches
        prepared_batches = []

        for i in range(num_updates):
            start_idx = i * samples_per_update
            end_idx = (i + 1) * samples_per_update

            # Create a slice of the large batch
            batch_data = TensorDict(
                {
                    "observations": large_data["observations"][start_idx:end_idx],
                    "actions": large_data["actions"][start_idx:end_idx],
                    "next": {
                        "rewards": large_data["next"]["rewards"][start_idx:end_idx],
                        "dones": large_data["next"]["dones"][start_idx:end_idx],
                        "truncations": large_data["next"]["truncations"][start_idx:end_idx],
                        "observations": large_data["next"]["observations"][start_idx:end_idx],
                        "effective_n_steps": large_data["next"]["effective_n_steps"][start_idx:end_idx],
                    },
                    "critic_observations": large_data["critic_observations"][start_idx:end_idx],
                },
                batch_size=samples_per_update,
            )
            batch_data["next"]["critic_observations"] = large_data["next"]["critic_observations"][start_idx:end_idx]

            prepared_batches.append(batch_data)

        return prepared_batches

    def _load_offline_dataset_cache(self) -> dict[str, torch.Tensor]:
        if self._offline_dataset_cache is not None:
            return self._offline_dataset_cache

        if not self._offline_dataset_path.exists():
            raise FileNotFoundError(
                f"Offline dataset not found at '{self._offline_dataset_path}'. "
                "Expected file: offline_data/fastsac_dataset.h5"
            )

        required_keys = (
            "observations",
            "actions",
            "critic_observations",
            "next_observations",
            "next_critic_observations",
            "rewards",
            "truncations",
            "dones",
        )

        with h5py.File(self._offline_dataset_path, "r") as offline_dataset_file:
            missing_keys = [key for key in required_keys if key not in offline_dataset_file]
            if missing_keys:
                raise KeyError(f"Offline dataset is missing required keys: {missing_keys}")

            self._offline_num_samples = int(
                offline_dataset_file.attrs.get("num_samples", offline_dataset_file["observations"].shape[0])
            )
            if self._offline_num_samples <= 0:
                raise ValueError("Offline dataset has no samples.")

            def _load_tensor(key: str, dtype: torch.dtype) -> torch.Tensor:
                array = np.asarray(offline_dataset_file[key][: self._offline_num_samples])
                tensor = torch.from_numpy(array).to(dtype=dtype).contiguous()
                if torch.cuda.is_available():
                    try:
                        tensor = tensor.pin_memory()
                    except RuntimeError:
                        # Pinning can fail under strict memory pressure; fallback to normal CPU tensor.
                        pass
                return tensor

            self._offline_dataset_cache = {
                "observations": _load_tensor("observations", torch.float32),
                "actions": _load_tensor("actions", torch.float32),
                "critic_observations": _load_tensor("critic_observations", torch.float32),
                "next_observations": _load_tensor("next_observations", torch.float32),
                "next_critic_observations": _load_tensor("next_critic_observations", torch.float32),
                "rewards": _load_tensor("rewards", torch.float32),
                "truncations": _load_tensor("truncations", torch.int64),
                "dones": _load_tensor("dones", torch.int64),
            }

        logger.info(
            f"Cached offline dataset '{self._offline_dataset_path}' in host memory "
            f"({self._offline_num_samples} samples)."
        )
        return self._offline_dataset_cache

    def offline_dataset_random_sampleing(
        self, batch_size: int, num_updates: int, normalize_obs, normalize_critic_obs
    ) -> list[TensorDict]:
        """Sample offline data from HDF5 and prepare update batches.

        Note: function name keeps user's requested spelling.
        """
        offline_cache = self._load_offline_dataset_cache()
        samples_per_update = batch_size * self.env.num_envs
        large_batch_size = samples_per_update * num_updates
        replace = large_batch_size > self._offline_num_samples

        if replace:
            logger.warning(
                f"Requested {large_batch_size} samples but dataset has {self._offline_num_samples}. "
                "Sampling with replacement."
            )

        if replace:
            indices = torch.randint(self._offline_num_samples, (large_batch_size,), device="cpu")
        else:
            indices = torch.randperm(self._offline_num_samples, device="cpu")[:large_batch_size]

        def _sample_cached(name: str, dtype: torch.dtype) -> torch.Tensor:
            return offline_cache[name].index_select(0, indices).to(device=self.device, dtype=dtype, non_blocking=True)

        large_data = TensorDict(
            {
                "observations": _sample_cached("observations", torch.float32),
                "actions": _sample_cached("actions", torch.float32),
                "next": {
                    "rewards": _sample_cached("rewards", torch.float32),
                    "dones": _sample_cached("dones", torch.long),
                    "truncations": _sample_cached("truncations", torch.long),
                    "observations": _sample_cached("next_observations", torch.float32),
                    "effective_n_steps": torch.ones(large_batch_size, device=self.device, dtype=torch.long),
                },
                "critic_observations": _sample_cached("critic_observations", torch.float32),
            },
            batch_size=large_batch_size,
            device=self.device,
        )
        large_data["next"]["critic_observations"] = _sample_cached("next_critic_observations", torch.float32)

        if self.config.use_symmetry:
            samples_per_update *= 2

            augmented_large_data: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {"next": {}}

            augmented_large_data["observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["observations"],
                env=self.env,
                obs_list=self.config.actor_obs_keys,
            )
            augmented_large_data["actions"] = self.symmetry_utils.augment_actions(actions=large_data["actions"])
            assert isinstance(augmented_large_data["next"], dict)
            augmented_large_data["next"]["observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["next"]["observations"],
                env=self.env,
                obs_list=self.config.actor_obs_keys,
            )
            augmented_large_data["critic_observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["critic_observations"],
                env=self.env,
                obs_list=self.config.critic_obs_keys,
            )
            augmented_large_data["next"]["critic_observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["next"]["critic_observations"],
                env=self.env,
                obs_list=self.config.critic_obs_keys,
            )

            observations_tensor = augmented_large_data["observations"]
            assert isinstance(observations_tensor, torch.Tensor), (
                "observations should be a Tensor after data augmentation"
            )
            num_aug = int(observations_tensor.shape[0] / large_data["next"]["rewards"].shape[0])
            augmented_large_data["next"]["rewards"] = large_data["next"]["rewards"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["dones"] = large_data["next"]["dones"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["truncations"] = large_data["next"]["truncations"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["effective_n_steps"] = large_data["next"]["effective_n_steps"].repeat(num_aug)  # type: ignore[index]
            large_data = augmented_large_data

        large_data["observations"] = normalize_obs(large_data["observations"])
        large_data["next"]["observations"] = normalize_obs(large_data["next"]["observations"])
        large_data["critic_observations"] = normalize_critic_obs(large_data["critic_observations"])
        large_data["next"]["critic_observations"] = normalize_critic_obs(large_data["next"]["critic_observations"])

        prepared_batches = []
        for i in range(num_updates):
            start_idx = i * samples_per_update
            end_idx = (i + 1) * samples_per_update
            batch_data = TensorDict(
                {
                    "observations": large_data["observations"][start_idx:end_idx],
                    "actions": large_data["actions"][start_idx:end_idx],
                    "next": {
                        "rewards": large_data["next"]["rewards"][start_idx:end_idx],
                        "dones": large_data["next"]["dones"][start_idx:end_idx],
                        "truncations": large_data["next"]["truncations"][start_idx:end_idx],
                        "observations": large_data["next"]["observations"][start_idx:end_idx],
                        "effective_n_steps": large_data["next"]["effective_n_steps"][start_idx:end_idx],
                    },
                    "critic_observations": large_data["critic_observations"][start_idx:end_idx],
                },
                batch_size=samples_per_update,
                device=self.device,
            )
            batch_data["next"]["critic_observations"] = large_data["next"]["critic_observations"][start_idx:end_idx]
            prepared_batches.append(batch_data)

        return prepared_batches

    def offline_dataset_random_sampling(
        self, batch_size: int, num_updates: int, normalize_obs, normalize_critic_obs
    ) -> list[TensorDict]:
        """Backward-compatible alias to the requested `sampleing` method."""
        return self.offline_dataset_random_sampleing(batch_size, num_updates, normalize_obs, normalize_critic_obs)

    def load(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return
        # Load checkpoint if specified
        torch_checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Handle DDP-wrapped models
        actor_state_dict = torch_checkpoint["actor_state_dict"]
        qnet_state_dict = torch_checkpoint["qnet_state_dict"]

        self.actor.load_state_dict(actor_state_dict)
        self.qnet.load_state_dict(qnet_state_dict)

        self.obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        self.critic_obs_normalizer.load_state_dict(torch_checkpoint["critic_obs_normalizer_state"])
        self.qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        self.log_alpha.data.copy_(torch_checkpoint["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(torch_checkpoint["actor_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(torch_checkpoint["q_optimizer_state_dict"])
        self.alpha_optimizer.load_state_dict(torch_checkpoint["alpha_optimizer_state_dict"])
        self.scaler.load_state_dict(torch_checkpoint["grad_scaler_state_dict"])
        self.global_step = torch_checkpoint["global_step"]
        self._restore_env_state(torch_checkpoint.get("env_state"))

    def offline_learn(self) -> None:
            args = self.config
            device = self.device
            if args.compile:
                if not hasattr(self, "_compiled_update_main"):
                    self._compiled_update_main = torch.compile(self._update_main)
                    self._compiled_update_pol = torch.compile(self._update_pol)
                    self._compiled_normalize_obs = torch.compile(self.obs_normalizer.forward)
                    self._compiled_normalize_critic_obs = torch.compile(self.critic_obs_normalizer.forward)
                update_main = self._compiled_update_main
                update_pol = self._compiled_update_pol
                normalize_obs = self._compiled_normalize_obs
                normalize_critic_obs = self._compiled_normalize_critic_obs
            else:
                update_main = self._update_main
                update_pol = self._update_pol
                normalize_obs = self.obs_normalizer.forward
                normalize_critic_obs = self.critic_obs_normalizer.forward

            qnet = self.qnet
            qnet_target = self.qnet_target
            env = self.env

            # Initialize metrics that might not be updated every step
            policy_entropy = torch.tensor(0.0, device=device)
            action_std = torch.tensor(0.0, device=device)
            actor_loss = torch.tensor(0.0, device=device)
            actor_grad_norm = torch.tensor(0.0, device=device)

            # Train in chunks between evaluation points.
            next_eval_boundary = ((self.global_step // self.eval_step) + 1) * self.eval_step
            target_step = min(args.num_learning_iterations, next_eval_boundary)
            if target_step <= self.global_step:
                target_step = min(args.num_learning_iterations, self.global_step + 1)

            pbar = tqdm.tqdm(total=max(target_step - self.global_step, 0), initial=0, leave=False)
            while self.global_step < target_step:
                self.global_step += 1

                # Synchronize curriculum metrics across GPUs before rollout
                if self.is_multi_gpu:
                    self._synchronize_curriculum_metrics()

                # NOTE: args.batch_size is the global batch size
                batch_size = max(args.batch_size // env.num_envs // self.gpu_world_size, 1)
                with self.logging_helper.record_learn_time():
                    offline_batches = self.offline_dataset_random_sampleing(
                        batch_size,
                        args.num_updates,
                        normalize_obs,
                        normalize_critic_obs,
                    )
                    for i, data in enumerate(offline_batches):
                        # Data is already normalized, just run the updates
                        (
                            buffer_rewards,
                            critic_grad_norm,
                            qf_loss,
                            qf_max,
                            qf_min,
                            alpha_loss,
                            conservative_loss,
                            bellman_loss,
                            cql_gap,
                            q_data_mean,
                        ) = update_main(data)
                        if args.num_updates > 1:
                            if i % args.policy_frequency == 1:
                                actor_grad_norm, actor_loss, policy_entropy, action_std = update_pol(data)
                        elif self.global_step % args.policy_frequency == 0:
                            actor_grad_norm, actor_loss, policy_entropy, action_std = update_pol(data)

                        # Accumulate training metrics for smoother logging
                        current_metrics = {
                            "actor_loss": actor_loss,
                            "qf_loss": qf_loss,
                            "qf_max": qf_max,
                            "qf_min": qf_min,
                            "actor_grad_norm": actor_grad_norm,
                            "critic_grad_norm": critic_grad_norm,
                            "buffer_rewards": buffer_rewards,
                            "alpha_loss": alpha_loss,
                            "alpha_value": self.log_alpha.exp().detach().mean(),
                            "policy_entropy": policy_entropy,
                            "action_std": action_std,
                            "cql_conservative_loss": conservative_loss,
                            "cql_bellman_loss": bellman_loss,
                            "cql_gap": cql_gap,
                            "q_data_mean": q_data_mean,
                        }
                        self.training_metrics.add(current_metrics)

                        with torch.no_grad():
                            src_ps = [p.data for p in qnet.parameters()]
                            tgt_ps = [p.data for p in qnet_target.parameters()]
                            torch._foreach_mul_(tgt_ps, 1.0 - args.tau)
                            torch._foreach_add_(tgt_ps, src_ps, alpha=args.tau)

                should_log = (self.global_step % args.logging_interval == 0) or (self.global_step <= 10)
                if should_log:
                    with torch.no_grad():
                        # Use accumulated training metrics for smoother logging (reduces noise)
                        accumulated_metrics = self.training_metrics.mean_and_clear()

                        # Convert tensor values to float for logging
                        loss_dict = {}
                        for key, value in accumulated_metrics.items():
                            if isinstance(value, torch.Tensor):
                                loss_dict[key] = value.item()
                            else:
                                loss_dict[key] = float(value)

                    # Use logging helper
                    self.logging_helper.post_epoch_logging(it=self.global_step, loss_dict=loss_dict, extra_log_dicts={})

                if args.save_interval > 0 and self.global_step % args.save_interval == 0:
                    if self.is_main_process:
                        logger.info(f"Saving model at global step {self.global_step}")
                        self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
                        self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))

                pbar.update(1)
            pbar.close()

            if self.is_main_process and self.global_step >= args.num_learning_iterations:
                self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
                self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))


    def save(self, path: str) -> None:  # type: ignore[override]
        env_state = self._collect_env_state()
        save_params(
            self.global_step,
            self.actor,
            self.qnet,
            self.qnet_target,
            self.log_alpha,
            self.obs_normalizer,
            self.critic_obs_normalizer,
            self.actor_optimizer,
            self.q_optimizer,
            self.alpha_optimizer,
            self.scaler,
            self.config,
            path,
            save_fn=self.logging_helper.save_checkpoint_artifact,
            env_state=env_state or None,
            metadata=self._checkpoint_metadata(iteration=self.global_step),
        )

    @torch.no_grad()
    def get_example_obs(self):
        """Used for exporting policy as onnx."""
        obs_dict = self.unwrapped_env.reset_all()
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return {
            "actor_obs": torch.cat([obs_dict[k] for k in self.config.actor_obs_keys], dim=1),
            "critic_obs": torch.cat([obs_dict[k] for k in self.config.critic_obs_keys], dim=1),
        }

    def get_inference_policy(self, device: str | None = None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        device = device or self.device
        # Use the underlying module for inference
        policy = self.actor.to(device)
        obs_normalizer = self.obs_normalizer.to(device)
        policy.eval()
        obs_normalizer.eval()

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            if self.obs_normalization:
                normalized_obs = obs_normalizer(obs["actor_obs"], update=False)
            else:
                normalized_obs = obs["actor_obs"]
            # Actions are already scaled by the actor
            return policy(normalized_obs)[0]

        return policy_fn

    @property
    def actor_onnx_wrapper(self):
        # Use the underlying module for ONNX export
        actor = copy.deepcopy(self.actor).to("cpu")
        obs_normalizer = copy.deepcopy(self.obs_normalizer).to("cpu")

        class ActorWrapper(nn.Module):
            def __init__(self, actor, obs_normalizer):
                super().__init__()
                self.actor = actor
                self.obs_normalizer = obs_normalizer

            def forward(self, actor_obs):
                if self.obs_normalizer is not None:
                    normalized_obs = self.obs_normalizer(actor_obs, update=False)
                else:
                    normalized_obs = actor_obs
                # Actions are already scaled by the actor
                return self.actor(normalized_obs)[0]

        return ActorWrapper(actor, obs_normalizer if self.obs_normalization else None)

    def extract_actor_obs(self, obs: torch.Tensor, obs_key: str) -> torch.Tensor:
        """
        Extract a specific observation component from the flattened actor observation tensor.

        Args:
            obs: Flattened actor observation tensor of shape [batch_size, actor_obs_dim]
            obs_key: The observation key to extract (e.g., 'perception_obs', 'actor_state_obs')

        Returns:
            Extracted observation tensor of shape [batch_size, obs_size]
        """
        if obs_key not in self.actor_obs_indices:
            raise ValueError(
                f"Observation key '{obs_key}' not found in actor observations. "
                f"Available keys: {list(self.actor_obs_indices.keys())}"
            )

        indices = self.actor_obs_indices[obs_key]
        return obs[..., indices["start"] : indices["end"]]

    def extract_critic_obs(self, obs: torch.Tensor, obs_key: str) -> torch.Tensor:
        """
        Extract a specific observation component from the flattened critic observation tensor.

        Args:
            obs: Flattened critic observation tensor of shape [batch_size, critic_obs_dim]
            obs_key: The observation key to extract (e.g., 'perception_obs', 'critic_state_obs')

        Returns:
            Extracted observation tensor of shape [batch_size, obs_size]
        """
        if obs_key not in self.critic_obs_indices:
            raise ValueError(
                f"Observation key '{obs_key}' not found in critic observations. "
                f"Available keys: {list(self.critic_obs_indices.keys())}"
            )

        indices = self.critic_obs_indices[obs_key]
        return obs[..., indices["start"] : indices["end"]]

    def get_actor_obs_info(self) -> dict[str, dict[str, int]]:
        """
        Get information about actor observation indices.

        Returns:
            Dictionary with obs_key -> {'start': int, 'end': int, 'size': int}
        """
        return self.actor_obs_indices.copy()

    def get_critic_obs_info(self) -> dict[str, dict[str, int]]:
        """
        Get information about critic observation indices.

        Returns:
            Dictionary with obs_key -> {'start': int, 'end': int, 'size': int}
        """
        return self.critic_obs_indices.copy()

    def export(self, onnx_file_path: str) -> None:
        """Export the `.onnx` of the policy to & save it to `path`.

        This is intended to enable deployment, but not resuming training.
        For storing checkpoints to resume training, see `FastSACAgent.save()`
        """
        # Save current training state
        was_training = self.actor.training

        # Set model to evaluation mode for export so we don't affect gradients mid-rollout
        self.actor.eval()
        if self.obs_normalization:
            self.obs_normalizer.eval()

        # Create dummy all-zero input for ONNX tracing.
        example_input_list = torch.zeros(1, self.actor_obs_dim, device="cpu")

        motion_command = self.unwrapped_env.command_manager.get_state("motion_command")
        if motion_command is not None:
            export_motion_and_policy_as_onnx(
                self.actor_onnx_wrapper,
                motion_command,
                onnx_file_path,
                self.device,
            )
        else:
            export_policy_as_onnx(
                wrapper=self.actor_onnx_wrapper,
                onnx_file_path=onnx_file_path,
                example_obs_dict={"actor_obs": example_input_list},
            )

        # Extract control gains and velocity limits & attach to onnx as metadata
        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.unwrapped_env)
        # Extract URDF text from the robot config
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)

        metadata = {
            "dof_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "command_ranges": cmd_ranges,
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
        }
        metadata.update(self._checkpoint_metadata(iteration=self.global_step))

        attach_onnx_metadata(
            onnx_path=onnx_file_path,
            metadata=metadata,
        )

        self.logging_helper.save_to_wandb(onnx_file_path)

        # Restore original training state
        if was_training:
            self.actor.train()
            if self.obs_normalization:
                self.obs_normalizer.train()

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None):
        self.env.set_is_evaluating()
        obs = self.env.reset()

        for _ in itertools.islice(itertools.count(), max_eval_steps):
            if self.obs_normalization:
                normalized_obs = self.obs_normalizer(obs, update=False)
            else:
                normalized_obs = obs
            # Actions are already scaled by the actor
            actions = self.actor(normalized_obs)[0]
            obs, _, _, _ = self.env.step(actions)

    def _get_or_create_eval_env(self, eval_num_envs: int) -> CQLEnv:
        if eval_num_envs <= 0:
            raise ValueError(f"eval_num_envs must be >= 1, got {eval_num_envs}")

        if eval_num_envs == self.env.num_envs:
            return self.env

        cached = self._eval_envs.get(eval_num_envs)
        if cached is not None:
            return cached

        if self._experiment_config is None:
            raise RuntimeError(
                "Experiment config metadata missing. attach_checkpoint_metadata() must be called before evaluation."
            )

        base_eval_cfg = self._experiment_config.get_eval_config()
        eval_cfg = dataclasses.replace(
            base_eval_cfg,
            training=dataclasses.replace(base_eval_cfg.training, num_envs=eval_num_envs),
        )
        eval_env_class = get_class(eval_cfg.env_class)
        eval_env = eval_env_class(get_tyro_env_config(eval_cfg), device=self.device)
        wrapped_eval_env = CQLEnv(eval_env, self.config.actor_obs_keys, self.config.critic_obs_keys)
        self._eval_envs[eval_num_envs] = wrapped_eval_env
        logger.info(f"Created dedicated evaluation env with num_envs={eval_num_envs}")
        return wrapped_eval_env

    @torch.no_grad()
    def evaluate_one_episode(
        self,
        max_eval_steps: int | None = None,
        use_early_termination: bool = False,
        eval_num_envs: int | None = None,
        ):
        # 가능하면 별도 eval env를 쓰는 게 가장 안전함
        eval_env = self.env if eval_num_envs is None else self._get_or_create_eval_env(eval_num_envs)
        eval_env.set_is_evaluating()
        was_training = self.actor.training
        self.actor.eval()
        if self.obs_normalization:
            self.obs_normalizer.eval()

        obs = eval_env.reset()
        episode_return = torch.zeros(eval_env.num_envs, device=self.device)
        episode_length = 0
        fail_reason = None

        for t in itertools.count():
            if max_eval_steps is not None and t >= max_eval_steps:
                fail_reason = "max_eval_steps"
                break

            if self.obs_normalization:
                normalized_obs = self.obs_normalizer(obs, update=False)
            else:
                normalized_obs = obs

            actions = self.actor(normalized_obs)[0]
            obs, rewards, dones, infos = eval_env.step(actions)

            episode_return += rewards
            episode_length += 1

            # 보통 single-env eval이면 dones.any() 대신 dones[0] 써도 됨
            if dones.any():
                fail_reason = "done"
                break

            if "time_outs" in infos and infos["time_outs"].any():
                fail_reason = "time_out"
                break

            if use_early_termination:
                # 예시: env 쪽에서 ET flag를 제공한다고 가정
                if "early_termination" in infos and infos["early_termination"].any():
                    fail_reason = "early_termination"
                    break

        # train 모드 복구
        if was_training:
            self.actor.train()
            if self.obs_normalization:
                self.obs_normalizer.train()

        if hasattr(eval_env, "set_is_training"):
            eval_env.set_is_training()

        return {
            "episode_return": episode_return.mean().item(),
            "episode_length": episode_length,
            "stop_reason": fail_reason,
        }
