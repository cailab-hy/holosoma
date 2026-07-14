from __future__ import annotations

import copy
import itertools
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import tqdm
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.cql.cql import Actor, CNNActor, DoubleQCritic
from holosoma.agents.cql.cql_utils import EmpiricalNormalization, save_params
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.config_types.algo import CQLConfig
from holosoma.data.hdf5_offline_dataset import (
    GPUTransitionCache,
    HDF5BlockReader,
    RAMShuffleBuffer,
    apply_observation_normalization,
    batch_to_device,
)
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.average_meters import TensorAverageMeterDict
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
        self._action_boundaries = self._compute_action_boundaries()

    def __getattr__(self, name: str):
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
        obs_dict, rew_buf, reset_buf, info_dict = self._env.step({"actions": actions})  # type: ignore[attr-defined]
        actor_obs = torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)
        critic_obs = torch.cat([obs_dict[k] for k in self._critic_obs_keys], dim=1)
        if "final_observations" in info_dict:
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
            "termination_reasons": info_dict.get("termination_reasons", {}),
            "to_log": info_dict["to_log"],
        }
        return actor_obs, rew_buf, reset_buf, extras

    def _compute_action_boundaries(self) -> torch.Tensor:
        robot_config = self._env.robot_config

        dof_pos_lower_limits = torch.tensor(robot_config.dof_pos_lower_limit_list, device=self._env.device)
        dof_pos_upper_limits = torch.tensor(robot_config.dof_pos_upper_limit_list, device=self._env.device)

        default_joint_angles = torch.zeros(len(robot_config.dof_names), device=self._env.device)
        for i, joint_name in enumerate(robot_config.dof_names):
            if joint_name in robot_config.init_state.default_joint_angles:
                default_joint_angles[i] = robot_config.init_state.default_joint_angles[joint_name]

        action_scale = robot_config.control.action_scale
        range_to_lower = torch.abs(dof_pos_lower_limits - default_joint_angles)
        range_to_upper = torch.abs(dof_pos_upper_limits - default_joint_angles)
        max_range = torch.maximum(range_to_lower, range_to_upper)
        action_scaling_factors = max_range / action_scale

        logger.info(f"Computed action scaling factors for {len(robot_config.dof_names)} DOFs")
        logger.info(f"Action scale: {action_scale}")
        logger.info(f"Scaling: {action_scaling_factors}")

        return action_scaling_factors


class CQLAgent(BaseAlgo):
    config: CQLConfig
    env: CQLEnv  # type: ignore[assignment]
    actor: Actor
    qnet: DoubleQCritic
    qnet_target: DoubleQCritic

    def __init__(
        self,
        env: BaseTask,
        config: CQLConfig,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict | None = None,
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

        self.eval_step = max(1, config.eval_interval)
        self._num_repeat_actions = config.cql_num_action_samples
        self._temperature = config.cql_temperature
        self._cql_weight = config.cql_weight
        self._num_near_actions = config.cql_near_action_samples

        self._offline_dataset_path = Path(config.offline_dataset_path)
        self._offline_dataset_reader: HDF5BlockReader | None = None
        self._offline_shuffle_buffer: RAMShuffleBuffer | None = None
        self._offline_gpu_cache: GPUTransitionCache | None = None
        self._offline_num_samples = 0
        self._critic_update_step = 0

        if config.cql_num_action_samples <= 0:
            raise ValueError(f"cql_num_action_samples must be > 0, got {config.cql_num_action_samples}")
        if config.cql_temperature <= 0.0:
            raise ValueError(f"cql_temperature must be > 0, got {config.cql_temperature}")
        if config.cql_weight < 0.0:
            raise ValueError(f"cql_weight must be >= 0, got {config.cql_weight}")
        if config.dr3_weight < 0.0:
            raise ValueError(f"dr3_weight must be >= 0, got {config.dr3_weight}")
        if config.cql_near_action_samples < 0:
            raise ValueError(f"cql_near_action_samples must be >= 0, got {config.cql_near_action_samples}")
        if config.cql_near_noise_std < 0.0:
            raise ValueError(f"cql_near_noise_std must be >= 0, got {config.cql_near_noise_std}")
        if config.use_lagrange:
            if config.cql_lagrange_learning_rate <= 0.0:
                raise ValueError(
                    "cql_lagrange_learning_rate must be > 0 when use_lagrange=True, "
                    f"got {config.cql_lagrange_learning_rate}"
                )
            if config.cql_lagrange_init <= 0.0:
                raise ValueError(f"cql_lagrange_init must be > 0, got {config.cql_lagrange_init}")
            if config.cql_lagrange_max <= 0.0:
                raise ValueError(f"cql_lagrange_max must be > 0, got {config.cql_lagrange_max}")
        if config.gamma <= 0.0 or config.gamma > 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {config.gamma}")
        if config.huber_beta <= 0.0:
            raise ValueError(f"huber_beta must be > 0, got {config.huber_beta}")
        if config.tau <= 0.0 or config.tau > 1.0:
            raise ValueError(f"tau must be in (0, 1], got {config.tau}")
        if config.alpha_init <= 0.0:
            raise ValueError(f"alpha_init must be > 0, got {config.alpha_init}")
        if config.policy_frequency <= 0:
            raise ValueError(f"policy_frequency must be > 0, got {config.policy_frequency}")
        if not config.use_gpu_cache:
            if config.offline_block_size <= 0:
                raise ValueError(f"offline_block_size must be > 0, got {config.offline_block_size}")
            if config.offline_buffer_capacity <= 0:
                raise ValueError(f"offline_buffer_capacity must be > 0, got {config.offline_buffer_capacity}")
            if config.offline_block_size > config.offline_buffer_capacity:
                raise ValueError(
                    "offline_block_size must be <= offline_buffer_capacity, "
                    f"got block_size={config.offline_block_size}, capacity={config.offline_buffer_capacity}"
                )
            if config.offline_refill_threshold < 0 or config.offline_refill_threshold >= config.offline_buffer_capacity:
                raise ValueError(
                    "offline_refill_threshold must be in [0, offline_buffer_capacity), "
                    f"got threshold={config.offline_refill_threshold}, capacity={config.offline_buffer_capacity}"
                )

    def setup(self) -> None:
        logger.info("Setting up scalar offline CQL")

        if self.is_multi_gpu and self.has_curricula_enabled():
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

        actor_obs_dim = 0
        self.actor_obs_indices: dict[str, dict[str, int]] = {}
        for obs_key in args.actor_obs_keys:
            history_len = algo_history_length_dict.get(obs_key, 1)
            obs_size = algo_obs_dim_dict[obs_key] * history_len
            self.actor_obs_indices[obs_key] = {
                "start": actor_obs_dim,
                "end": actor_obs_dim + obs_size,
                "size": obs_size,
            }
            actor_obs_dim += obs_size
        self.actor_obs_dim = actor_obs_dim

        critic_obs_dim = 0
        self.critic_obs_indices: dict[str, dict[str, int]] = {}
        for obs_key in args.critic_obs_keys:
            history_len = algo_history_length_dict.get(obs_key, 1)
            obs_size = algo_obs_dim_dict[obs_key] * history_len
            self.critic_obs_indices[obs_key] = {
                "start": critic_obs_dim,
                "end": critic_obs_dim + obs_size,
                "size": obs_size,
            }
            critic_obs_dim += obs_size
        self.critic_obs_dim = critic_obs_dim
        logger.info(f"Offline dataset sampling backend: use_gpu_cache={args.use_gpu_cache}")
        if args.use_gpu_cache:
            if not torch.cuda.is_available():
                raise RuntimeError("use_gpu_cache=True requires CUDA, but no CUDA device is available.")
            self._offline_dataset_reader = None
            self._offline_shuffle_buffer = None
            self._offline_gpu_cache = GPUTransitionCache(
                self._offline_dataset_path,
                device=self.device,
                expected_observation_dim=self.actor_obs_dim,
                expected_action_dim=self.env.robot_config.actions_dim,
                expected_critic_observation_dim=self.critic_obs_dim,
            )
            self._offline_num_samples = self._offline_gpu_cache.num_samples
            estimated_vram_gib = self._offline_gpu_cache.total_bytes / float(1024**3)
            logger.info(
                f"Configured offline GPU transition cache with {self._offline_num_samples} samples."
            )
            logger.info(f"Offline GPU cache keys: {list(self._offline_gpu_cache.key_names)}")
            logger.info(f"Estimated offline GPU cache VRAM footprint: {estimated_vram_gib:.2f} GiB")
        else:
            self._offline_gpu_cache = None
            self._offline_dataset_reader = HDF5BlockReader(
                self._offline_dataset_path,
                expected_observation_dim=self.actor_obs_dim,
                expected_action_dim=self.env.robot_config.actions_dim,
                expected_critic_observation_dim=self.critic_obs_dim,
                pin_memory=False,
            )
            self._offline_num_samples = self._offline_dataset_reader.num_samples
            self._offline_shuffle_buffer = RAMShuffleBuffer(
                self._offline_dataset_reader,
                block_size=args.offline_block_size,
                capacity=args.offline_buffer_capacity,
                refill_threshold=args.offline_refill_threshold,
                pin_memory=args.offline_pin_memory and torch.cuda.is_available(),
                shuffle_block_order=args.offline_shuffle_block_order,
            )
            estimated_buffer_gib = self._offline_shuffle_buffer.capacity_bytes / float(1024**3)
            logger.info(
                "Configured offline RAM shuffle buffer: "
                f"block_size={args.offline_block_size}, "
                f"capacity={args.offline_buffer_capacity}, "
                f"refill_threshold={args.offline_refill_threshold}, "
                f"estimated_ram={estimated_buffer_gib:.2f} GiB"
            )

        self.scaler = GradScaler(enabled=args.amp)
        #reward_scale
        self.reward_scale = args.reward_scale

        self.obs_normalization = args.obs_normalization
        if self.obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(shape=actor_obs_dim, device=device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(shape=critic_obs_dim, device=device)
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        n_act = self.env.robot_config.actions_dim
        if not args.use_tanh:
            raise ValueError("CQL requires use_tanh=True for bounded action training.")
        env_action_scale = env._action_boundaries
        env_action_bias = torch.zeros(n_act, device=device)
        self.env_action_scale = env_action_scale
        self.env_action_bias = env_action_bias
        action_scale = env_action_scale
        action_bias = env_action_bias
        self.action_space_mode = "env_scaled_action_training_v1"
        logger.info("CQL action semantics: actor/critic use env-scaled action space.")

        actor_obs_keys = list(args.actor_obs_keys)
        if args.use_cnn_encoder:
            actor_obs_keys = [k for k in actor_obs_keys if k != args.encoder_obs_key]
        actor_cls = CNNActor if args.use_cnn_encoder else Actor
        self.actor = actor_cls(
            obs_indices=self.actor_obs_indices,
            obs_keys=actor_obs_keys,
            n_act=n_act,
            num_envs=env.num_envs,
            hidden_dim=args.actor_hidden_dim,
            log_std_max=args.log_std_max,
            log_std_min=args.log_std_min,
            use_tanh=args.use_tanh,
            use_layer_norm=args.use_layer_norm,
            device=device,
            action_scale=action_scale,
            action_bias=action_bias,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )

        self.qnet = DoubleQCritic(
            obs_indices=self.critic_obs_indices,
            obs_keys=list(args.critic_obs_keys),
            n_act=n_act,
            hidden_dim=args.critic_hidden_dim,
            use_layer_norm=args.use_layer_norm,
            device=device,
        )
        self.qnet_target = DoubleQCritic(
            obs_indices=self.critic_obs_indices,
            obs_keys=list(args.critic_obs_keys),
            n_act=n_act,
            hidden_dim=args.critic_hidden_dim,
            use_layer_norm=args.use_layer_norm,
            device=device,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.log_alpha = torch.tensor([math.log(args.alpha_init)], requires_grad=True, device=device)
        self.target_entropy = -float(n_act) * float(args.target_entropy_ratio)
        self.log_cql_alpha: torch.Tensor | None = None
        self.cql_alpha_optimizer: optim.Optimizer | None = None
        if args.use_lagrange:
            self.log_cql_alpha = torch.tensor(
                [math.log(args.cql_lagrange_init)],
                requires_grad=True,
                device=device,
            )

        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=args.actor_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.q_optimizer = optim.AdamW(
            self.qnet.parameters(),
            lr=args.critic_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.alpha_optimizer = optim.AdamW(
            [self.log_alpha],
            lr=args.alpha_learning_rate,
            fused=True,
            betas=(0.9, 0.95),
        )
        if args.use_lagrange:
            assert self.log_cql_alpha is not None
            self.cql_alpha_optimizer = optim.AdamW(
                [self.log_cql_alpha],
                lr=args.cql_lagrange_learning_rate,
                fused=True,
                betas=(0.9, 0.95),
            )

        def _env_policy(obs: torch.Tensor, dones: torch.Tensor | None = None, deterministic: bool = False) -> torch.Tensor:
            return self._to_env_actions(self.actor.explore(obs, dones=dones, deterministic=deterministic))

        self.policy = _env_policy
        logger.info(f"CQL dims: actor_obs_dim={actor_obs_dim}, critic_obs_dim={critic_obs_dim}, n_act={n_act}")

        if args.use_symmetry:
            self.symmetry_utils = SymmetryUtils(env._env)

        if self.is_multi_gpu:
            self._synchronize_model_parameters()

    @contextmanager
    def _maybe_amp(self):
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.config.amp):
            yield

    def _synchronize_model_parameters(self) -> None:
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)
        for param in self.qnet.parameters():
            torch.distributed.broadcast(param.data, src=0)
        torch.distributed.broadcast(self.log_alpha.data, src=0)
        if self.config.use_lagrange and self.log_cql_alpha is not None:
            torch.distributed.broadcast(self.log_cql_alpha.data, src=0)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        logger.info(f"Synchronized CQL model parameters across {self.gpu_world_size} GPUs")

    def _all_reduce_model_grads(self, model: nn.Module) -> None:
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

    def _soft_update_q_target(self) -> None:
        with torch.no_grad():
            src_ps = [p.data for p in self.qnet.parameters()]
            tgt_ps = [p.data for p in self.qnet_target.parameters()]
            torch._foreach_mul_(tgt_ps, 1.0 - self.config.tau)
            torch._foreach_add_(tgt_ps, src_ps, alpha=self.config.tau)

    def _to_critic_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def _to_env_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def _sync_actor_action_space_buffers(self) -> None:
        with torch.no_grad():
            self.actor.action_scale.copy_(
                self.env_action_scale.to(device=self.actor.action_scale.device, dtype=self.actor.action_scale.dtype)
            )
            self.actor.action_bias.copy_(
                self.env_action_bias.to(device=self.actor.action_bias.device, dtype=self.actor.action_bias.dtype)
            )

    @torch.no_grad()
    def _compute_action_ood_stats(self, data: TensorDict) -> dict[str, torch.Tensor]:
        """Compute per-dimension dataset-vs-policy action coverage stats in critic action space."""
        dataset_actions = self._to_critic_actions(data["actions"])  # [B, action_dim]
        actor_observations = data["observations"]  # [B, actor_obs_dim]

        policy_actions = self.actor(actor_observations)[0]  # [B, action_dim] in critic action space
        dataset_actions = dataset_actions.float()
        policy_actions = policy_actions.float()

        quantiles = torch.tensor([0.01, 0.50, 0.99], device=dataset_actions.device, dtype=dataset_actions.dtype)
        dataset_q = torch.quantile(dataset_actions, q=quantiles, dim=0)  # [3, action_dim]
        policy_q = torch.quantile(policy_actions, q=quantiles, dim=0)  # [3, action_dim]

        dataset_p1, dataset_p50, dataset_p99 = dataset_q[0], dataset_q[1], dataset_q[2]
        policy_p1, policy_p50, policy_p99 = policy_q[0], policy_q[1], policy_q[2]

        # Positive overflow means policy exceeds dataset support band.
        upper_overflow = torch.clamp(policy_p99 - dataset_p99, min=0.0)
        lower_overflow = torch.clamp(dataset_p1 - policy_p1, min=0.0)

        stats: dict[str, torch.Tensor] = {
            "action_ood/mean_upper_overflow": upper_overflow.abs().mean(),
            "action_ood/mean_lower_overflow": lower_overflow.abs().mean(),
            "action_ood/max_upper_overflow": upper_overflow.max(),
            "action_ood/max_lower_overflow": lower_overflow.max(),
            "action_ood/policy_abs_action_mean": policy_actions.abs().mean(),
            "action_ood/dataset_abs_action_mean": dataset_actions.abs().mean(),
        }

        num_detail_dims = min(4, int(dataset_actions.shape[-1]))
        for dim_idx in range(num_detail_dims):
            stats[f"action_ood/dim{dim_idx}_dataset_p1"] = dataset_p1[dim_idx]
            stats[f"action_ood/dim{dim_idx}_dataset_p50"] = dataset_p50[dim_idx]
            stats[f"action_ood/dim{dim_idx}_dataset_p99"] = dataset_p99[dim_idx]
            stats[f"action_ood/dim{dim_idx}_policy_p1"] = policy_p1[dim_idx]
            stats[f"action_ood/dim{dim_idx}_policy_p50"] = policy_p50[dim_idx]
            stats[f"action_ood/dim{dim_idx}_policy_p99"] = policy_p99[dim_idx]
            stats[f"action_ood/dim{dim_idx}_upper_overflow"] = upper_overflow[dim_idx]
            stats[f"action_ood/dim{dim_idx}_lower_overflow"] = lower_overflow[dim_idx]

        return stats

    def _update_q(
        self,
        data: TensorDict,
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
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        args = self.config
        scaler = self.scaler
        reward_scale = self.reward_scale
        with self._maybe_amp():
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
            dataset_actions = self._to_critic_actions(data["actions"])
            rewards = data["next"]["rewards"]
            rewards = reward_scale *rewards
            dones = data["next"]["dones"].bool()
            # Truncated ends (timeout rows: dones=1 AND truncations=1) are not true
            # terminals: the exporter stores the final pre-reset observation in
            # next.observations, so they can be bootstrapped through. This matches the
            # online FastSAC semantics and is effectively required for random-start
            # (start_at_timestep_zero_prob < 1) datasets, where timeout-as-terminal
            # assigns contradictory targets to the same mid-phase states. Gated by
            # config to keep legacy runs bit-identical.
            if args.bootstrap_truncations:
                truncations = data["next"]["truncations"].bool()
                bootstrap = (truncations | ~dones).float()
            else:
                bootstrap = (~dones).float()
            alpha = self.log_alpha.exp().detach()

            with torch.no_grad():
                discount = args.gamma ** data["next"]["effective_n_steps"]

                # Make shapes safe: [B]
                rewards_ = rewards.view(-1)
                bootstrap_ = bootstrap.view(-1)
                discount_ = discount.view(-1)

                if args.cql_max_target_backup:
                    batch_size = next_observations.shape[0]
                    num_backup_actions = args.cql_max_target_backup_samples

                    # [B, obs_dim] -> [B*K, obs_dim]
                    expanded_next_obs = (
                        next_observations.unsqueeze(1)
                        .expand(batch_size, num_backup_actions, *next_observations.shape[1:])
                        .reshape(batch_size * num_backup_actions, *next_observations.shape[1:])
                    )

                    expanded_next_critic_obs = (
                        next_critic_observations.unsqueeze(1)
                        .expand(batch_size, num_backup_actions, *next_critic_observations.shape[1:])
                        .reshape(batch_size * num_backup_actions, *next_critic_observations.shape[1:])
                    )

                    # Sample K next actions per next state
                    next_actions, next_log_probs = self.actor.get_actions_and_log_probs(
                        expanded_next_obs
                    )

                    next_q1_target, next_q2_target = self.qnet_target(
                        expanded_next_critic_obs,
                        next_actions,
                    )

                    # Safe shape: [B*K] -> [B, K]
                    next_q1_target = next_q1_target.view(batch_size, num_backup_actions)
                    next_q2_target = next_q2_target.view(batch_size, num_backup_actions)
                    next_log_probs = next_log_probs.view(batch_size, num_backup_actions)

                    next_target_min_q_all = torch.minimum(next_q1_target, next_q2_target)

                    # max_i min(Q1,Q2)(s', a_i)
                    next_target_min_q, max_target_indices = next_target_min_q_all.max(dim=1)

                    next_log_probs = next_log_probs.gather(
                        dim=1,
                        index=max_target_indices.unsqueeze(1),
                    ).squeeze(1)
                else:
                    next_actions, next_log_probs = self.actor.get_actions_and_log_probs(
                        next_observations
                    )

                    next_q1_target, next_q2_target = self.qnet_target(
                        next_critic_observations,
                        next_actions,
                    )

                    next_target_min_q = torch.minimum(next_q1_target, next_q2_target).view(-1)
                    next_log_probs = next_log_probs.view(-1)

                if args.backup_entropy:
                    next_v = next_target_min_q - alpha * next_log_probs
                else:
                    next_v = next_target_min_q

                q_target = rewards_ + discount_ * bootstrap_ * next_v

                q_target_raw_p01 = torch.quantile(q_target.float(), 0.01)
                q_target_raw_p99 = torch.quantile(q_target.float(), 0.99)
                q_target_legacy_clip_low_frac = (q_target < -10000.0).float().mean()
                q_target_legacy_clip_high_frac = (q_target > 10000.0).float().mean()
                target_value_max = q_target.max()
                target_value_min = q_target.min()

            q1, q2 = self.qnet(critic_observations, dataset_actions)
            if args.bellman_loss_type == "huber":
                bellman_loss = F.smooth_l1_loss(q1, q_target, beta=args.huber_beta) + F.smooth_l1_loss(
                    q2,
                    q_target,
                    beta=args.huber_beta,
                )
            else:
                bellman_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

            q_data_mean = torch.minimum(q1,q2).mean()
            dr3_raw_loss = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            dr3_loss = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            dr3_active_frac = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            if args.dr3_weight > 0.0:
                with torch.no_grad():
                    dr3_next_actions, _ = self.actor.get_actions_and_log_probs(next_observations)
                q1_features, q2_features = self.qnet.features(critic_observations, dataset_actions)
                next_q1_features, next_q2_features = self.qnet.features(
                    next_critic_observations,
                    dr3_next_actions.detach(),
                )
                if args.dr3_normalize_features:
                    q1_features = F.normalize(q1_features, dim=-1)
                    q2_features = F.normalize(q2_features, dim=-1)
                    next_q1_features = F.normalize(next_q1_features, dim=-1)
                    next_q2_features = F.normalize(next_q2_features, dim=-1)
                dr3_per_sample = 0.5 * (
                    (q1_features * next_q1_features).sum(dim=-1)
                    + (q2_features * next_q2_features).sum(dim=-1)
                )
                dr3_mask = bootstrap_.to(device=dr3_per_sample.device, dtype=dr3_per_sample.dtype)
                dr3_active_count = dr3_mask.sum().clamp_min(1.0)
                dr3_active_frac = dr3_mask.mean()
                dr3_raw_loss = (dr3_per_sample * dr3_mask).sum() / dr3_active_count
                dr3_loss = args.dr3_weight * dr3_raw_loss
            rand_q_mean = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            curr_q_mean = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            next_q_mean = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            curr_logp = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            next_logp = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            random_density = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            with torch.no_grad():
                pi_actions_det = self.actor(observations)[0]
                q1_pi_det, q2_pi_det = self.qnet(critic_observations, pi_actions_det)
                q_pi_minus_q_data = (
                    torch.minimum(q1_pi_det, q2_pi_det) - torch.minimum(q1.detach(), q2.detach())
                ).mean()

            if self._cql_weight > 0.0:
                batch_size = dataset_actions.shape[0]
                num_repeat = self._num_repeat_actions
                num_near = self._num_near_actions

                expanded_obs = observations[:, None, :].expand(batch_size, num_repeat, -1).reshape(
                    batch_size * num_repeat, -1
                )
                expanded_critic_obs = critic_observations[:, None, :].expand(
                    batch_size, num_repeat, -1
                ).reshape(batch_size * num_repeat, -1)
                expanded_next_obs = next_observations[:, None, :].expand(
                    batch_size, num_repeat, -1
                ).reshape(batch_size * num_repeat, -1)

                with torch.no_grad():
                    curr_actions, curr_logp = self.actor.get_actions_and_log_probs(expanded_obs)
                    next_actions_rep, next_logp = self.actor.get_actions_and_log_probs(expanded_next_obs)

                action_scale = self.actor.action_scale.to(device=self.device, dtype=dataset_actions.dtype)
                action_bias = self.actor.action_bias.to(device=self.device, dtype=dataset_actions.dtype)
                rand_actions = torch.empty(
                    batch_size * num_repeat,
                    dataset_actions.shape[-1],
                    device=self.device,
                    dtype=dataset_actions.dtype,
                ).uniform_(-1.0, 1.0)
                if self.config.use_tanh:
                    rand_actions = rand_actions * action_scale + action_bias

                q1_rand, q2_rand = self.qnet(expanded_critic_obs, rand_actions)
                q1_curr, q2_curr = self.qnet(expanded_critic_obs, curr_actions)
                q1_next, q2_next = self.qnet(expanded_critic_obs, next_actions_rep)

                q1_rand = q1_rand.view(batch_size, num_repeat)
                q2_rand = q2_rand.view(batch_size, num_repeat)
                q1_curr = q1_curr.view(batch_size, num_repeat)
                q2_curr = q2_curr.view(batch_size, num_repeat)
                q1_next = q1_next.view(batch_size, num_repeat)
                q2_next = q2_next.view(batch_size, num_repeat)
                curr_logp = curr_logp.view(batch_size, num_repeat)
                next_logp = next_logp.view(batch_size, num_repeat)

                if self.config.use_tanh:
                    random_density = (
                        math.log(0.5) * dataset_actions.shape[-1]
                        - torch.log(action_scale + 1e-6).sum()
                    )
                else:
                    random_density = math.log(0.5) * dataset_actions.shape[-1]
                
                q1_terms = [
                    q1_rand - random_density,
                    q1_curr - curr_logp,
                    q1_next - next_logp,
                ]
                q2_terms = [
                    q2_rand - random_density,
                    q2_curr - curr_logp,
                    q2_next - next_logp,
                ]
                cat_q1 = torch.cat(
                    q1_terms,
                    dim=1,
                )
                cat_q2 = torch.cat(
                    q2_terms,
                    dim=1,
                )

                cql1_loss = (torch.logsumexp(cat_q1 / self._temperature, dim=1) * self._temperature - q1).mean()
                cql2_loss = (torch.logsumexp(cat_q2 / self._temperature, dim=1) * self._temperature - q2).mean()
                cql_gap = 0.5 * (cql1_loss + cql2_loss)
                rand_q_mean = 0.5 * ((q1_rand - random_density).mean() + (q2_rand - random_density).mean())
                curr_q_mean = 0.5 * ((q1_curr).mean() + (q2_curr).mean())
                next_q_mean = 0.5 * ((q1_next).mean() + (q2_next).mean())

                if args.use_lagrange and self.log_cql_alpha is not None:
                    cql_alpha = self.log_cql_alpha.exp().detach().clamp(max=args.cql_lagrange_max)
                    target_gap = torch.tensor(args.cql_target_action_gap, device=self.device, dtype=bellman_loss.dtype)
                    conservative_loss = cql_alpha * self._cql_weight * 0.5*(
                        (cql1_loss - target_gap) + (cql2_loss - target_gap)
                    )
                else:
                    conservative_loss = self._cql_weight *0.5* (cql1_loss + cql2_loss)
            else:
                conservative_loss = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                cql_gap = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            
            q_loss = bellman_loss + conservative_loss + dr3_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(q_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(self.qnet)

        scaler.unscale_(self.q_optimizer)
        if args.max_grad_norm > 0:
            q_grad_norm = torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), args.max_grad_norm)
        else:
            q_grad_norm = torch.tensor(0.0, device=self.device)

        scaler.step(self.q_optimizer)
        scaler.update()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.config.use_autotune:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                _, log_probs = self.actor.get_actions_and_log_probs(observations)
                alpha_loss = (-self.log_alpha.exp() * (log_probs.detach() + self.target_entropy)).mean()
            scaler.scale(alpha_loss).backward()

            if self.is_multi_gpu and self.log_alpha.grad is not None:
                torch.distributed.all_reduce(self.log_alpha.grad.data, op=torch.distributed.ReduceOp.SUM)
                self.log_alpha.grad.data.copy_(self.log_alpha.grad.data / self.gpu_world_size)

            scaler.unscale_(self.alpha_optimizer)
            scaler.step(self.alpha_optimizer)
            scaler.update()

        return (
            rewards.mean().detach(),
            q_grad_norm.detach(),
            q_loss.detach(),
            target_value_max.detach(),
            target_value_min.detach(),
            q_target_raw_p01.detach(),
            q_target_raw_p99.detach(),
            q_target_legacy_clip_low_frac.detach(),
            q_target_legacy_clip_high_frac.detach(),
            dr3_raw_loss.detach(),
            dr3_loss.detach(),
            dr3_active_frac.detach(),
            alpha_loss.detach(),
            conservative_loss.detach(),
            bellman_loss.detach(),
            cql_gap.detach(),
            q_data_mean.detach(),
            q_pi_minus_q_data.detach(),
            rand_q_mean.detach(),
            curr_q_mean.detach(),
            next_q_mean.detach(),
            curr_logp.mean().detach(),
            next_logp.mean().detach(),
            random_density,
        )

    def _update_cql_lagrange(self, cql_gap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update CQL Lagrange multiplier outside compiled critic step to avoid double-backward in torch.compile."""
        if (
            not self.config.use_lagrange
            or self._cql_weight <= 0.0
            or self.log_cql_alpha is None
            or self.cql_alpha_optimizer is None
        ):
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        target_gap = torch.tensor(
            self.config.cql_target_action_gap,
            device=self.device,
            dtype=cql_gap.dtype,
        )
        cql_alpha = self.log_cql_alpha.exp().clamp(max=self.config.cql_lagrange_max)
        # Equivalent to:
        # -0.5 * alpha * w * ((diff1 - tau) + (diff2 - tau))
        # because cql_gap = 0.5 * (diff1 + diff2)
        cql_alpha_loss = -self._cql_weight * cql_alpha * (cql_gap.detach() - target_gap)

        self.cql_alpha_optimizer.zero_grad(set_to_none=True)
        cql_alpha_loss.backward()

        if self.is_multi_gpu and self.log_cql_alpha.grad is not None:
            torch.distributed.all_reduce(self.log_cql_alpha.grad.data, op=torch.distributed.ReduceOp.SUM)
            self.log_cql_alpha.grad.data.copy_(self.log_cql_alpha.grad.data / self.gpu_world_size)

        self.cql_alpha_optimizer.step()
        with torch.no_grad():
            self.log_cql_alpha.data.clamp_(max=math.log(self.config.cql_lagrange_max))
            cql_alpha_value = self.log_cql_alpha.exp().clamp(max=self.config.cql_lagrange_max)
        return cql_alpha_value.detach(), cql_alpha_loss.detach()

    def _update_actor(
        self,
        data: TensorDict,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        args = self.config
        scaler = self.scaler

        with self._maybe_amp():
            actor_observations = data["observations"]  # [B, actor_obs_dim]
            critic_observations = data["critic_observations"]  # [B, critic_obs_dim]

            _, _, log_std = self.actor(actor_observations)  # _, _, [B, act_dim]
            actions, log_probs = self.actor.get_actions_and_log_probs(actor_observations)  # [B, act_dim], [B]
            with torch.no_grad():
                action_std = log_std.exp().mean()
                policy_entropy = -log_probs.mean()

            q1_pi, q2_pi = self.qnet(critic_observations, actions)
            qf_value = torch.minimum(q1_pi, q2_pi)
            actor_rl_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_rl_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(self.actor)

        scaler.unscale_(self.actor_optimizer)
        if args.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), args.max_grad_norm)
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)

        scaler.step(self.actor_optimizer)
        scaler.update()

        return (
            actor_grad_norm.detach(),
            actor_rl_loss.detach(),
            policy_entropy.detach(),
            action_std.detach(),
        )

    def _sample_offline_batch(
        self,
        batch_size: int,
        normalize_obs,
        normalize_critic_obs,
    ) -> TensorDict:
        if self._offline_gpu_cache is not None:
            batch = self._offline_gpu_cache.sample(batch_size=batch_size)
        else:
            if self._offline_shuffle_buffer is None:
                raise RuntimeError("Offline shuffle buffer is not initialized. Call setup() before offline_learn().")
            batch = self._offline_shuffle_buffer.sample(batch_size=batch_size)
            batch = batch_to_device(batch, device=self.device, non_blocking=True)

        if self.config.use_symmetry:
            augmented_batch: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {"next": {}}
            augmented_batch["observations"] = self.symmetry_utils.augment_observations(
                obs=batch["observations"],
                env=self.env,
                obs_list=self.config.actor_obs_keys,
            )
            augmented_batch["actions"] = self.symmetry_utils.augment_actions(actions=batch["actions"])
            assert isinstance(augmented_batch["next"], dict)
            augmented_batch["next"]["observations"] = self.symmetry_utils.augment_observations(
                obs=batch["next"]["observations"],
                env=self.env,
                obs_list=self.config.actor_obs_keys,
            )
            augmented_batch["critic_observations"] = self.symmetry_utils.augment_observations(
                obs=batch["critic_observations"],
                env=self.env,
                obs_list=self.config.critic_obs_keys,
            )
            augmented_batch["next"]["critic_observations"] = self.symmetry_utils.augment_observations(
                obs=batch["next"]["critic_observations"],
                env=self.env,
                obs_list=self.config.critic_obs_keys,
            )
            observations_tensor = augmented_batch["observations"]
            assert isinstance(observations_tensor, torch.Tensor)
            num_aug = int(observations_tensor.shape[0] / batch["next"]["rewards"].shape[0])
            augmented_batch["next"]["rewards"] = batch["next"]["rewards"].repeat(num_aug)  # type: ignore[index]
            augmented_batch["next"]["dones"] = batch["next"]["dones"].repeat(num_aug)  # type: ignore[index]
            augmented_batch["next"]["truncations"] = batch["next"]["truncations"].repeat(num_aug)  # type: ignore[index]
            augmented_batch["next"]["effective_n_steps"] = batch["next"]["effective_n_steps"].repeat(
                num_aug
            )  # type: ignore[index]
            batch = augmented_batch

        batch = apply_observation_normalization(batch, normalize_obs, normalize_critic_obs)
        effective_batch_size = int(batch["observations"].shape[0])
        next_batch = {
            "observations": batch["next"]["observations"],
            "critic_observations": batch["next"]["critic_observations"],
            "rewards": batch["next"]["rewards"],
            "truncations": batch["next"]["truncations"].to(torch.long),
            "dones": batch["next"]["dones"].to(torch.long),
            "effective_n_steps": batch["next"]["effective_n_steps"],
        }
        return TensorDict(
            {
                "observations": batch["observations"],
                "actions": batch["actions"],
                "critic_observations": batch["critic_observations"],
                "next": next_batch,
            },
            batch_size=effective_batch_size,
            device=self.device,
        )

    def load(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return

        torch_checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        checkpoint_action_mode = torch_checkpoint.get("action_space_mode", "legacy")
        compatible_checkpoint_action_mode = (
            "env_scaled_action_training_v1" if checkpoint_action_mode == "legacy" else checkpoint_action_mode
        )
        expected_action_mode = getattr(self, "action_space_mode", "env_scaled_action_training_v1")
        if compatible_checkpoint_action_mode != expected_action_mode:
            logger.warning(
                "Loading a checkpoint with different CQL action semantics "
                f"(checkpoint action_space_mode={checkpoint_action_mode}, current={expected_action_mode})."
            )

        checkpoint_args = torch_checkpoint.get("args", {})
        checkpoint_obs_norm = checkpoint_args.get("obs_normalization")
        if checkpoint_obs_norm is not None and bool(checkpoint_obs_norm) != bool(self.obs_normalization):
            raise RuntimeError(
                "Checkpoint/config mismatch for observation normalization: "
                f"checkpoint obs_normalization={checkpoint_obs_norm}, "
                f"current config obs_normalization={self.obs_normalization}."
            )

        required_keys = ("actor_state_dict", "qnet_state_dict", "log_alpha")
        missing_required = [k for k in required_keys if k not in torch_checkpoint]
        if missing_required:
            raise RuntimeError(
                f"Checkpoint missing required CQL keys: {missing_required}. "
                "Expected a scalar-CQL checkpoint."
            )

        self.actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        self._sync_actor_action_space_buffers()
        self.qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])

        if "qnet_target_state_dict" in torch_checkpoint:
            self.qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        else:
            self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.log_alpha.data.copy_(torch_checkpoint["log_alpha"].to(self.device))

        obs_norm_state = torch_checkpoint.get("obs_normalizer_state")
        critic_obs_norm_state = torch_checkpoint.get("critic_obs_normalizer_state")

        if self.obs_normalization:
            if not isinstance(obs_norm_state, dict) or not obs_norm_state:
                raise RuntimeError("Checkpoint missing valid obs_normalizer_state while obs normalization is enabled.")
            if not isinstance(critic_obs_norm_state, dict) or not critic_obs_norm_state:
                raise RuntimeError(
                    "Checkpoint missing valid critic_obs_normalizer_state while obs normalization is enabled."
                )

        self.obs_normalizer.load_state_dict(obs_norm_state if isinstance(obs_norm_state, dict) else {})
        self.critic_obs_normalizer.load_state_dict(
            critic_obs_norm_state if isinstance(critic_obs_norm_state, dict) else {}
        )

        if "actor_optimizer_state_dict" in torch_checkpoint:
            self.actor_optimizer.load_state_dict(torch_checkpoint["actor_optimizer_state_dict"])
        if "q_optimizer_state_dict" in torch_checkpoint:
            self.q_optimizer.load_state_dict(torch_checkpoint["q_optimizer_state_dict"])
        if "alpha_optimizer_state_dict" in torch_checkpoint:
            self.alpha_optimizer.load_state_dict(torch_checkpoint["alpha_optimizer_state_dict"])
        if (
            self.config.use_lagrange
            and self.log_cql_alpha is not None
            and "cql_log_alpha" in torch_checkpoint
        ):
            self.log_cql_alpha.data.copy_(torch_checkpoint["cql_log_alpha"].to(self.device))
        if (
            self.config.use_lagrange
            and self.cql_alpha_optimizer is not None
            and "cql_alpha_optimizer_state_dict" in torch_checkpoint
        ):
            self.cql_alpha_optimizer.load_state_dict(torch_checkpoint["cql_alpha_optimizer_state_dict"])
        if "grad_scaler_state_dict" in torch_checkpoint and torch_checkpoint["grad_scaler_state_dict"] is not None:
            self.scaler.load_state_dict(torch_checkpoint["grad_scaler_state_dict"])

        self.global_step = int(torch_checkpoint.get("global_step", 0))
        self._restore_env_state(torch_checkpoint.get("env_state"))

    def __del__(self) -> None:
        shuffle_buffer = getattr(self, "_offline_shuffle_buffer", None)
        if shuffle_buffer is not None:
            shuffle_buffer.close()
        offline_gpu_cache = getattr(self, "_offline_gpu_cache", None)
        if offline_gpu_cache is not None:
            offline_gpu_cache.close()

    def offline_learn(self, max_steps: int | None = None) -> None:
        args = self.config

        if max_steps is None:
            max_steps = args.eval_interval if args.eval_interval > 0 else args.num_learning_iterations - self.global_step

        if max_steps <= 0:
            return

        target_step = min(self.global_step + max_steps, args.num_learning_iterations)
        if target_step <= self.global_step:
            return

        if args.compile:
            if not hasattr(self, "_compiled_update_q"):
                self._compiled_update_q = torch.compile(self._update_q)
                self._compiled_update_actor = torch.compile(self._update_actor)
            update_q = self._compiled_update_q
            update_actor = self._compiled_update_actor
        else:
            update_q = self._update_q
            update_actor = self._update_actor

        if self.env.num_envs > 1 and self.is_main_process:
            logger.info(
                "Offline CQL gradient updates sample only the fixed dataset. "
                f"Current num_envs={self.env.num_envs} is used for vectorized evaluation."
            )

        normalize_obs = self.obs_normalizer.forward
        normalize_critic_obs = self.critic_obs_normalizer.forward

        pbar = tqdm.tqdm(total=max(target_step - self.global_step, 0), initial=0, leave=False)
        if self._offline_gpu_cache is not None:
            logger.info(
                f"Sampling offline dataset from GPU cache loaded from '{self._offline_dataset_path}' "
                f"with {self._offline_num_samples} samples."
            )
        else:
            logger.info(
                f"Streaming offline dataset from '{self._offline_dataset_path}' "
                f"with {self._offline_num_samples} samples."
            )
        while self.global_step < target_step:
            self.global_step += 1

            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            batch_size = max(args.batch_size // self.gpu_world_size, 1)
            with self.logging_helper.record_learn_time():
                for _ in range(args.num_updates):
                    data = self._sample_offline_batch(
                        batch_size=batch_size,
                        normalize_obs=normalize_obs,
                        normalize_critic_obs=normalize_critic_obs,
                    )
                    (
                        reward_mean,
                        q_grad_norm,
                        q_loss,
                        q_target_max,
                        q_target_min,
                        q_target_raw_p01,
                        q_target_raw_p99,
                        q_target_legacy_clip_low_frac,
                        q_target_legacy_clip_high_frac,
                        dr3_raw_loss,
                        dr3_loss,
                        dr3_active_frac,
                        alpha_loss,
                        conservative_loss,
                        bellman_loss,
                        cql_gap,
                        q_data_mean,
                        q_pi_minus_q_data,
                        rand_q,
                        curr_q,
                        next_q,
                        curr_logp,
                        next_logp,
                        random_density,
                    ) = update_q(data)

                    cql_alpha_value, cql_lagrange_loss = self._update_cql_lagrange(cql_gap)

                    self._critic_update_step += 1
                    is_actor_warmup = self.global_step <= args.actor_warmup_steps
                    is_actor_update_step = (not is_actor_warmup) and (
                        self._critic_update_step % args.policy_frequency == 0
                    )
                    if is_actor_update_step:
                        (
                            actor_grad_norm,
                            actor_loss,
                            policy_entropy,
                            action_std,
                        ) = update_actor(data)
                    else:
                        actor_grad_norm = torch.tensor(0.0, device=self.device)
                        actor_loss = torch.tensor(0.0, device=self.device)
                        policy_entropy = torch.tensor(0.0, device=self.device)
                        action_std = torch.tensor(0.0, device=self.device)

                    self._soft_update_q_target()

                    action_ood_stats = self._compute_action_ood_stats(data)
                    self.training_metrics.add(
                        {
                            "random_q": rand_q,
                            "current_q": curr_q,
                            "next_q": next_q,
                            "buffer_rewards": reward_mean,
                            "q_grad_norm": q_grad_norm,
                            "q_loss": q_loss,
                            "q_target_max": q_target_max,
                            "q_target_min": q_target_min,
                            "q_target_raw_p01": q_target_raw_p01,
                            "q_target_raw_p99": q_target_raw_p99,
                            "q_target_legacy_clip_low_frac": q_target_legacy_clip_low_frac,
                            "q_target_legacy_clip_high_frac": q_target_legacy_clip_high_frac,
                            "dr3_raw_loss": dr3_raw_loss,
                            "dr3_loss": dr3_loss,
                            "dr3_active_frac": dr3_active_frac,
                            "alpha_loss": alpha_loss,
                            "alpha_value": self.log_alpha.exp().detach().mean(),
                            "actor_grad_norm": actor_grad_norm,
                            "actor_loss": actor_loss,
                            "policy_entropy": policy_entropy,
                            "action_std": action_std,
                            "cql_conservative_loss": conservative_loss,
                            "cql_bellman_loss": bellman_loss,
                            "cql_gap": cql_gap,
                            "q_data_mean": q_data_mean,
                            "q_pi_minus_q_data": q_pi_minus_q_data,
                            "cql_alpha_value": cql_alpha_value,
                            "cql_lagrange_loss": cql_lagrange_loss,
                            "cql_target_action_gap": torch.tensor(
                                args.cql_target_action_gap if args.use_lagrange else 0.0,
                                device=self.device,
                            ),
                            "is_actor_warmup": float(is_actor_warmup),
                            "is_actor_update_step": float(is_actor_update_step),
                            **action_ood_stats,
                            "current_logprob": curr_logp,
                            "next_logprob": next_logp,
                            "random_density": random_density,
                        }

                    )

            should_log = (self.global_step % args.logging_interval == 0) or (self.global_step <= 10)
            if should_log:
                with torch.no_grad():
                    accumulated_metrics = self.training_metrics.mean_and_clear()
                    loss_dict = {
                        key: (value.item() if isinstance(value, torch.Tensor) else float(value))
                        for key, value in accumulated_metrics.items()
                    }
                self.logging_helper.post_epoch_logging(it=self.global_step, loss_dict=loss_dict, extra_log_dicts={})

            if args.save_interval > 0 and self.global_step % args.save_interval == 0:
                if self.is_main_process:
                    logger.info(f"Saving CQL model at global step {self.global_step}")
                    self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
                    self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))

            pbar.update(1)
        pbar.close()

        if self.is_main_process and self.global_step >= args.num_learning_iterations:
            self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
            self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))

    def save(self, path: str) -> None:  # type: ignore[override]
        env_state = self._collect_env_state()
        metadata = self._checkpoint_metadata(iteration=self.global_step)
        metadata["action_space_mode"] = self.action_space_mode
        save_params(
            self.global_step,
            self.actor,
            self.qnet,
            self.qnet_target,
            self.obs_normalizer,
            self.critic_obs_normalizer,
            self.log_alpha,
            self.actor_optimizer,
            self.q_optimizer,
            self.alpha_optimizer,
            self.scaler,
            self.config,
            path,
            save_fn=self.logging_helper.save_checkpoint_artifact,
            env_state=env_state or None,
            metadata=metadata,
            cql_log_alpha=self.log_cql_alpha if self.config.use_lagrange else None,
            cql_alpha_optimizer=self.cql_alpha_optimizer if self.config.use_lagrange else None,
        )

    @torch.no_grad()
    def get_example_obs(self):
        obs_dict = self.unwrapped_env.reset_all()
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return {
            "actor_obs": torch.cat([obs_dict[k] for k in self.config.actor_obs_keys], dim=1),
            "critic_obs": torch.cat([obs_dict[k] for k in self.config.critic_obs_keys], dim=1),
        }

    def get_inference_policy(self, device: str | None = None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        device = device or self.device
        policy = self.actor.to(device)
        obs_normalizer = self.obs_normalizer.to(device)
        policy.eval()
        obs_normalizer.eval()

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            if self.obs_normalization:
                normalized_obs = obs_normalizer(obs["actor_obs"], update=False)
            else:
                normalized_obs = obs["actor_obs"]
            return self._to_env_actions(policy(normalized_obs)[0])

        return policy_fn

    def _eval_termination_reason_flags(self, infos: dict[str, Any], num_envs: int) -> dict[str, torch.Tensor]:
        raw_reasons = infos.get("termination_reasons", {})
        if not isinstance(raw_reasons, dict):
            raw_reasons = {}

        reason_flags: dict[str, torch.Tensor] = {}
        for reason, value in raw_reasons.items():
            if isinstance(value, torch.Tensor):
                reason_flags[str(reason)] = value.to(device=self.device, dtype=torch.bool)

        if "timeout" not in reason_flags:
            time_outs = infos.get("time_outs")
            if isinstance(time_outs, torch.Tensor):
                reason_flags["timeout"] = time_outs.to(device=self.device, dtype=torch.bool)
            else:
                reason_flags["timeout"] = torch.zeros(num_envs, device=self.device, dtype=torch.bool)

        return reason_flags

    def _eval_stop_reason_for_env(
        self,
        reason_flags: dict[str, torch.Tensor],
        env_idx: int,
        fallback: str | None = None,
    ) -> str | None:
        for reason in ("bad_tracking", "motion_ends", "timeout"):
            flags = reason_flags.get(reason)
            if flags is not None and bool(flags[env_idx].item()):
                return reason
        for reason, flags in reason_flags.items():
            if reason in {"bad_tracking", "motion_ends", "timeout"}:
                continue
            if bool(flags[env_idx].item()):
                return reason
        return fallback

    @property
    def actor_onnx_wrapper(self):
        actor = copy.deepcopy(self.actor).to("cpu")
        obs_normalizer = copy.deepcopy(self.obs_normalizer).to("cpu")

        class ActorWrapper(nn.Module):
            def __init__(
                self,
                actor,
                obs_normalizer,
            ):
                super().__init__()
                self.actor = actor
                self.obs_normalizer = obs_normalizer

            def forward(self, actor_obs):
                if self.obs_normalizer is not None:
                    normalized_obs = self.obs_normalizer(actor_obs, update=False)
                else:
                    normalized_obs = actor_obs
                return self.actor(normalized_obs)[0]

        return ActorWrapper(
            actor,
            obs_normalizer if self.obs_normalization else None,
        )

    def export(self, onnx_file_path: str) -> None:
        was_training = self.actor.training

        self.actor.eval()
        if self.obs_normalization:
            self.obs_normalizer.eval()

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

        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.unwrapped_env)
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)

        metadata = {
            "dof_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "command_ranges": cmd_ranges,
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
            "action_space_mode": self.action_space_mode,
        }
        metadata.update(self._checkpoint_metadata(iteration=self.global_step))

        attach_onnx_metadata(
            onnx_path=onnx_file_path,
            metadata=metadata,
        )
        self.logging_helper.save_to_wandb(onnx_file_path)

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
            actions = self._to_env_actions(self.actor(normalized_obs)[0])
            obs, _, _, _ = self.env.step(actions)

    @torch.no_grad()
    def evaluate_one_episode(
        self,
        max_eval_steps: int | None = None,
        use_early_termination: bool = False,
    ):
        self.env.set_is_evaluating()
        was_training = self.actor.training

        self.actor.eval()
        if self.obs_normalization:
            self.obs_normalizer.eval()

        obs = self.env.reset()
        eval_env_idx = 0
        episode_return = 0.0
        episode_length = 0
        stop_reason = None
        bad_tracking_details: list[str] = []

        for t in itertools.count():
            if max_eval_steps is not None and t >= max_eval_steps:
                stop_reason = "max_eval_steps"
                break

            if self.obs_normalization:
                normalized_obs = self.obs_normalizer(obs, update=False)
            else:
                normalized_obs = obs

            actions = self._to_env_actions(self.actor(normalized_obs)[0])
            obs, rewards, dones, infos = self.env.step(actions)

            episode_return += float(rewards[eval_env_idx].item())
            episode_length += 1

            reason_flags = self._eval_termination_reason_flags(infos, int(self.env.num_envs))
            specific_stop_reason = self._eval_stop_reason_for_env(reason_flags, eval_env_idx)
            if specific_stop_reason is not None:
                stop_reason = specific_stop_reason
                if stop_reason == "bad_tracking":
                    bad_tracking_details = self._eval_bad_tracking_details_for_env(reason_flags, eval_env_idx)
                break

            if use_early_termination and "early_termination" in infos:
                if bool(infos["early_termination"][eval_env_idx].item()):
                    stop_reason = "early_termination"
                    break

            if bool(dones[eval_env_idx].item()):
                stop_reason = "done"
                break

        if was_training:
            self.actor.train()
            if self.obs_normalization:
                self.obs_normalizer.train()

        if hasattr(self.env, "set_is_training"):
            self.env.set_is_training()

        return {
            "episode_return": float(episode_return),
            "episode_length": int(episode_length),
            "stop_reason": stop_reason,
            "bad_tracking_details": bad_tracking_details,
        }

    @torch.no_grad()
    def evaluate_vectorized_episodes(
        self,
        max_eval_steps: int | None = None,
        use_early_termination: bool = False,
    ) -> list[dict[str, float | int | str | None]]:
        self.env.set_is_evaluating()
        was_training = self.actor.training

        self.actor.eval()
        if self.obs_normalization:
            self.obs_normalizer.eval()

        obs = self.env.reset()
        num_envs = int(self.env.num_envs)
        episode_returns = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        episode_lengths = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        finished = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        stop_reasons: list[str | None] = [None] * num_envs
        bad_tracking_details: list[list[str]] = [[] for _ in range(num_envs)]

        def _info_bool(name: str, infos: dict[str, Any]) -> torch.Tensor:
            value = infos.get(name)
            if isinstance(value, torch.Tensor):
                return value.to(device=self.device, dtype=torch.bool)
            return torch.zeros(num_envs, device=self.device, dtype=torch.bool)

        for step_idx in itertools.count():
            if max_eval_steps is not None and step_idx >= max_eval_steps:
                unfinished_indices = torch.nonzero(~finished, as_tuple=False).flatten().detach().cpu().tolist()
                for env_idx in unfinished_indices:
                    stop_reasons[int(env_idx)] = "max_eval_steps"
                break

            if self.obs_normalization:
                normalized_obs = self.obs_normalizer(obs, update=False)
            else:
                normalized_obs = obs

            actions = self._to_env_actions(self.actor(normalized_obs)[0])
            obs, rewards, dones, infos = self.env.step(actions)

            active = ~finished
            step_rewards = rewards.to(device=self.device, dtype=torch.float32)
            episode_returns += torch.where(active, step_rewards, torch.zeros_like(episode_returns))
            episode_lengths += active.to(torch.long)

            reason_flags = self._eval_termination_reason_flags(infos, num_envs)
            timeout_flags = reason_flags.get("timeout", _info_bool("time_outs", infos))
            early_flags = _info_bool("early_termination", infos) if use_early_termination else torch.zeros_like(finished)
            done_flags = dones.to(device=self.device, dtype=torch.bool)

            newly_timed_out = active & timeout_flags
            newly_early_terminated = active & ~newly_timed_out & early_flags
            newly_done = active & ~newly_timed_out & ~newly_early_terminated & done_flags

            for env_idx in torch.nonzero(newly_timed_out, as_tuple=False).flatten().detach().cpu().tolist():
                env_idx_int = int(env_idx)
                stop_reason = self._eval_stop_reason_for_env(reason_flags, env_idx_int, "timeout")
                stop_reasons[env_idx_int] = stop_reason
                if stop_reason == "bad_tracking":
                    bad_tracking_details[env_idx_int] = self._eval_bad_tracking_details_for_env(
                        reason_flags, env_idx_int
                    )
            for env_idx in torch.nonzero(newly_early_terminated, as_tuple=False).flatten().detach().cpu().tolist():
                stop_reasons[int(env_idx)] = "early_termination"
            for env_idx in torch.nonzero(newly_done, as_tuple=False).flatten().detach().cpu().tolist():
                env_idx_int = int(env_idx)
                stop_reason = self._eval_stop_reason_for_env(reason_flags, env_idx_int, "done")
                stop_reasons[env_idx_int] = stop_reason
                if stop_reason == "bad_tracking":
                    bad_tracking_details[env_idx_int] = self._eval_bad_tracking_details_for_env(
                        reason_flags, env_idx_int
                    )

            finished |= newly_timed_out | newly_early_terminated | newly_done
            if bool(finished.all().item()):
                break

        if was_training:
            self.actor.train()
            if self.obs_normalization:
                self.obs_normalizer.train()

        if hasattr(self.env, "set_is_training"):
            self.env.set_is_training()

        returns = episode_returns.detach().cpu().tolist()
        lengths = episode_lengths.detach().cpu().tolist()
        return [
            {
                "episode_return": float(returns[env_idx]),
                "episode_length": int(lengths[env_idx]),
                "stop_reason": stop_reasons[env_idx],
                "bad_tracking_details": bad_tracking_details[env_idx],
            }
            for env_idx in range(num_envs)
        ]
