from __future__ import annotations

"""Offline-only SAC agent for fast_sac-format HDF5 datasets.

Implementation note (requested summary):
- Reused from `fast_sac`:
  - Network architecture and action semantics: `Actor/CNNActor/Critic/CNNCritic`
    with tanh-Gaussian policy and per-joint env/scaled action boundaries.
  - Actor/critic observation-key split and flattened index mapping.
  - Observation normalization (`EmpiricalNormalization`), AMP/multi-GPU gradient
    handling, checkpoint/export/inference/evaluation structure.
  - Distributional critic backup style (categorical projection on fixed support).
- Modified for offline-only training:
  - Removed online rollout and replay-buffer collection completely.
  - Added HDF5 fixed-dataset sampling through
    `HDF5BlockReader/RAMShuffleBuffer/GPUTransitionCache`.
  - Kept pure SAC losses only (critic Bellman fit + actor entropy-regularized
    objective + alpha autotune), with NO CQL/BC/risk/lagrange/tail penalty terms.

Expected HDF5 semantics (fast_sac export compatible):
- Required keys:
  `observations`, `actions`, `critic_observations`, `next_observations`,
  `next_critic_observations`, `rewards`, `truncations`, `dones`.
- Optional next-step metadata keys (flattened form):
  `next_done_bad_tracking`, `next_done_motion_ends`, `next_done_timeout`,
  `next_episode_step`, `next_global_step`, `next_err_root_pos`,
  `next_err_root_ori`, `next_err_body_pos_max`, `next_err_body_pos_mean`,
  `next_err_object_pos`, `next_err_object_ori`.
- Important: this agent trusts dataset-provided `next_observations` as-is
  (including terminal final pre-reset observations exported by fast_sac).
  No terminal next-observation reinterpretation is performed in the loader path.

Distributional critic reuse rationale:
- Reuse is direct because `fast_sac` critic already exposes projection and
  expectation utilities used by SAC backups.
- Actor loss uses expectation over support: `Q(s,a)=E[Z(s,a)]` from softmax
  probabilities and critic support atoms.
- Bellman target is built as entropy-regularized target return and projected to
  support via critic `projection(...)`, then fitted with cross-entropy.
"""

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
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.offline_sac.offline_sac import Actor, CNNActor, CNNCritic, Critic
from holosoma.agents.offline_sac.offline_sac_utils import EmpiricalNormalization, save_params
from holosoma.config_types.algo import OfflineSACConfig
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


class OfflineSACEnv:
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


class OfflineSACAgent(BaseAlgo):
    config: OfflineSACConfig
    env: OfflineSACEnv  # type: ignore[assignment]
    actor: Actor
    qnet: Critic
    qnet_target: Critic

    def __init__(
        self,
        env: BaseTask,
        config: OfflineSACConfig,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict | None = None,
    ):
        wrapped_env = OfflineSACEnv(env, config.actor_obs_keys, config.critic_obs_keys)
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
        self._offline_dataset_path = Path(config.offline_dataset_path)
        self._offline_dataset_reader: HDF5BlockReader | None = None
        self._offline_shuffle_buffer: RAMShuffleBuffer | None = None
        self._offline_gpu_cache: GPUTransitionCache | None = None
        self._offline_num_samples = 0
        self._critic_update_step = 0

        if config.gamma <= 0.0 or config.gamma > 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {config.gamma}")
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
        logger.info("Setting up offline distributional SAC")

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
            logger.info(f"Configured offline GPU transition cache with {self._offline_num_samples} samples.")
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

        self.obs_normalization = args.obs_normalization
        if self.obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(shape=actor_obs_dim, device=device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(shape=critic_obs_dim, device=device)
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        n_act = self.env.robot_config.actions_dim
        action_scale = env._action_boundaries if args.use_tanh else torch.ones(n_act, device=device)
        action_bias = torch.zeros(n_act, device=device)

        if args.use_cnn_encoder:
            actor_mlp_obs_keys = [k for k in args.actor_obs_keys if k != args.encoder_obs_key]
            critic_mlp_obs_keys = [k for k in args.critic_obs_keys if k != args.encoder_obs_key]
            actor_cls = CNNActor
            critic_cls = CNNCritic
        else:
            actor_mlp_obs_keys = list(args.actor_obs_keys)
            critic_mlp_obs_keys = list(args.critic_obs_keys)
            actor_cls = Actor
            critic_cls = Critic

        self.actor = actor_cls(
            obs_indices=self.actor_obs_indices,
            obs_keys=actor_mlp_obs_keys,
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

        self.qnet = critic_cls(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_mlp_obs_keys,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
            device=device,
        )
        self.qnet_target = critic_cls(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_mlp_obs_keys,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
            device=device,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.log_alpha = torch.tensor([math.log(args.alpha_init)], requires_grad=True, device=device)
        self.target_entropy = -float(n_act) * float(args.target_entropy_ratio)
        self.policy = self.actor.explore

        self.q_optimizer = optim.AdamW(
            self.qnet.parameters(),
            lr=args.critic_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=args.actor_learning_rate,
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

        if args.use_symmetry:
            self.symmetry_utils = SymmetryUtils(env._env)

        logger.info(
            "Offline SAC dims: "
            f"actor_obs_dim={actor_obs_dim}, critic_obs_dim={critic_obs_dim}, n_act={n_act}, "
            f"dataset_samples={self._offline_num_samples}"
        )

        if self.is_multi_gpu:
            self._synchronize_model_parameters()

    @contextmanager
    def _maybe_amp(self):
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        is_cuda = str(self.device).startswith("cuda")
        device_type = "cuda" if is_cuda else "cpu"
        with autocast(device_type=device_type, dtype=amp_dtype, enabled=self.config.amp and is_cuda):
            yield

    def _synchronize_model_parameters(self) -> None:
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)
        for param in self.qnet.parameters():
            torch.distributed.broadcast(param.data, src=0)
        torch.distributed.broadcast(self.log_alpha.data, src=0)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        logger.info(f"Synchronized offline SAC model parameters across {self.gpu_world_size} GPUs")

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

    def _update_critic(
        self,
        data: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        args = self.config
        scaler = self.scaler

        with self._maybe_amp():
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()
            discount = args.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                next_actions, next_log_probs = self.actor.get_actions_and_log_probs(next_observations)
                target_distributions = self.qnet_target.projection(
                    next_critic_observations,
                    next_actions,
                    rewards - discount * bootstrap * self.log_alpha.exp() * next_log_probs,
                    bootstrap,
                    discount,
                )
                target_values = self.qnet_target.get_value(target_distributions)
                target_value_max = target_values.max()
                target_value_min = target_values.min()

            q_outputs = self.qnet(critic_observations, actions)
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            q_loss = critic_losses.mean(dim=1).sum(dim=0)

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

        return (
            rewards.mean().detach(),
            q_grad_norm.detach(),
            q_loss.detach(),
            target_value_max.detach(),
            target_value_min.detach(),
        )

    def _update_actor(
        self,
        data: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        args = self.config
        scaler = self.scaler

        with self._maybe_amp():
            actor_observations = data["observations"]
            critic_observations = data["critic_observations"]

            actions, log_probs = self.actor.get_actions_and_log_probs(actor_observations)
            with torch.no_grad():
                _, _, log_std = self.actor(actor_observations)
                action_std = log_std.exp().mean()
                policy_entropy = -log_probs.mean()

            q_outputs = self.qnet(critic_observations, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = self.qnet.get_value(q_probs)
            qf_value = q_values.mean(dim=0)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()

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
            actor_loss.detach(),
            policy_entropy.detach(),
            action_std.detach(),
        )

    def _update_alpha(self, data: TensorDict) -> torch.Tensor:
        if not self.config.use_autotune:
            return torch.tensor(0.0, device=self.device)

        scaler = self.scaler
        with self._maybe_amp():
            _, log_probs = self.actor.get_actions_and_log_probs(data["observations"])
            alpha_loss = (-self.log_alpha.exp() * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad(set_to_none=True)
        scaler.scale(alpha_loss).backward()

        if self.is_multi_gpu and self.log_alpha.grad is not None:
            torch.distributed.all_reduce(self.log_alpha.grad.data, op=torch.distributed.ReduceOp.SUM)
            self.log_alpha.grad.data.copy_(self.log_alpha.grad.data / self.gpu_world_size)

        scaler.unscale_(self.alpha_optimizer)
        scaler.step(self.alpha_optimizer)
        scaler.update()

        return alpha_loss.detach()

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
        return TensorDict(
            {
                "observations": batch["observations"],
                "actions": batch["actions"],
                "critic_observations": batch["critic_observations"],
                "next": {
                    "observations": batch["next"]["observations"],
                    "critic_observations": batch["next"]["critic_observations"],
                    "rewards": batch["next"]["rewards"],
                    "truncations": batch["next"]["truncations"].to(torch.long),
                    "dones": batch["next"]["dones"].to(torch.long),
                    "effective_n_steps": batch["next"]["effective_n_steps"].to(torch.long),
                },
            },
            batch_size=effective_batch_size,
            device=self.device,
        )

    def load(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return

        torch_checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        checkpoint_action_mode = torch_checkpoint.get("action_space_mode", "legacy")
        if checkpoint_action_mode != "env_scaled_action_training_v1":
            logger.warning(
                "Loading a checkpoint with different action semantics "
                f"(action_space_mode={checkpoint_action_mode})."
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
                f"Checkpoint missing required offline SAC keys: {missing_required}. "
                "Expected an OfflineSAC checkpoint."
            )

        self.actor.load_state_dict(torch_checkpoint["actor_state_dict"])
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
            if args.eval_interval > 0:
                max_steps = args.eval_interval
            else:
                max_steps = args.num_learning_iterations - self.global_step

        if max_steps <= 0:
            return

        target_step = min(self.global_step + max_steps, args.num_learning_iterations)
        if target_step <= self.global_step:
            return

        if args.compile:
            if not hasattr(self, "_compiled_update_critic"):
                self._compiled_update_critic = torch.compile(self._update_critic)
                self._compiled_update_actor = torch.compile(self._update_actor)
            update_critic = self._compiled_update_critic
            update_actor = self._compiled_update_actor
            # Keep alpha update eager for stability (optimizer+grad side effects in compile can be brittle).
            update_alpha = self._update_alpha
        else:
            update_critic = self._update_critic
            update_actor = self._update_actor
            update_alpha = self._update_alpha

        if self.env.num_envs > 1 and self.is_main_process:
            logger.warning(
                "Offline SAC does not use vectorized environment rollouts. "
                f"Current num_envs={self.env.num_envs} only increases simulator memory usage."
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

                    reward_mean, q_grad_norm, q_loss, q_target_max, q_target_min = update_critic(data)
                    alpha_loss = update_alpha(data)

                    self._critic_update_step += 1
                    is_actor_update_step = self._critic_update_step % args.policy_frequency == 0
                    if is_actor_update_step:
                        actor_grad_norm, actor_loss, policy_entropy, action_std = update_actor(data)
                    else:
                        actor_grad_norm = torch.tensor(0.0, device=self.device)
                        actor_loss = torch.tensor(0.0, device=self.device)
                        policy_entropy = torch.tensor(0.0, device=self.device)
                        action_std = torch.tensor(0.0, device=self.device)

                    self._soft_update_q_target()

                    self.training_metrics.add(
                        {
                            "buffer_rewards": reward_mean,
                            "qf_loss": q_loss,
                            "qf_max": q_target_max,
                            "qf_min": q_target_min,
                            "critic_grad_norm": q_grad_norm,
                            "alpha_loss": alpha_loss,
                            "alpha_value": self.log_alpha.exp().detach().mean(),
                            "actor_loss": actor_loss,
                            "actor_grad_norm": actor_grad_norm,
                            "policy_entropy": policy_entropy,
                            "action_std": action_std,
                            "is_actor_update_step": float(is_actor_update_step),
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
                    logger.info(f"Saving offline SAC model at global step {self.global_step}")
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
        metadata["action_space_mode"] = "env_scaled_action_training_v1"
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
            metadata=metadata,
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
            return policy(normalized_obs)[0]

        return policy_fn

    @property
    def actor_onnx_wrapper(self):
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
                return self.actor(normalized_obs)[0]

        return ActorWrapper(
            actor,
            obs_normalizer if self.obs_normalization else None,
        )

    def extract_actor_obs(self, obs: torch.Tensor, obs_key: str) -> torch.Tensor:
        if obs_key not in self.actor_obs_indices:
            raise ValueError(
                f"Observation key '{obs_key}' not found in actor observations. "
                f"Available keys: {list(self.actor_obs_indices.keys())}"
            )
        indices = self.actor_obs_indices[obs_key]
        return obs[..., indices["start"] : indices["end"]]

    def extract_critic_obs(self, obs: torch.Tensor, obs_key: str) -> torch.Tensor:
        if obs_key not in self.critic_obs_indices:
            raise ValueError(
                f"Observation key '{obs_key}' not found in critic observations. "
                f"Available keys: {list(self.critic_obs_indices.keys())}"
            )
        indices = self.critic_obs_indices[obs_key]
        return obs[..., indices["start"] : indices["end"]]

    def get_actor_obs_info(self) -> dict[str, dict[str, int]]:
        return self.actor_obs_indices.copy()

    def get_critic_obs_info(self) -> dict[str, dict[str, int]]:
        return self.critic_obs_indices.copy()

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
            actions = self.actor(normalized_obs)[0]
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

        for t in itertools.count():
            if max_eval_steps is not None and t >= max_eval_steps:
                stop_reason = "max_eval_steps"
                break

            if self.obs_normalization:
                normalized_obs = self.obs_normalizer(obs, update=False)
            else:
                normalized_obs = obs

            actions = self.actor(normalized_obs)[0]
            obs, rewards, dones, infos = self.env.step(actions)

            episode_return += float(rewards[eval_env_idx].item())
            episode_length += 1

            if bool(dones[eval_env_idx].item()):
                stop_reason = "done"
                break

            if "time_outs" in infos and bool(infos["time_outs"][eval_env_idx].item()):
                stop_reason = "time_out"
                break

            if use_early_termination and "early_termination" in infos:
                if bool(infos["early_termination"][eval_env_idx].item()):
                    stop_reason = "early_termination"
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
        }


# Backward compatibility for older configs/checkpoints that referenced previous class names.
OFFLINESACEnv = OfflineSACEnv
OFFLINESACAgent = OfflineSACAgent
