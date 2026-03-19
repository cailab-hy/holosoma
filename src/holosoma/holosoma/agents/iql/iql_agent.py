from __future__ import annotations

import copy
import itertools
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import h5py
import numpy as np
import tqdm
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.iql.iql import Actor, CNNActor, DoubleQCritic, ValueNetwork
from holosoma.agents.iql.iql_utils import EmpiricalNormalization, save_params
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.config_types.algo import IQLConfig
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


class IQLEnv:
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


class IQLAgent(BaseAlgo):
    config: IQLConfig
    env: IQLEnv  # type: ignore[assignment]
    actor: Actor
    qnet: DoubleQCritic
    qnet_target: DoubleQCritic
    value_net: ValueNetwork

    def __init__(
        self,
        env: BaseTask,
        config: IQLConfig,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict | None = None,
    ):
        wrapped_env = IQLEnv(env, config.actor_obs_keys, config.critic_obs_keys)
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
        self._offline_dataset_cache: dict[str, torch.Tensor] | None = None
        self._offline_num_samples = 0

        if not 0.0 < config.expectile < 1.0:
            raise ValueError(f"expectile must be in (0, 1), got {config.expectile}")
        if config.beta <= 0.0:
            raise ValueError(f"beta must be > 0, got {config.beta}")
        if config.max_weight <= 0.0:
            raise ValueError(f"max_weight must be > 0, got {config.max_weight}")
        if config.discount <= 0.0 or config.discount > 1.0:
            raise ValueError(f"discount must be in (0, 1], got {config.discount}")
        if config.tau <= 0.0 or config.tau > 1.0:
            raise ValueError(f"tau must be in (0, 1], got {config.tau}")

    def setup(self) -> None:
        logger.info("Setting up offline IQL")

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

        # Scalar critics for IQL baseline.
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

        self.value_net = ValueNetwork(
            obs_indices=self.critic_obs_indices,
            obs_keys=list(args.critic_obs_keys),
            hidden_dim=args.value_hidden_dim,
            use_layer_norm=args.use_layer_norm,
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
        self.value_optimizer = optim.AdamW(
            self.value_net.parameters(),
            lr=args.value_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )

        self.policy = self.actor.explore
        logger.info(f"IQL dims: actor_obs_dim={actor_obs_dim}, critic_obs_dim={critic_obs_dim}, n_act={n_act}")

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
        for param in self.value_net.parameters():
            torch.distributed.broadcast(param.data, src=0)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        logger.info(f"Synchronized IQL model parameters across {self.gpu_world_size} GPUs")

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

    def _update_q(
        self,
        data: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        args = self.config
        scaler = self.scaler

        with self._maybe_amp():
            critic_observations = data["critic_observations"]  # [B, critic_obs_dim]
            next_critic_observations = data["next"]["critic_observations"]  # [B, critic_obs_dim]
            actions = data["actions"]  # [B, action_dim]
            rewards = data["next"]["rewards"]  # [B]
            dones = data["next"]["dones"].bool()  # [B]
            truncations = data["next"]["truncations"].bool()  # [B]
            bootstrap = (truncations | ~dones).float()  # [B]
            discount = args.discount ** data["next"]["effective_n_steps"]  # [B]

            with torch.no_grad():
                next_v = self.value_net(next_critic_observations)  # [B]
                q_target = rewards + discount * bootstrap * next_v  # [B]

            q1, q2 = self.qnet(critic_observations, actions)  # [B], [B]
            q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

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
            q_loss.detach(),
            q_grad_norm.detach(),
            q_target.mean().detach(),
            q1.mean().detach(),
            q2.mean().detach(),
            rewards.mean().detach(),
        )

    def _update_value(
        self,
        data: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        args = self.config
        scaler = self.scaler

        with self._maybe_amp():
            critic_observations = data["critic_observations"]  # [B, critic_obs_dim]
            actions = data["actions"]  # [B, action_dim]

            with torch.no_grad():
                q1, q2 = self.qnet(critic_observations, actions)  # [B], [B]
                q_min = torch.minimum(q1, q2)  # [B]

            v = self.value_net(critic_observations)  # [B]
            diff = q_min - v  # [B]
            weight = torch.where(diff < 0, 1.0 - args.expectile, args.expectile)
            value_loss = (weight * diff.pow(2)).mean()

        self.value_optimizer.zero_grad(set_to_none=True)
        scaler.scale(value_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(self.value_net)

        scaler.unscale_(self.value_optimizer)
        if args.max_grad_norm > 0:
            value_grad_norm = torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), args.max_grad_norm)
        else:
            value_grad_norm = torch.tensor(0.0, device=self.device)

        scaler.step(self.value_optimizer)
        scaler.update()

        return (
            value_loss.detach(),
            value_grad_norm.detach(),
            v.mean().detach(),
            q_min.mean().detach(),
            diff.mean().detach(),
        )

    def _update_actor(
        self,
        data: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        args = self.config
        scaler = self.scaler

        with self._maybe_amp():
            actor_observations = data["observations"]  # [B, actor_obs_dim]
            critic_observations = data["critic_observations"]  # [B, critic_obs_dim]
            dataset_actions = data["actions"]  # [B, action_dim]

            with torch.no_grad():
                q1, q2 = self.qnet(critic_observations, dataset_actions)  # [B], [B]
                q_min = torch.minimum(q1, q2)  # [B]
                v = self.value_net(critic_observations)  # [B]
                adv = q_min - v  # [B]
                weights = torch.exp(args.beta * adv).clamp(max=args.max_weight)  # [B]

            log_prob_data = self.actor.log_prob_dataset_actions(actor_observations, dataset_actions)  # [B]
            actor_loss = -(weights * log_prob_data).mean()

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
            actor_loss.detach(),
            actor_grad_norm.detach(),
            adv.mean().detach(),
            weights.mean().detach(),
            log_prob_data.mean().detach(),
        )

    def _load_offline_dataset_cache(self) -> dict[str, torch.Tensor]:
        if self._offline_dataset_cache is not None:
            return self._offline_dataset_cache

        if not self._offline_dataset_path.exists():
            raise FileNotFoundError(
                f"Offline dataset not found at '{self._offline_dataset_path}'. "
                "Provide a valid offline dataset path in IQL config."
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
            f"Cached offline dataset '{self._offline_dataset_path}' in host memory ({self._offline_num_samples} samples)."
        )
        return self._offline_dataset_cache

    def offline_dataset_random_sampling(
        self,
        batch_size: int,
        num_updates: int,
        normalize_obs,
        normalize_critic_obs,
    ) -> list[TensorDict]:
        """Sample fixed offline data only. No online collection."""
        offline_cache = self._load_offline_dataset_cache()

        samples_per_update = batch_size * self.env.num_envs
        large_batch_size = samples_per_update * num_updates
        replace = large_batch_size > self._offline_num_samples

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
            assert isinstance(observations_tensor, torch.Tensor)
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

        prepared_batches: list[TensorDict] = []
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

    def load(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return

        torch_checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        checkpoint_args = torch_checkpoint.get("args", {})
        checkpoint_obs_norm = checkpoint_args.get("obs_normalization")
        if checkpoint_obs_norm is not None and bool(checkpoint_obs_norm) != bool(self.obs_normalization):
            raise RuntimeError(
                "Checkpoint/config mismatch for observation normalization: "
                f"checkpoint obs_normalization={checkpoint_obs_norm}, "
                f"current config obs_normalization={self.obs_normalization}."
            )

        required_keys = ("actor_state_dict", "qnet_state_dict", "value_net_state_dict")
        missing_required = [k for k in required_keys if k not in torch_checkpoint]
        if missing_required:
            raise RuntimeError(
                f"Checkpoint missing required IQL keys: {missing_required}. "
                "Expected an IQL checkpoint."
            )

        self.actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        self.qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        self.value_net.load_state_dict(torch_checkpoint["value_net_state_dict"])

        if "qnet_target_state_dict" in torch_checkpoint:
            self.qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        else:
            self.qnet_target.load_state_dict(self.qnet.state_dict())

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
        if "value_optimizer_state_dict" in torch_checkpoint:
            self.value_optimizer.load_state_dict(torch_checkpoint["value_optimizer_state_dict"])
        if "grad_scaler_state_dict" in torch_checkpoint and torch_checkpoint["grad_scaler_state_dict"] is not None:
            self.scaler.load_state_dict(torch_checkpoint["grad_scaler_state_dict"])

        self.global_step = int(torch_checkpoint.get("global_step", 0))
        self._restore_env_state(torch_checkpoint.get("env_state"))

    def offline_learn(self, max_steps: int | None = None) -> None:
        """Offline-only training loop.

        No environment stepping, no rollout collection, no replay append.
        """
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
            if not hasattr(self, "_compiled_update_q"):
                self._compiled_update_q = torch.compile(self._update_q)
                self._compiled_update_value = torch.compile(self._update_value)
                self._compiled_update_actor = torch.compile(self._update_actor)
            update_q = self._compiled_update_q
            update_value = self._compiled_update_value
            update_actor = self._compiled_update_actor
        else:
            update_q = self._update_q
            update_value = self._update_value
            update_actor = self._update_actor

        normalize_obs = self.obs_normalizer.forward
        normalize_critic_obs = self.critic_obs_normalizer.forward

        pbar = tqdm.tqdm(total=max(target_step - self.global_step, 0), initial=0, leave=False)
        while self.global_step < target_step:
            self.global_step += 1

            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            batch_size = max(args.batch_size // self.env.num_envs // self.gpu_world_size, 1)
            with self.logging_helper.record_learn_time():
                offline_batches = self.offline_dataset_random_sampling(
                    batch_size=batch_size,
                    num_updates=args.num_updates,
                    normalize_obs=normalize_obs,
                    normalize_critic_obs=normalize_critic_obs,
                )

                for data in offline_batches:
                    q_loss, q_grad_norm, q_target_mean, q1_mean, q2_mean, reward_mean = update_q(data)
                    value_loss, value_grad_norm, v_mean, q_min_mean, value_diff_mean = update_value(data)
                    actor_loss, actor_grad_norm, adv_mean, weight_mean, log_prob_mean = update_actor(data)

                    self._soft_update_q_target()

                    self.training_metrics.add(
                        {
                            "q_loss": q_loss,
                            "value_loss": value_loss,
                            "actor_loss": actor_loss,
                            "q_grad_norm": q_grad_norm,
                            "value_grad_norm": value_grad_norm,
                            "actor_grad_norm": actor_grad_norm,
                            "q_target_mean": q_target_mean,
                            "q1_mean": q1_mean,
                            "q2_mean": q2_mean,
                            "v_mean": v_mean,
                            "q_min_mean": q_min_mean,
                            "value_diff_mean": value_diff_mean,
                            "adv_mean": adv_mean,
                            "weight_mean": weight_mean,
                            "log_prob_data_mean": log_prob_mean,
                            "buffer_rewards": reward_mean,
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
                    logger.info(f"Saving IQL model at global step {self.global_step}")
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
            self.value_net,
            self.obs_normalizer,
            self.critic_obs_normalizer,
            self.actor_optimizer,
            self.q_optimizer,
            self.value_optimizer,
            self.scaler,
            self.config,
            path,
            save_fn=self.logging_helper.save_checkpoint_artifact,
            env_state=env_state or None,
            metadata=self._checkpoint_metadata(iteration=self.global_step),
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

        return ActorWrapper(actor, obs_normalizer if self.obs_normalization else None)

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
        """Evaluate exactly one environment trajectory (env index 0)."""
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
