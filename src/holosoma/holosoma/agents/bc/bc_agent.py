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
from holosoma.agents.bc.bc import Actor, CNNActor
from holosoma.agents.bc.bc_utils import EmpiricalNormalization, save_params
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.config_types.algo import BCConfig
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


class BCEnv:
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


class BCAgent(BaseAlgo):
    config: BCConfig
    env: BCEnv  # type: ignore[assignment]
    actor: Actor

    def __init__(
        self,
        env: BaseTask,
        config: BCConfig,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict | None = None,
    ):
        wrapped_env = BCEnv(env, config.actor_obs_keys, config.critic_obs_keys)
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

        if config.actor_learning_rate <= 0.0:
            raise ValueError(f"actor_learning_rate must be > 0, got {config.actor_learning_rate}")
        if config.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {config.batch_size}")
        if config.num_updates <= 0:
            raise ValueError(f"num_updates must be > 0, got {config.num_updates}")
        if config.eval_interval < 0:
            raise ValueError(f"eval_interval must be >= 0, got {config.eval_interval}")

    def setup(self) -> None:
        logger.info("Setting up offline BC")

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

        self.scaler = GradScaler(enabled=args.amp)

        self.obs_normalization = args.obs_normalization
        if self.obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(shape=actor_obs_dim, device=device)
        else:
            self.obs_normalizer = nn.Identity()

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

        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=args.actor_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )

        def _env_policy(obs: torch.Tensor, dones: torch.Tensor | None = None, deterministic: bool = False) -> torch.Tensor:
            u_actions = self.actor.explore(obs, dones=dones, deterministic=deterministic)
            return self._to_env_actions(u_actions)

        self.policy = _env_policy
        logger.info(f"BC dims: actor_obs_dim={actor_obs_dim}, critic_obs_dim={critic_obs_dim}, n_act={n_act}")

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
        logger.info(f"Synchronized BC model parameters across {self.gpu_world_size} GPUs")

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

    def _to_u_actions(self, actions: torch.Tensor) -> torch.Tensor:
        action_scale = self.actor.action_scale.to(device=actions.device, dtype=actions.dtype)
        action_bias = self.actor.action_bias.to(device=actions.device, dtype=actions.dtype)
        return ((actions - action_bias) / (action_scale + 1e-6)).clamp(-1.0, 1.0)

    def _to_env_actions(self, u_actions: torch.Tensor) -> torch.Tensor:
        action_scale = self.actor.action_scale.to(device=u_actions.device, dtype=u_actions.dtype)
        action_bias = self.actor.action_bias.to(device=u_actions.device, dtype=u_actions.dtype)
        return u_actions * action_scale + action_bias

    def _maybe_to_u_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.numel() == 0:
            return actions
        max_abs = actions.detach().abs().max()
        if bool((max_abs <= 1.05).item()):
            return actions.clamp(-1.0, 1.0)
        return self._to_u_actions(actions)

    def _update_actor(
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
    ]:
        args = self.config
        scaler = self.scaler

        with self._maybe_amp():
            actor_observations = data["observations"]  # [B, actor_obs_dim]
            dataset_actions_u = self._maybe_to_u_actions(data["actions"])  # [B, action_dim]

            log_prob_data = self.actor.log_prob_dataset_actions(actor_observations, dataset_actions_u)  # [B]
            actor_loss = -log_prob_data.mean()

            policy_actions_u, _, log_std = self.actor(actor_observations)
            bc_l1_u = F.l1_loss(policy_actions_u, dataset_actions_u)
            bc_mse_u = F.mse_loss(policy_actions_u, dataset_actions_u)
            action_std = log_std.exp().mean()
            policy_abs_u_mean = policy_actions_u.abs().mean()
            dataset_abs_u_mean = dataset_actions_u.abs().mean()

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
            log_prob_data.mean().detach(),
            bc_l1_u.detach(),
            bc_mse_u.detach(),
            action_std.detach(),
            policy_abs_u_mean.detach(),
            dataset_abs_u_mean.detach(),
        )

    def _load_offline_dataset_cache(self) -> dict[str, torch.Tensor]:
        if self._offline_dataset_cache is not None:
            return self._offline_dataset_cache

        if not self._offline_dataset_path.exists():
            raise FileNotFoundError(
                f"Offline dataset not found at '{self._offline_dataset_path}'. "
                "Provide a valid offline dataset path in BC config."
            )

        required_keys = ("observations", "actions")

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

            def _load_feature_tensor(key: str, expected_dim: int) -> torch.Tensor:
                tensor = _load_tensor(key, torch.float32)
                if tensor.ndim != 2 or tensor.shape[1] != expected_dim:
                    raise ValueError(
                        f"Offline dataset key '{key}' has shape {tuple(tensor.shape)}, "
                        f"expected [N, {expected_dim}]"
                    )
                return tensor

            self._offline_dataset_cache = {
                "observations": _load_feature_tensor("observations", self.actor_obs_dim),
                "actions": _load_feature_tensor("actions", self.env.robot_config.actions_dim),
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
    ) -> list[TensorDict]:
        offline_cache = self._load_offline_dataset_cache()
        large_batch_size = batch_size * num_updates
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

        observations = _sample_cached("observations", torch.float32)
        actions = _sample_cached("actions", torch.float32)

        if self.config.use_symmetry:
            observations = self.symmetry_utils.augment_observations(
                obs=observations,
                env=self.env,
                obs_list=self.config.actor_obs_keys,
            )
            actions = self.symmetry_utils.augment_actions(actions=actions)

        actions = self._maybe_to_u_actions(actions).clamp(-1.0, 1.0)
        observations = normalize_obs(observations)

        total_samples = observations.shape[0]
        if total_samples % num_updates != 0:
            raise ValueError(
                f"Augmented BC batch size {total_samples} is not divisible by num_updates={num_updates}."
            )

        samples_per_update = total_samples // num_updates
        prepared_batches: list[TensorDict] = []
        for i in range(num_updates):
            start_idx = i * samples_per_update
            end_idx = (i + 1) * samples_per_update

            batch_data = TensorDict(
                {
                    "observations": observations[start_idx:end_idx],
                    "actions": actions[start_idx:end_idx],
                },
                batch_size=samples_per_update,
                device=self.device,
            )
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

        required_keys = ("actor_state_dict",)
        missing_required = [k for k in required_keys if k not in torch_checkpoint]
        if missing_required:
            raise RuntimeError(
                f"Checkpoint missing required BC keys: {missing_required}. "
                "Expected a BC checkpoint."
            )

        self.actor.load_state_dict(torch_checkpoint["actor_state_dict"])

        obs_norm_state = torch_checkpoint.get("obs_normalizer_state")
        if self.obs_normalization:
            if not isinstance(obs_norm_state, dict) or not obs_norm_state:
                raise RuntimeError("Checkpoint missing valid obs_normalizer_state while obs normalization is enabled.")

        self.obs_normalizer.load_state_dict(obs_norm_state if isinstance(obs_norm_state, dict) else {})

        if "actor_optimizer_state_dict" in torch_checkpoint:
            self.actor_optimizer.load_state_dict(torch_checkpoint["actor_optimizer_state_dict"])
        if "grad_scaler_state_dict" in torch_checkpoint and torch_checkpoint["grad_scaler_state_dict"] is not None:
            self.scaler.load_state_dict(torch_checkpoint["grad_scaler_state_dict"])

        self.global_step = int(torch_checkpoint.get("global_step", 0))
        self._restore_env_state(torch_checkpoint.get("env_state"))

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
            if not hasattr(self, "_compiled_update_actor"):
                self._compiled_update_actor = torch.compile(self._update_actor)
            update_actor = self._compiled_update_actor
        else:
            update_actor = self._update_actor

        if self.env.num_envs > 1 and self.is_main_process:
            logger.warning(
                "Offline BC does not use vectorized environment rollouts. "
                f"Current num_envs={self.env.num_envs} only increases simulator memory usage."
            )

        normalize_obs = self.obs_normalizer.forward

        pbar = tqdm.tqdm(total=max(target_step - self.global_step, 0), initial=0, leave=False)
        while self.global_step < target_step:
            self.global_step += 1

            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            batch_size = max(args.batch_size // self.gpu_world_size, 1)
            with self.logging_helper.record_learn_time():
                offline_batches = self.offline_dataset_random_sampling(
                    batch_size=batch_size,
                    num_updates=args.num_updates,
                    normalize_obs=normalize_obs,
                )

                for data in offline_batches:
                    (
                        actor_grad_norm,
                        actor_loss,
                        log_prob_data_mean,
                        bc_l1_u,
                        bc_mse_u,
                        action_std,
                        policy_abs_u_mean,
                        dataset_abs_u_mean,
                    ) = update_actor(data)

                    self.training_metrics.add(
                        {
                            "actor_grad_norm": actor_grad_norm,
                            "actor_loss": actor_loss,
                            "log_prob_data_mean": log_prob_data_mean,
                            "bc_l1_u": bc_l1_u,
                            "bc_mse_u": bc_mse_u,
                            "action_std": action_std,
                            "policy_abs_u_mean": policy_abs_u_mean,
                            "dataset_abs_u_mean": dataset_abs_u_mean,
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
                    logger.info(f"Saving BC model at global step {self.global_step}")
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
            self.obs_normalizer,
            self.actor_optimizer,
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
            u_actions = policy(normalized_obs)[0]
            return self._to_env_actions(u_actions)

        return policy_fn

    @property
    def actor_onnx_wrapper(self):
        actor = copy.deepcopy(self.actor).to("cpu")
        obs_normalizer = copy.deepcopy(self.obs_normalizer).to("cpu")
        action_scale = copy.deepcopy(self.actor.action_scale).to("cpu")
        action_bias = copy.deepcopy(self.actor.action_bias).to("cpu")

        class ActorWrapper(nn.Module):
            def __init__(self, actor, obs_normalizer, action_scale, action_bias):
                super().__init__()
                self.actor = actor
                self.obs_normalizer = obs_normalizer
                self.register_buffer("action_scale", action_scale)
                self.register_buffer("action_bias", action_bias)

            def forward(self, actor_obs):
                if self.obs_normalizer is not None:
                    normalized_obs = self.obs_normalizer(actor_obs, update=False)
                else:
                    normalized_obs = actor_obs
                u_actions = self.actor(normalized_obs)[0]
                return u_actions * self.action_scale + self.action_bias

        return ActorWrapper(
            actor,
            obs_normalizer if self.obs_normalization else None,
            action_scale,
            action_bias,
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
            u_actions = self.actor(normalized_obs)[0]
            env_actions = self._to_env_actions(u_actions)
            obs, _, _, _ = self.env.step(env_actions)

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

            u_actions = self.actor(normalized_obs)[0]
            env_actions = self._to_env_actions(u_actions)
            obs, rewards, dones, infos = self.env.step(env_actions)

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


class CQLAgent(BCAgent):
    """Backward-compatible alias for old configs pointing to bc_agent.CQLAgent."""

    pass
