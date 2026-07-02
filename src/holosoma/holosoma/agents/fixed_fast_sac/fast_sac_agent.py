from __future__ import annotations

import copy
import itertools
import math
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Sequence

import tqdm
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.fast_sac.fast_sac import Actor, CNNActor, CNNCritic, Critic
from holosoma.agents.fast_sac.fast_sac_utils import (
    EmpiricalNormalization,
    SimpleReplayBuffer,
    save_params,
)
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.config_types.algo import FastSACConfig
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


class FastSACEnv:
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

        termination_reasons = info_dict.get("termination_reasons", {})
        tracking_metrics = info_dict.get("tracking_metrics", {})

        def _bool_info(name: str) -> torch.Tensor:
            value = termination_reasons.get(name)
            if isinstance(value, torch.Tensor):
                return value.to(device=rew_buf.device, dtype=torch.bool)
            return torch.zeros_like(rew_buf, dtype=torch.bool)

        def _float_info(name: str) -> torch.Tensor:
            value = tracking_metrics.get(name)
            if isinstance(value, torch.Tensor):
                return value.to(device=rew_buf.device, dtype=torch.float32)
            # wonwoo: keep offline export robust even if the environment does not expose this metric yet.
            return torch.zeros_like(rew_buf, dtype=torch.float32)

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
            "termination_reasons": {
                "bad_tracking": _bool_info("bad_tracking"),
                "motion_ends": _bool_info("motion_ends"),
                "timeout": _bool_info("timeout"),
            },
            "tracking_metrics": {
                "err_root_pos": _float_info("err_root_pos"),
                "err_root_ori": _float_info("err_root_ori"),
                "err_body_pos_max": _float_info("err_body_pos_max"),
                "err_body_pos_mean": _float_info("err_body_pos_mean"),
                "err_object_pos": _float_info("err_object_pos"),
                "err_object_ori": _float_info("err_object_ori"),
            },
            "episode_step": info_dict.get("episode_step", torch.zeros_like(rew_buf, dtype=torch.long)),
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
import os
import atexit
from pathlib import Path

import h5py
import numpy as np
import torch


# ------------------------------------------------------------------
# utilities
# ------------------------------------------------------------------
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_float32(x):
    return _to_numpy(x).astype(np.float32, copy=False)


def _as_uint8(x):
    return _to_numpy(x).astype(np.uint8, copy=False)


def _as_int64(x):
    return _to_numpy(x).astype(np.int64, copy=False)


def _flatten_transition(prefix: str, value, out: dict[str, Any]) -> None:
    if isinstance(value, TensorDict):
        value = value.to_dict()
    if isinstance(value, dict):
        for key, nested_value in value.items():
            next_prefix = f"{prefix}_{key}" if prefix else key
            _flatten_transition(next_prefix, nested_value, out)
        return
    out[prefix] = value


def _convert_h5_field(name: str, value) -> np.ndarray:
    if name.endswith(("truncations", "dones", "done_bad_tracking", "done_motion_ends", "done_timeout")):
        return _as_uint8(value)
    if name.endswith(("episode_step", "global_step")):
        return _as_int64(value)
    return _as_float32(value)


# ------------------------------------------------------------------
# HDF5 writer
# ------------------------------------------------------------------
class H5TransitionWriter:
    def __init__(
        self,
        path: str,
        flush_every: int = 100,
        compression: str = "gzip",
        compression_opts: int = 4,
    ):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        self.f = h5py.File(self.path, "w")
        self.flush_every = flush_every
        self.compression = compression
        self.compression_opts = compression_opts
        self.num_appends = 0

        # existing dataset length if resuming
        self.n = int(self.f.attrs.get("num_samples", 0))
        self.f.attrs["format"] = "holosoma_offpolicy_transition_v1"

        atexit.register(self.close)

    def _require_dataset(self, name: str, arr: np.ndarray):
        if name in self.f:
            return self.f[name]

        ds = self.f.create_dataset(
            name=name,
            shape=(0,) + arr.shape[1:],
            maxshape=(None,) + arr.shape[1:],
            dtype=arr.dtype,
            chunks=True,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        return ds

    def append(self, transition):
        flat_transition: dict[str, Any] = {}
        _flatten_transition("", transition, flat_transition)

        batch: dict[str, np.ndarray] = {}
        for name, value in flat_transition.items():
            h5_name = name
            if h5_name.startswith("next_"):
                if h5_name == "next_rewards":
                    h5_name = "rewards"
                elif h5_name == "next_truncations":
                    h5_name = "truncations"
                elif h5_name == "next_dones":
                    h5_name = "dones"
            batch[h5_name] = _convert_h5_field(h5_name, value)

        batch_size = batch["observations"].shape[0]

        # sanity check
        for k, v in batch.items():
            if v.shape[0] != batch_size:
                raise ValueError(
                    f"Inconsistent batch size for '{k}': "
                    f"expected {batch_size}, got {v.shape[0]}"
                )

        start = self.n
        end = self.n + batch_size

        for name, arr in batch.items():
            ds = self._require_dataset(name, arr)
            ds.resize((end,) + ds.shape[1:])
            ds[start:end] = arr

        self.n = end
        self.f.attrs["num_samples"] = self.n
        # TODO(wonwoo): persist bad-tracking thresholds in HDF5 attrs once the saver receives config metadata.

        self.num_appends += 1
        if self.num_appends % self.flush_every == 0:
            self.f.flush()

    def flush(self):
        if self.f is not None:
            self.f.flush()

    def close(self):
        if getattr(self, "f", None) is not None:
            try:
                self.f.flush()
                self.f.close()
            finally:
                self.f = None


# ------------------------------------------------------------------
# global saver API
# ------------------------------------------------------------------
_H5_WRITER = None


def init_transition_saver(
    path: str = "offline_data/fastsac_dataset.h5",
    flush_every: int = 100,
):
    global _H5_WRITER
    if _H5_WRITER is not None:
        _H5_WRITER.close()
    _H5_WRITER = H5TransitionWriter(
        path=path,
        flush_every=flush_every,
    )
    return _H5_WRITER


def save_transition(transition, path: str = "offline_data/fastsac_dataset.h5"):
    global _H5_WRITER
    if _H5_WRITER is None:
        _H5_WRITER = H5TransitionWriter(path=path)
    _H5_WRITER.append(transition)


def close_transition_saver():
    global _H5_WRITER
    if _H5_WRITER is not None:
        _H5_WRITER.close()
        _H5_WRITER = None

class FastSACAgent(BaseAlgo):
    """
    FastSAC is an efficient variant of Soft Actor-Critic (SAC) tuned for
    large-scale training with massively parallel simulation.
    See https://arxiv.org/abs/2505.22642 for more details about FastTD3.
    Detailed technical report for FastSAC will be available soon.
    """

    config: FastSACConfig
    env: FastSACEnv  # type: ignore[assignment]
    actor: Actor
    qnet: Critic

    def __init__(
        self, env: BaseTask, config: FastSACConfig, device: str, log_dir: str, multi_gpu_cfg: dict | None = None
    ):
        wrapped_env = FastSACEnv(env, config.actor_obs_keys, config.critic_obs_keys)

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

        



    def setup(self) -> None:
        logger.info("Setting up FastSAC")

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        args = self.config

        scaler = self.scaler
        actor = self.actor
        qnet = self.qnet
        qnet_target = self.qnet_target
        q_optimizer = self.q_optimizer
        alpha_optimizer = self.alpha_optimizer

        with self._maybe_amp():
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
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

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

    def load(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return
        torch_checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        if "actor_state_dict" not in torch_checkpoint:
            raise RuntimeError("Checkpoint missing required FastSAC key: actor_state_dict")

        self.actor.load_state_dict(torch_checkpoint["actor_state_dict"])

        if "qnet_state_dict" in torch_checkpoint:
            self.qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        if "qnet_target_state_dict" in torch_checkpoint:
            self.qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        elif "qnet_state_dict" in torch_checkpoint:
            self.qnet_target.load_state_dict(self.qnet.state_dict())

        obs_norm_state = torch_checkpoint.get("obs_normalizer_state")
        critic_obs_norm_state = torch_checkpoint.get("critic_obs_normalizer_state")
        if isinstance(obs_norm_state, dict):
            self.obs_normalizer.load_state_dict(obs_norm_state)
        elif self.obs_normalization:
            logger.warning("FastSAC checkpoint has no obs_normalizer_state; eval will use fresh normalizer stats.")
        if isinstance(critic_obs_norm_state, dict):
            self.critic_obs_normalizer.load_state_dict(critic_obs_norm_state)
        elif self.obs_normalization:
            logger.warning("FastSAC checkpoint has no critic_obs_normalizer_state; eval critic stats are fresh.")

        if "log_alpha" in torch_checkpoint:
            self.log_alpha.data.copy_(torch_checkpoint["log_alpha"].to(self.device))
        if "actor_optimizer_state_dict" in torch_checkpoint:
            self.actor_optimizer.load_state_dict(torch_checkpoint["actor_optimizer_state_dict"])
        if "q_optimizer_state_dict" in torch_checkpoint:
            self.q_optimizer.load_state_dict(torch_checkpoint["q_optimizer_state_dict"])
        if "alpha_optimizer_state_dict" in torch_checkpoint:
            self.alpha_optimizer.load_state_dict(torch_checkpoint["alpha_optimizer_state_dict"])
        grad_scaler_state = torch_checkpoint.get("grad_scaler_state_dict")
        if grad_scaler_state is not None:
            self.scaler.load_state_dict(grad_scaler_state)

        self.global_step = int(torch_checkpoint.get("global_step", 0))
        self._restore_env_state(torch_checkpoint.get("env_state"))

    def learn(self) -> None:
        args = self.config
        device = self.device

        if args.compile:
            policy = torch.compile(self.policy)
            normalize_obs = torch.compile(self.obs_normalizer.forward)
        else:
            policy = self.policy
            normalize_obs = self.obs_normalizer.forward

        env = self.env
        obs, critic_obs = env.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)

        dones = None
        pbar = tqdm.tqdm(total=args.num_learning_iterations, initial=0)

        save_h5 = self.is_main_process
        if save_h5:
            init_transition_saver(args.offline_dataset_path, flush_every=1)
            logger.info(
                "Fixed FastSAC transition export enabled: "
                f"sample_envs={min(max(int(args.episode_data_active_envs), 0), env.num_envs)}, "
                f"path={args.offline_dataset_path}"
            )

        self.global_step = 0   # checkpoint step과 분리해서 수집 step으로 사용

        try:
            while self.global_step < args.num_learning_iterations:
                with self.logging_helper.record_collection_time():
                    with torch.no_grad(), self._maybe_amp():
                        norm_obs = normalize_obs(obs, update=False)
                        actions = policy(obs=norm_obs, dones=dones)   # explore 유지

                    next_obs, rewards, dones, infos = env.step(actions.float())
                    truncations = infos["time_outs"]
                    next_critic_obs = infos["observations"]["critic"]

                    self.logging_helper.update_episode_stats(rewards, dones, infos)

                    done_mask = dones.bool()
                    transition_to_save_next_obs = torch.where(
                        done_mask[:, None],
                        infos["observations"]["final"]["actor_obs"],
                        next_obs,
                    )
                    transition_to_save_next_critic_obs = torch.where(
                        done_mask[:, None],
                        infos["observations"]["final"]["critic_obs"],
                        next_critic_obs,
                    )

                    termination_reasons = infos.get("termination_reasons", {})
                    tracking_metrics = infos.get("tracking_metrics", {})

                    def _reason_tensor(name: str) -> torch.Tensor:
                        value = termination_reasons.get(name)
                        if isinstance(value, torch.Tensor):
                            return value.to(device=device, dtype=torch.uint8)
                        return torch.zeros(env.num_envs, device=device, dtype=torch.uint8)

                    def _metric_tensor(name: str) -> torch.Tensor:
                        value = tracking_metrics.get(name)
                        if isinstance(value, torch.Tensor):
                            return value.to(device=device, dtype=torch.float32)
                        return torch.zeros(env.num_envs, device=device, dtype=torch.float32)

                    episode_step = infos.get("episode_step")
                    if isinstance(episode_step, torch.Tensor):
                        episode_step_tensor = episode_step.to(device=device, dtype=torch.long)
                    else:
                        episode_step_tensor = torch.zeros(env.num_envs, device=device, dtype=torch.long)

                    global_step_tensor = torch.full(
                        (env.num_envs,),
                        int(self.global_step),
                        device=device,
                        dtype=torch.long,
                    )

                    transition_to_save = TensorDict(
                        {
                            "observations": obs,
                            "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                            "critic_observations": critic_obs,
                            "next": {
                                "observations": transition_to_save_next_obs,
                                "critic_observations": transition_to_save_next_critic_obs,
                                "rewards": torch.as_tensor(rewards, device=device, dtype=torch.float),
                                "truncations": truncations.to(torch.uint8),
                                "dones": dones.to(torch.uint8),
                                "done_bad_tracking": _reason_tensor("bad_tracking"),
                                "done_motion_ends": _reason_tensor("motion_ends"),
                                "done_timeout": _reason_tensor("timeout"),
                                "episode_step": episode_step_tensor,
                                "global_step": global_step_tensor,
                                "err_root_pos": _metric_tensor("err_root_pos"),
                                "err_root_ori": _metric_tensor("err_root_ori"),
                                "err_body_pos_max": _metric_tensor("err_body_pos_max"),
                                "err_body_pos_mean": _metric_tensor("err_body_pos_mean"),
                                "err_object_pos": _metric_tensor("err_object_pos"),
                                "err_object_ori": _metric_tensor("err_object_ori"),
                            },
                        },
                        batch_size=(env.num_envs,),
                        device=device,
                    )

                    if save_h5:
                        num_to_save = min(max(int(args.episode_data_active_envs), 0), env.num_envs)
                        if num_to_save > 0:
                            rand_idx = torch.randperm(env.num_envs, device=device)[:num_to_save]
                            save_transition(transition_to_save[rand_idx].clone().cpu())

                    obs = next_obs
                    critic_obs = next_critic_obs

                self.global_step += 1
                pbar.update(1)
                if self.global_step % args.logging_interval == 0:
                    loss_dict = {"env_rewards": float(torch.as_tensor(rewards).mean().item())}
                    self.logging_helper.post_epoch_logging(
                        it=self.global_step,
                        loss_dict=loss_dict,
                        extra_log_dicts={},
                    )

        finally:
            if save_h5:
                close_transition_saver()

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

    @torch.no_grad()
    def evaluate_one_episode(
        self,
        max_eval_steps: int | None = None,
        use_early_termination: bool = False,
    ) -> dict[str, float | int | str | None]:
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

        for step_idx in itertools.count():
            if max_eval_steps is not None and step_idx >= max_eval_steps:
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

            actions = self.actor(normalized_obs)[0]
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
