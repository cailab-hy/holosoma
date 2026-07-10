from __future__ import annotations

import copy
import itertools
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np
import tqdm
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.psc.psc import DoubleQCritic, FactorizedActor, resolve_action_groups
from holosoma.agents.psc.psc_utils import EmpiricalNormalization, save_params
from holosoma.agents.psc.syndiag import (
    build_coalitions,
    coalition_group_mask,
    coalition_q_values,
    compute_group_drift,
    group_dim_mask,
    quartile_delta_stats,
    recall_top_pair,
    singleton_columns,
    superadditivity_quad,
    synergy_residuals,
)
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.config_types.algo import BFCQLConfig
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


class PSCEnv:
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


class PSCAgent(BaseAlgo):
    config: BFCQLConfig
    env: PSCEnv  # type: ignore[assignment]
    actor: FactorizedActor
    qnet: DoubleQCritic
    qnet_target: DoubleQCritic

    def __init__(
        self,
        env: BaseTask,
        config: BFCQLConfig,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict | None = None,
    ):
        wrapped_env = PSCEnv(env, config.actor_obs_keys, config.critic_obs_keys)
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
        self._ood_actor_num = config.ood_actor_num
        self._temperature = config.cql_temperature
        self._cql_weight = config.cql_weight
        self._num_near_actions = config.cql_near_action_samples

        self._offline_dataset_path = Path(config.offline_dataset_path)
        self._offline_dataset_reader: HDF5BlockReader | None = None
        self._offline_shuffle_buffer: RAMShuffleBuffer | None = None
        self._offline_gpu_cache: GPUTransitionCache | None = None
        self._offline_num_samples = 0
        self._critic_update_step = 0

        # Synergy-OOD diagnostics (logging only; full state built in _syndiag_setup).
        syndiag_cfg = getattr(config, "syndiag", None)
        self._syndiag_cfg = syndiag_cfg
        self._syndiag_enabled = bool(syndiag_cfg is not None and syndiag_cfg.enabled)
        if self._syndiag_enabled:
            if syndiag_cfg.interval <= 0:
                raise ValueError(f"syndiag.interval must be > 0, got {syndiag_cfg.interval}")
            if syndiag_cfg.dump_interval < 0:
                raise ValueError(f"syndiag.dump_interval must be >= 0, got {syndiag_cfg.dump_interval}")
            if syndiag_cfg.dump_topk <= 0:
                raise ValueError(f"syndiag.dump_topk must be > 0, got {syndiag_cfg.dump_topk}")
            if syndiag_cfg.delta_min < 0.0:
                raise ValueError(f"syndiag.delta_min must be >= 0, got {syndiag_cfg.delta_min}")
            if syndiag_cfg.max_coalitions <= 0:
                raise ValueError(f"syndiag.max_coalitions must be > 0, got {syndiag_cfg.max_coalitions}")
            if syndiag_cfg.dump_max_rows <= 0:
                raise ValueError(f"syndiag.dump_max_rows must be > 0, got {syndiag_cfg.dump_max_rows}")

        if config.cql_num_action_samples <= 0:
            raise ValueError(f"cql_num_action_samples must be > 0, got {config.cql_num_action_samples}")
        if not config.psc.enabled:
            raise ValueError("PSCAgent requires psc.enabled=True.")
        if not config.psc.basis_path:
            raise ValueError("psc.basis_path is required; run scripts/psc_spectrum.py first.")
        if config.psc.rand_range_mult <= 0.0:
            raise ValueError(f"psc.rand_range_mult must be > 0, got {config.psc.rand_range_mult}")
        if not 0.0 <= config.psc.scale_floor_quantile <= 1.0:
            raise ValueError(
                f"psc.scale_floor_quantile must be in [0, 1], got {config.psc.scale_floor_quantile}"
            )
        if any(size <= 0 for size in config.psc.block_sizes):
            raise ValueError(f"psc.block_sizes must be positive, got {config.psc.block_sizes}")
        if config.ood_actor_num <= 0:
            raise ValueError(f"ood_actor_num must be > 0, got {config.ood_actor_num}")
        if config.cql_temperature <= 0.0:
            raise ValueError(f"cql_temperature must be > 0, got {config.cql_temperature}")
        if config.cql_weight < 0.0:
            raise ValueError(f"cql_weight must be >= 0, got {config.cql_weight}")
        if config.cql_near_action_samples < 0:
            raise ValueError(f"cql_near_action_samples must be >= 0, got {config.cql_near_action_samples}")
        if config.cql_near_noise_std < 0.0:
            raise ValueError(f"cql_near_noise_std must be >= 0, got {config.cql_near_noise_std}")
        if config.use_lagrange:
            if config.cql_target_action_gap < 0.0:
                raise ValueError(
                    f"cql_target_action_gap must be >= 0 in Lagrange mode, got {config.cql_target_action_gap}"
                )
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
        if config.q_min is not None and config.q_max is not None and config.q_min > config.q_max:
            raise ValueError(f"q_min must be <= q_max, got q_min={config.q_min}, q_max={config.q_max}")
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
        logger.info("Setting up PSC: principal-subspace conservatism over the dataset action covariance")

        if self.is_multi_gpu and self.has_curricula_enabled():
            logger.info(f"Multi-GPU curriculum synchronization enabled across {self.gpu_world_size} GPUs")

        args = self.config
        if args.use_cnn_encoder:
            raise ValueError("PSC currently supports vector observations only; set use_cnn_encoder=False.")

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
            raise ValueError("PSC requires use_tanh=True for bounded action training.")
        env_action_scale = env._action_boundaries
        env_action_bias = torch.zeros(n_act, device=device)
        self.env_action_scale = env_action_scale
        self.env_action_bias = env_action_bias
        self.normalized_action_training = bool(args.normalized_action_training)
        if self.normalized_action_training:
            action_scale = torch.ones(n_act, device=device)
            action_bias = torch.zeros(n_act, device=device)
            self.action_space_mode = "normalized_action_training_v1"
            logger.info("PSC action semantics: actor/critic use normalized u-space [-1, 1].")
        else:
            action_scale = env_action_scale
            action_bias = env_action_bias
            self.action_space_mode = "env_scaled_action_training_v1"
            logger.info("PSC action semantics: actor/critic use legacy env-scaled action space.")

        actor_obs_keys = list(args.actor_obs_keys)
        group_names, group_indices = resolve_action_groups(args.bf_cql_action_grouping, env.robot_config.dof_names)
        self.actor = FactorizedActor(
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
            action_group_indices=group_indices,
            action_group_names=group_names,
        )
        self.bf_cql_group_names = group_names
        self.bf_cql_group_indices = group_indices

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
        logger.info(f"PSC dims: actor_obs_dim={actor_obs_dim}, critic_obs_dim={critic_obs_dim}, n_act={n_act}")
        logger.info(
            "PSC action groups: "
            + ", ".join(
                f"{name}:{list(indices)}" for name, indices in zip(group_names, group_indices, strict=True)
            )
        )

        if args.use_symmetry:
            self.symmetry_utils = SymmetryUtils(env._env)

        self._syndiag_setup()

        self._setup_psc()

        if self.is_multi_gpu:
            self._synchronize_model_parameters()

    # ------------------------------------------------------------------
    # Synergy-OOD diagnostics (syndiag) — logging only, NO loss changes.
    #
    # Hook sites:
    #   1. _update_q's existing torch.no_grad() diagnostic block exports the
    #      already-computed per-sample min(Q1,Q2)(s, a_D) and the deterministic
    #      actor action (both detached) through the return tuple. _update_q is
    #      torch.compile'd, so the tick itself cannot live inside it (file I/O
    #      and step-dependent Python branching would break/recompile the graph).
    #   2. offline_learn calls _syndiag_maybe_tick right next to the existing
    #      _compute_action_ood_stats call and merges the returned scalars into
    #      training_metrics.
    #
    # On non-tick steps the overhead is the enabled flag + one modulo check.
    # Diagnostics never touch the autograd graph, optimizers, observation
    # normalizers, or the default RNG (all math is deterministic).
    # ------------------------------------------------------------------

    def _syndiag_setup(self) -> None:
        """Build coalition structure and diagnostic-only buffers (never shared with training)."""
        self._syndiag_tick_count = 0
        self._syndiag_fail_count = 0
        self._syndiag_sigma_initialized = False
        if not self._syndiag_enabled:
            return

        cfg = self._syndiag_cfg
        n_act = self.env.robot_config.actions_dim
        num_groups = len(self.bf_cql_group_names)

        self._syndiag_coalitions = build_coalitions(
            self.bf_cql_group_names,
            cfg.max_coalitions,
            warn=logger.warning,
        )
        self._syndiag_group_dim_mask = group_dim_mask(self.bf_cql_group_indices, n_act, self.device)  # [G, A]
        self._syndiag_coalition_group_mask = coalition_group_mask(
            self._syndiag_coalitions, num_groups, self.device
        )  # [C, G]
        self._syndiag_coalition_dim_mask = (
            self._syndiag_coalition_group_mask.to(torch.float32) @ self._syndiag_group_dim_mask.to(torch.float32)
        ) > 0.5  # [C, A]
        self._syndiag_singleton_cols = singleton_columns(self._syndiag_coalitions, num_groups).to(self.device)

        pair_cols = [c for c, coal in enumerate(self._syndiag_coalitions) if len(coal.group_ids) == 2]
        triple_cols = [c for c, coal in enumerate(self._syndiag_coalitions) if len(coal.group_ids) == 3]
        self._syndiag_pair_cols = torch.tensor(pair_cols, dtype=torch.long, device=self.device)
        self._syndiag_triple_cols = torch.tensor(triple_cols, dtype=torch.long, device=self.device)
        self._syndiag_pair_group_ids = torch.tensor(
            [list(self._syndiag_coalitions[c].group_ids) for c in pair_cols] or [[0, 0]],
            dtype=torch.long,
            device=self.device,
        )

        # Per-action-dim running std of dataset actions (normalized space).
        # Dedicated diagnostic buffer: NOT the observation normalizers, never
        # checkpointed, updated only on diagnostic ticks with momentum 0.999.
        self._syndiag_sigma = torch.ones(n_act, device=self.device)
        self._syndiag_dump_dir = Path(self.log_dir) / "syndiag"

        num_named = len(self._syndiag_coalitions) - num_groups - len(pair_cols)
        logger.info(
            f"syndiag enabled: interval={cfg.interval} critic updates, dump_interval={cfg.dump_interval} ticks, "
            f"coalitions={len(self._syndiag_coalitions)} ({num_groups} singletons, {len(pair_cols)} pairs, "
            f"{num_named} named)"
        )
        logger.info("syndiag coalition names: {}".format([c.name for c in self._syndiag_coalitions]))

    def _syndiag_maybe_tick(
        self,
        data: TensorDict,
        q_data_min: torch.Tensor,
        pi_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the diagnostic tick on schedule; never crash training."""
        if not self._syndiag_enabled:
            return {}
        if self._critic_update_step % self._syndiag_cfg.interval != 0:
            return {}
        try:
            metrics = self._syndiag_tick(data, q_data_min, pi_actions)
        except Exception:
            self._syndiag_fail_count += 1
            if self._syndiag_fail_count == 1:
                logger.exception("syndiag tick failed; training is unaffected (logging only).")
            if self._syndiag_fail_count >= 3:
                self._syndiag_enabled = False
                logger.warning("syndiag disabled after 3 consecutive tick failures.")
            return {}
        self._syndiag_fail_count = 0
        return metrics

    @torch.no_grad()
    def _syndiag_tick(
        self,
        data: TensorDict,
        q_data_min: torch.Tensor,
        pi_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self._syndiag_cfg
        self._syndiag_tick_count += 1

        a_data = self._to_critic_actions(data["actions"]).float()
        a_pi = pi_actions.float()
        q_data_min = q_data_min.float()

        # 1. Group drift with a diagnostic-only running sigma_D (momentum 0.999,
        #    bootstrapped from the first tick's batch std).
        batch_sigma = a_data.std(dim=0, unbiased=False)
        if self._syndiag_sigma_initialized:
            self._syndiag_sigma.mul_(0.999).add_(batch_sigma, alpha=1.0 - 0.999)
        else:
            self._syndiag_sigma.copy_(batch_sigma)
            self._syndiag_sigma_initialized = True
        drift = compute_group_drift(a_pi, a_data, self._syndiag_sigma, self._syndiag_group_dim_mask)  # [B, G]

        # 2. Coalition values from ONE batched twin-critic forward, evaluated
        #    under the same autocast context as training so v(M) compares
        #    q_cf against the reused q_data at matching precision.
        with self._maybe_amp():
            q_cf = coalition_q_values(
                self.qnet,
                data["critic_observations"],
                a_pi,
                a_data,
                self._syndiag_coalition_dim_mask,
            )
        q_cf = q_cf.float()  # [B, C]
        v = q_cf - q_data_min[:, None]  # [B, C]
        delta = synergy_residuals(v, self._syndiag_coalition_group_mask, self._syndiag_singleton_cols)  # [B, C]

        # 3. Aggregates.
        metrics: dict[str, torch.Tensor] = {}
        for g, name in enumerate(self.bf_cql_group_names):
            metrics[f"syndiag/drift_{name}"] = drift[:, g].mean()
        for c, coalition in enumerate(self._syndiag_coalitions):
            metrics[f"syndiag/v_{coalition.name}"] = v[:, c].mean()
            metrics[f"syndiag/delta_{coalition.name}"] = delta[:, c].mean()

        coalition_sizes = self._syndiag_coalition_group_mask.sum(dim=1).to(drift.dtype)  # [C]
        block_drift = (drift @ self._syndiag_coalition_group_mask.to(drift.dtype).t()) / coalition_sizes[None, :]

        for size_name, cols in (("pairs", self._syndiag_pair_cols), ("triples", self._syndiag_triple_cols)):
            if cols.numel() == 0:
                continue
            stats = quartile_delta_stats(block_drift[:, cols], delta[:, cols])
            if stats is None:
                continue
            for q in range(1, 5):
                metrics[f"syndiag/delta_{size_name}_driftQ{q}"] = stats[f"q{q}"]
            metrics[f"syndiag/delta_{size_name}_q4_over_q1"] = stats["q4_over_q1"]
            metrics[f"syndiag/delta_{size_name}_q4_minus_q1"] = stats["q4_minus_q1"]

        if self._syndiag_pair_cols.numel() > 0:
            delta_pairs = delta[:, self._syndiag_pair_cols]
            active_frac = None
            for k in (2, 3):
                recall, active_frac = recall_top_pair(
                    delta_pairs,
                    self._syndiag_pair_group_ids,
                    drift,
                    top_k=k,
                    delta_min=cfg.delta_min,
                )
                if recall is not None:
                    metrics[f"syndiag/recall_pair_top{k}"] = recall
            if active_frac is not None:
                metrics["syndiag/active_frac"] = active_frac

        superadd = superadditivity_quad(delta, self._syndiag_coalitions)
        if superadd is not None:
            metrics["syndiag/superadditivity_quad"] = superadd

        # 4. Raw dump for offline counterfactual replay (Part B).
        if (
            cfg.dump_interval > 0
            and self._syndiag_tick_count % cfg.dump_interval == 0
            and self.is_main_process
        ):
            self._syndiag_dump(
                data,
                a_data=a_data,
                a_pi=a_pi,
                drift=drift,
                v=v,
                delta=delta,
                q_data_min=q_data_min,
                q_cf=q_cf,
            )

        return metrics

    @torch.no_grad()
    def _syndiag_dump(
        self,
        data: TensorDict,
        *,
        a_data: torch.Tensor,
        a_pi: torch.Tensor,
        drift: torch.Tensor,
        v: torch.Tensor,
        delta: torch.Tensor,
        q_data_min: torch.Tensor,
        q_cf: torch.Tensor,
    ) -> None:
        """Write one compressed npz consumed by tools/eval_counterfactual_gap.py."""
        cfg = self._syndiag_cfg
        rows = min(int(a_data.shape[0]), cfg.dump_max_rows)  # deterministic subsample: first rows

        top_k = min(cfg.dump_topk, delta.shape[1])
        top_delta, top_cols = delta[:rows].topk(top_k, dim=1)  # [rows, K]
        top_masks = self._syndiag_coalition_dim_mask[top_cols]  # [rows, K, A]
        a_cf_top = torch.where(top_masks, a_pi[:rows, None, :], a_data[:rows, None, :])
        a_cf_top_env = self._to_env_actions(a_cf_top)

        dataset_index = data.get("dataset_index", None)
        if isinstance(dataset_index, torch.Tensor):
            dataset_index_np = dataset_index[:rows].detach().cpu().numpy()
        else:
            dataset_index_np = np.full((rows,), -1, dtype=np.int64)
        raw_observations = data.get("syndiag_raw_observations", None)
        if isinstance(raw_observations, torch.Tensor):
            raw_observations_np = raw_observations[:rows].detach().float().cpu().numpy()
        else:
            raw_observations_np = np.zeros((rows, 0), dtype=np.float32)

        self._syndiag_dump_dir.mkdir(parents=True, exist_ok=True)
        dump_path = self._syndiag_dump_dir / f"dump_step{self.global_step:08d}.npz"
        np.savez_compressed(
            dump_path,
            schema_version=np.int64(1),
            global_step=np.int64(self.global_step),
            dataset_path=str(self._offline_dataset_path),
            group_names=np.array(self.bf_cql_group_names),
            group_dim_mask=self._syndiag_group_dim_mask.cpu().numpy(),
            coalition_names=np.array([c.name for c in self._syndiag_coalitions]),
            coalition_group_mask=self._syndiag_coalition_group_mask.cpu().numpy(),
            coalition_dim_mask=self._syndiag_coalition_dim_mask.cpu().numpy(),
            dataset_index=dataset_index_np,
            observations_raw=raw_observations_np,
            actions_raw=data["actions"][:rows].detach().float().cpu().numpy(),
            a_pi_norm=a_pi[:rows].cpu().numpy(),
            a_pi_env=self._to_env_actions(a_pi[:rows]).cpu().numpy(),
            drift=drift[:rows].cpu().numpy(),
            v=v[:rows].cpu().numpy(),
            delta=delta[:rows].cpu().numpy(),
            q_data_min=q_data_min[:rows].cpu().numpy(),
            q_cf=q_cf[:rows].cpu().numpy(),
            top_coalition_ids=top_cols.cpu().numpy(),
            top_delta=top_delta.cpu().numpy(),
            a_cf_env_top=a_cf_top_env.cpu().numpy(),
            sigma=self._syndiag_sigma.cpu().numpy(),
        )
        logger.info(f"syndiag: wrote raw dump {dump_path} ({rows} rows, {top_k} coalitions/sample)")

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
        logger.info(f"Synchronized PSC model parameters across {self.gpu_world_size} GPUs")

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

    def _to_normalized_actions(self, actions: torch.Tensor) -> torch.Tensor:
        action_scale = self.env_action_scale.to(device=actions.device, dtype=actions.dtype)
        action_bias = self.env_action_bias.to(device=actions.device, dtype=actions.dtype)
        return ((actions - action_bias) / (action_scale + 1e-6)).clamp(-1.0, 1.0)

    def _to_critic_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.normalized_action_training:
            return self._to_normalized_actions(actions)
        return actions

    def _to_env_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.normalized_action_training:
            return actions
        action_scale = self.env_action_scale.to(device=actions.device, dtype=actions.dtype)
        action_bias = self.env_action_bias.to(device=actions.device, dtype=actions.dtype)
        return actions * action_scale + action_bias

    def _sync_actor_action_space_buffers(self) -> None:
        with torch.no_grad():
            if self.normalized_action_training:
                self.actor.action_scale.fill_(1.0)
                self.actor.action_bias.zero_()
            else:
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

    def _counterfactual_group_actions(
        self,
        base_actions: torch.Tensor,
        group_indices: tuple[int, ...],
        group_actions: torch.Tensor,
    ) -> torch.Tensor:
        counterfactual_actions = base_actions.clone()
        counterfactual_actions[:, list(group_indices)] = group_actions
        return counterfactual_actions

    def _sample_actor_ood_group_mask(
        self,
        num_rows: int,
        base_group_idx: int,
        num_groups: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        actor_group_count = min(self._ood_actor_num, num_groups)
        selected_group_mask = torch.zeros(num_rows, num_groups, device=device, dtype=torch.bool)
        selected_group_mask[:, base_group_idx] = True

        num_extra_groups = actor_group_count - 1
        if num_extra_groups > 0:
            scores = torch.rand(num_rows, num_groups, device=device)
            scores[:, base_group_idx] = -1.0
            extra_group_ids = scores.topk(num_extra_groups, dim=1).indices
            selected_group_mask.scatter_(1, extra_group_ids, True)
        return selected_group_mask

    def _counterfactual_actor_group_actions(
        self,
        base_actions: torch.Tensor,
        actor_actions: torch.Tensor,
        selected_group_mask: torch.Tensor,
    ) -> torch.Tensor:
        counterfactual_actions = base_actions.clone()
        for group_idx, group_indices in enumerate(self.bf_cql_group_indices):
            row_ids = torch.nonzero(selected_group_mask[:, group_idx], as_tuple=False).flatten()
            if row_ids.numel() == 0:
                continue
            col_ids = torch.as_tensor(group_indices, device=base_actions.device, dtype=torch.long)
            counterfactual_actions[row_ids[:, None], col_ids[None, :]] = actor_actions[
                row_ids[:, None], col_ids[None, :]
            ]
        return counterfactual_actions


    @staticmethod
    def _psc_validate_basis(basis: dict, n_act: int, run_space: str) -> None:
        """Hard-fail on any basis/run mismatch (see PSC_NOTES.md)."""
        for key in ("mu", "U", "eigvals", "meta"):
            if key not in basis:
                raise ValueError(f"PSC basis is missing key '{key}'.")
        U = basis["U"]
        if U.ndim != 2 or U.shape[0] != U.shape[1]:
            raise ValueError(f"PSC basis U must be square, got {tuple(U.shape)}.")
        if U.shape[0] != n_act:
            raise ValueError(f"PSC basis action_dim {U.shape[0]} != agent action_dim {n_act}.")
        eye_err = (U.T @ U - torch.eye(n_act, dtype=U.dtype, device=U.device)).abs().max()
        if float(eye_err) >= 1e-4:
            raise ValueError(f"PSC basis U is not orthonormal (||U^T U - I||_inf = {float(eye_err):.2e}).")
        meta = basis["meta"]
        if int(meta.get("action_dim", -1)) != n_act:
            raise ValueError(f"PSC basis meta action_dim {meta.get('action_dim')} != agent action_dim {n_act}.")
        if str(meta.get("space")) != run_space:
            raise ValueError(
                f"PSC basis was measured in '{meta.get('space')}' action space but this run uses "
                f"'{run_space}' (normalized_action_training mismatch)."
            )

    def _psc_init_from_basis(self, basis: dict) -> None:
        """Register the eigen-basis and derived block structure on the agent."""
        psc = self.config.psc
        device = self.device
        self._psc_U = basis["U"].to(device=device, dtype=torch.float32)
        self._psc_mu = basis["mu"].to(device=device, dtype=torch.float32)
        self._psc_eigvals = basis["eigvals"].to(device=device, dtype=torch.float32).clamp_min(0.0)
        self._psc_basis_meta = dict(basis["meta"])
        n_act = self._psc_U.shape[0]

        block_sizes = [int(s) for s in psc.block_sizes]
        if sum(block_sizes) != n_act:
            raise ValueError(f"psc.block_sizes {block_sizes} must sum to action_dim {n_act}.")
        starts = [0]
        for size in block_sizes:
            starts.append(starts[-1] + size)
        # eigen-index blocks in DESCENDING-eigenvalue order: block 0 = top directions
        self._psc_block_slices = [(starts[i], starts[i + 1]) for i in range(len(block_sizes))]
        self._psc_block_sizes = block_sizes

        if psc.rand_scale_mode == "sqrt_eig_floored":
            sqrt_eig = self._psc_eigvals.sqrt()
            floor = torch.quantile(sqrt_eig, float(psc.scale_floor_quantile))
            self._psc_rand_scales = torch.maximum(sqrt_eig, floor)
        else:
            raise ValueError(f"Unknown psc.rand_scale_mode={psc.rand_scale_mode!r}")

        # projection energy of each action dim onto each eigen-block: used to
        # partition per-dim policy log-probs across blocks (exact for axis-aligned
        # bases; v0 approximation otherwise — see PSC_NOTES.md)
        energy_rows = []
        for block_start, block_end in self._psc_block_slices:
            energy_rows.append((self._psc_U[:, block_start:block_end] ** 2).sum(dim=1))
        self._psc_block_energy = torch.stack(energy_rows, dim=0)  # [G, A]

        action_scale = self.actor.action_scale.to(device=device, dtype=torch.float32)
        action_bias = self.actor.action_bias.to(device=device, dtype=torch.float32)
        self._psc_action_low = action_bias - action_scale
        self._psc_action_high = action_bias + action_scale

        ratios = self._psc_eigvals / self._psc_eigvals.sum().clamp_min(1e-12)
        p = ratios.clamp_min(1e-12)
        self._psc_effective_rank = float(torch.exp(-(p * p.log()).sum()))
        self._psc_rand_scale_mean = (float(psc.rand_range_mult) * self._psc_rand_scales).mean()
        logger.info(
            f"PSC basis loaded from {psc.basis_path}: space={self._psc_basis_meta.get('space')}, "
            f"effective_rank={self._psc_effective_rank:.2f}/{n_act}, "
            f"blocks={self._psc_block_sizes} (descending eigenvalue order), "
            f"rand_scale_mean={float(self._psc_rand_scale_mean):.4f}"
        )

    def _setup_psc(self) -> None:
        psc = self.config.psc
        n_act = int(self.actor.action_scale.numel())
        basis = torch.load(psc.basis_path, map_location=self.device, weights_only=False)
        run_space = "normalized" if self.normalized_action_training else "env"
        self._psc_validate_basis(basis, n_act=n_act, run_space=run_space)
        self._psc_init_from_basis(basis)
        if psc.recompute_check:
            self._psc_recompute_check()

    def _psc_recompute_check(self, max_samples: int = 100_000) -> None:
        """Recompute Sigma_D on a deterministic subsample; hard-fail if the top-k
        eigenspace disagrees with the loaded basis (projection energy <= 0.95)."""
        if self._offline_gpu_cache is not None:
            storage_actions = self._offline_gpu_cache._storage["actions"]
            raw = storage_actions[: min(max_samples, storage_actions.shape[0])]
        elif self._offline_dataset_reader is not None:
            block = self._offline_dataset_reader.read_block(
                start=0, block_size=min(max_samples, self._offline_dataset_reader.num_samples)
            )
            raw = block["actions"].to(self.device)
        else:
            logger.warning("PSC recompute_check skipped: no offline dataset source available.")
            return

        actions = self._to_critic_actions(raw.to(dtype=torch.float32)).to(torch.float64)
        centered = actions - actions.mean(dim=0)
        sigma = centered.T @ centered / max(actions.shape[0] - 1, 1)
        eigvals, eigvecs = torch.linalg.eigh(sigma)  # ascending
        k = self._psc_block_sizes[0]
        v_new_topk = eigvecs[:, -k:]  # top-k of the recomputed spectrum
        basis_topk = self._psc_U[:, :k].to(torch.float64)
        energy = float(((basis_topk.T @ v_new_topk) ** 2).sum() / k)
        if energy <= 0.95:
            raise RuntimeError(
                f"PSC recompute_check failed: top-{k} projection energy between the recomputed "
                f"covariance ({actions.shape[0]} samples) and the loaded basis is {energy:.4f} <= 0.95. "
                "The basis is stale for this dataset — re-run scripts/psc_spectrum.py."
            )
        logger.info(f"PSC recompute_check passed: top-{k} projection energy = {energy:.4f}")

    def _psc_counterfactual_actor_actions(
        self,
        u_data: torch.Tensor,
        u_actor: torch.Tensor,
        selected_group_mask: torch.Tensor,
        psc_U: torch.Tensor,
        psc_mu: torch.Tensor,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> torch.Tensor:
        """Splice actor eigen-coordinates into the dataset anchor on selected blocks,
        map back to action space, clamp to the training-space action bounds."""
        u_cf = u_data.clone()
        for group_idx, (block_start, block_end) in enumerate(self._psc_block_slices):
            row_ids = torch.nonzero(selected_group_mask[:, group_idx], as_tuple=False).flatten()
            if row_ids.numel() == 0:
                continue
            u_cf[row_ids, block_start:block_end] = u_actor[row_ids, block_start:block_end]
        return (u_cf @ psc_U.t() + psc_mu).clamp(action_low, action_high)

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
            rewards = reward_scale * data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            # PSC spec: truncated ends (timeout / segment_ends) must BOOTSTRAP — recurring
            # regression, do not revert to (~dones) only.
            bootstrap = (truncations | ~dones).float()
            alpha = self.log_alpha.exp().detach()

            with torch.no_grad():
                discount = args.gamma ** data["next"]["effective_n_steps"]
                rewards_ = rewards.view(-1)
                bootstrap_ = bootstrap.view(-1)
                discount_ = discount.view(-1)

                if args.cql_max_target_backup:
                    batch_size = next_observations.shape[0]
                    num_backup_actions = args.cql_max_target_backup_samples
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
                    next_actions, next_log_probs = self.actor.get_actions_and_log_probs(expanded_next_obs)
                    next_q1_target, next_q2_target = self.qnet_target(
                        expanded_next_critic_obs,
                        next_actions,
                    )
                    next_q1_target = next_q1_target.view(batch_size, num_backup_actions)
                    next_q2_target = next_q2_target.view(batch_size, num_backup_actions)
                    next_log_probs = next_log_probs.view(batch_size, num_backup_actions)
                    next_target_min_q_all = torch.minimum(next_q1_target, next_q2_target)
                    next_target_min_q, max_target_indices = next_target_min_q_all.max(dim=1)
                    next_log_probs = next_log_probs.gather(
                        dim=1,
                        index=max_target_indices.unsqueeze(1),
                    ).squeeze(1)
                else:
                    next_actions, next_log_probs = self.actor.get_actions_and_log_probs(next_observations)
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
                if args.q_min is not None or args.q_max is not None:
                    q_target = q_target.clamp(min=args.q_min, max=args.q_max)
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

            q_data_mean = torch.minimum(q1, q2).mean()
            rand_q_mean = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            curr_q_mean = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            next_q_mean = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            curr_logp_mean = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            next_logp_mean = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            random_density_mean = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            with torch.no_grad():
                pi_actions_det = self.actor(observations)[0]
                q1_pi_det, q2_pi_det = self.qnet(critic_observations, pi_actions_det)
                q_pi_minus_q_data = (
                    torch.minimum(q1_pi_det, q2_pi_det) - torch.minimum(q1.detach(), q2.detach())
                ).mean()
                # syndiag hook: export already-computed per-sample quantities so the
                # periodic diagnostics in offline_learn can reuse them without
                # re-evaluating the critic on dataset actions (detached, logging only).
                syndiag_q_data_min = torch.minimum(q1, q2).detach()
                syndiag_pi_actions = pi_actions_det.detach()

            psc_block_gaps = torch.zeros(len(self._psc_block_slices), device=self.device, dtype=bellman_loss.dtype)
            if self._cql_weight > 0.0:
                batch_size = dataset_actions.shape[0]
                num_repeat = self._num_repeat_actions
                expanded_obs = observations[:, None, :].expand(batch_size, num_repeat, -1).reshape(
                    batch_size * num_repeat,
                    -1,
                )
                expanded_critic_obs = critic_observations[:, None, :].expand(
                    batch_size,
                    num_repeat,
                    -1,
                ).reshape(batch_size * num_repeat, -1)
                expanded_next_obs = next_observations[:, None, :].expand(
                    batch_size,
                    num_repeat,
                    -1,
                ).reshape(batch_size * num_repeat, -1)
                expanded_dataset_actions = dataset_actions[:, None, :].expand(
                    batch_size,
                    num_repeat,
                    -1,
                ).reshape(batch_size * num_repeat, -1)

                # PSC: the ONLY change vs BF-CQL — counterfactual blocks live in the
                # eigen-basis of the dataset action covariance. u = U^T (a - mu);
                # per block, perturb u[S_g] (rand/curr/next) and anchor u[-S_g] to the
                # rotated dataset action; map back and clamp to training-space bounds.
                with torch.no_grad():
                    curr_actions, curr_logp_per_dim = self.actor.get_actions_and_log_prob_per_dim(expanded_obs)
                    next_actions_rep, next_logp_per_dim = self.actor.get_actions_and_log_prob_per_dim(
                        expanded_next_obs
                    )
                    # v0 density partition (PSC_NOTES.md): per-dim policy log-probs are
                    # distributed across eigen-blocks by projection energy; exact for
                    # axis-aligned bases (reduces to BF-CQL group log-probs when U=I).
                    block_energy = self._psc_block_energy.to(dtype=curr_logp_per_dim.dtype)
                    curr_group_logps_stacked = curr_logp_per_dim @ block_energy.t()
                    next_group_logps_stacked = next_logp_per_dim @ block_energy.t()

                psc_U = self._psc_U.to(dtype=dataset_actions.dtype)
                psc_mu = self._psc_mu.to(dtype=dataset_actions.dtype)
                action_low = self._psc_action_low.to(dtype=dataset_actions.dtype)
                action_high = self._psc_action_high.to(dtype=dataset_actions.dtype)
                u_data = (expanded_dataset_actions - psc_mu) @ psc_U
                u_curr = (curr_actions - psc_mu) @ psc_U
                u_next = (next_actions_rep - psc_mu) @ psc_U

                cql1_loss_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                cql2_loss_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                rand_q_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                curr_q_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                next_q_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                curr_logp_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                next_logp_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                random_density_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                psc_gap_list = []

                num_groups_int = len(self._psc_block_slices)
                for group_idx, (block_start, block_end) in enumerate(self._psc_block_slices):
                    group_dim = self._psc_block_sizes[group_idx]
                    block_rand_scale = (
                        float(args.psc.rand_range_mult)
                        * self._psc_rand_scales[block_start:block_end]
                    ).to(dtype=dataset_actions.dtype)
                    rand_group_coeffs = torch.empty(
                        batch_size * num_repeat,
                        group_dim,
                        device=self.device,
                        dtype=dataset_actions.dtype,
                    ).uniform_(-1.0, 1.0)
                    rand_group_coeffs = rand_group_coeffs * block_rand_scale
                    # exact uniform density over the block; the rotation is
                    # volume-preserving (|det U| = 1) so no extra constant appears
                    random_density = (
                        torch.tensor(math.log(0.5) * group_dim, device=self.device, dtype=dataset_actions.dtype)
                        - torch.log(block_rand_scale + 1e-6).sum()
                    )

                    u_rand = u_data.clone()
                    u_rand[:, block_start:block_end] = rand_group_coeffs
                    rand_counterfactual_actions = (u_rand @ psc_U.t() + psc_mu).clamp(action_low, action_high)

                    selected_group_mask = self._sample_actor_ood_group_mask(
                        num_rows=batch_size * num_repeat,
                        base_group_idx=group_idx,
                        num_groups=num_groups_int,
                        device=self.device,
                    )
                    curr_counterfactual_actions = self._psc_counterfactual_actor_actions(
                        u_data, u_curr, selected_group_mask, psc_U, psc_mu, action_low, action_high
                    )
                    next_counterfactual_actions = self._psc_counterfactual_actor_actions(
                        u_data, u_next, selected_group_mask, psc_U, psc_mu, action_low, action_high
                    )
                    selected_group_mask_float = selected_group_mask.to(dtype=curr_group_logps_stacked.dtype)
                    curr_actor_logp = (curr_group_logps_stacked * selected_group_mask_float).sum(dim=1)
                    next_actor_logp = (next_group_logps_stacked * selected_group_mask_float).sum(dim=1)

                    counterfactual_actions = torch.cat(
                        [
                            rand_counterfactual_actions,
                            curr_counterfactual_actions,
                            next_counterfactual_actions,
                        ],
                        dim=0,
                    )
                    counterfactual_critic_obs = expanded_critic_obs.repeat(3, 1)
                    q1_counterfactual, q2_counterfactual = self.qnet(
                        counterfactual_critic_obs,
                        counterfactual_actions,
                    )
                    q1_rand, q1_curr, q1_next = q1_counterfactual.chunk(3, dim=0)
                    q2_rand, q2_curr, q2_next = q2_counterfactual.chunk(3, dim=0)

                    q1_rand = q1_rand.view(batch_size, num_repeat)
                    q2_rand = q2_rand.view(batch_size, num_repeat)
                    q1_curr = q1_curr.view(batch_size, num_repeat)
                    q2_curr = q2_curr.view(batch_size, num_repeat)
                    q1_next = q1_next.view(batch_size, num_repeat)
                    q2_next = q2_next.view(batch_size, num_repeat)
                    curr_actor_logp = curr_actor_logp.view(batch_size, num_repeat)
                    next_actor_logp = next_actor_logp.view(batch_size, num_repeat)

                    q1_terms = torch.cat(
                        [
                            q1_rand - random_density,
                            q1_curr - curr_actor_logp,
                            q1_next - next_actor_logp,
                        ],
                        dim=1,
                    )
                    q2_terms = torch.cat(
                        [
                            q2_rand - random_density,
                            q2_curr - curr_actor_logp,
                            q2_next - next_actor_logp,
                        ],
                        dim=1,
                    )

                    cql1_group_loss = (
                        torch.logsumexp(q1_terms / self._temperature, dim=1) * self._temperature - q1
                    ).mean()
                    cql2_group_loss = (
                        torch.logsumexp(q2_terms / self._temperature, dim=1) * self._temperature - q2
                    ).mean()
                    cql1_loss_total = cql1_loss_total + cql1_group_loss
                    cql2_loss_total = cql2_loss_total + cql2_group_loss
                    psc_gap_list.append(0.5 * (cql1_group_loss + cql2_group_loss).detach())
                    rand_q_total = rand_q_total + 0.5 * (
                        (q1_rand - random_density).mean() + (q2_rand - random_density).mean()
                    )
                    curr_q_total = curr_q_total + 0.5 * (q1_curr.mean() + q2_curr.mean())
                    next_q_total = next_q_total + 0.5 * (q1_next.mean() + q2_next.mean())
                    curr_logp_total = curr_logp_total + curr_actor_logp.mean()
                    next_logp_total = next_logp_total + next_actor_logp.mean()
                    random_density_total = random_density_total + random_density.to(dtype=bellman_loss.dtype)

                psc_block_gaps = torch.stack(psc_gap_list).to(dtype=bellman_loss.dtype)
                num_groups = float(len(self._psc_block_slices))
                cql1_loss = cql1_loss_total / num_groups
                cql2_loss = cql2_loss_total / num_groups
                cql_gap = 0.5 * (cql1_loss + cql2_loss)
                rand_q_mean = rand_q_total / num_groups
                curr_q_mean = curr_q_total / num_groups
                next_q_mean = next_q_total / num_groups
                curr_logp_mean = curr_logp_total / num_groups
                next_logp_mean = next_logp_total / num_groups
                random_density_mean = random_density_total / num_groups

                if args.use_lagrange and self.log_cql_alpha is not None:
                    cql_alpha = self.log_cql_alpha.exp().detach().clamp(max=args.cql_lagrange_max)
                    target_gap = torch.tensor(args.cql_target_action_gap, device=self.device, dtype=bellman_loss.dtype)
                    conservative_loss = cql_alpha * self._cql_weight * 0.5 * (
                        (cql1_loss - target_gap) + (cql2_loss - target_gap)
                    )
                else:
                    conservative_loss = self._cql_weight * 0.5 * (cql1_loss + cql2_loss)
            else:
                conservative_loss = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                cql_gap = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)

            q_loss = bellman_loss + conservative_loss

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
            alpha_loss.detach(),
            conservative_loss.detach(),
            bellman_loss.detach(),
            cql_gap.detach(),
            q_data_mean.detach(),
            q_pi_minus_q_data.detach(),
            rand_q_mean.detach(),
            curr_q_mean.detach(),
            next_q_mean.detach(),
            curr_logp_mean.detach(),
            next_logp_mean.detach(),
            random_density_mean.detach(),
            syndiag_q_data_min,
            syndiag_pi_actions,
            psc_block_gaps,
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

        # syndiag (logging only): keep references to the raw, pre-normalization
        # observations and the sampler-provided dataset row index. Zero-copy;
        # apply_observation_normalization reassigns keys instead of mutating.
        syndiag_enabled = getattr(self, "_syndiag_enabled", False)
        syndiag_raw_observations = batch["observations"] if syndiag_enabled else None
        syndiag_dataset_index = batch.get("dataset_index") if syndiag_enabled else None

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
        data = TensorDict(
            {
                "observations": batch["observations"],
                "actions": batch["actions"],
                "critic_observations": batch["critic_observations"],
                "next": next_batch,
            },
            batch_size=effective_batch_size,
            device=self.device,
        )
        if syndiag_enabled:
            if (
                isinstance(syndiag_dataset_index, torch.Tensor)
                and int(syndiag_dataset_index.shape[0]) == effective_batch_size
            ):
                data["dataset_index"] = syndiag_dataset_index.to(device=self.device, dtype=torch.long)
            else:
                # e.g. symmetry augmentation replaced the batch; index unavailable.
                data["dataset_index"] = torch.full(
                    (effective_batch_size,), -1, dtype=torch.long, device=self.device
                )
            data["syndiag_raw_observations"] = syndiag_raw_observations
        return data

    def load(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return

        torch_checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        checkpoint_action_mode = torch_checkpoint.get("action_space_mode", "legacy")
        compatible_checkpoint_action_mode = (
            "env_scaled_action_training_v1" if checkpoint_action_mode == "legacy" else checkpoint_action_mode
        )
        expected_action_mode = getattr(self, "action_space_mode", "normalized_action_training_v1")
        if compatible_checkpoint_action_mode != expected_action_mode:
            logger.warning(
                "Loading a legacy checkpoint with different PSC action semantics "
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
                f"Checkpoint missing required PSC keys: {missing_required}. "
                "Expected a PSC checkpoint."
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

        psc_basis = torch_checkpoint.get("psc_basis")
        if psc_basis is not None:
            run_space = "normalized" if self.normalized_action_training else "env"
            self._psc_validate_basis(
                {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in psc_basis.items()},
                n_act=int(self.actor.action_scale.numel()),
                run_space=run_space,
            )
            self._psc_init_from_basis(
                {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in psc_basis.items()}
            )
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
                "Offline PSC gradient updates sample only the fixed dataset. "
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
                        syndiag_q_data_min,
                        syndiag_pi_actions,
                        psc_block_gaps,
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
                    # syndiag hook site 2: periodic synergy-OOD diagnostics
                    # (logging only; {} on non-tick steps).
                    syndiag_stats = self._syndiag_maybe_tick(data, syndiag_q_data_min, syndiag_pi_actions)
                    psc_metric_dict = {
                        f"psc/gap_block_{block_idx}": psc_block_gaps[block_idx]
                        for block_idx in range(psc_block_gaps.numel())
                    }
                    psc_metric_dict["psc/rand_scale_mean"] = self._psc_rand_scale_mean
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
                            "bf_cql/ood_actor_num": torch.tensor(
                                float(args.ood_actor_num),
                                device=self.device,
                            ),
                            "is_actor_warmup": float(is_actor_warmup),
                            "is_actor_update_step": float(is_actor_update_step),
                            **action_ood_stats,
                            **psc_metric_dict,
                            **syndiag_stats,
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
                    logger.info(f"Saving PSC model at global step {self.global_step}")
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
        metadata["algo"] = "psc"
        metadata["bf_cql_action_grouping"] = self.config.bf_cql_action_grouping
        metadata["psc_basis"] = {
            "U": self._psc_U.detach().cpu(),
            "mu": self._psc_mu.detach().cpu(),
            "eigvals": self._psc_eigvals.detach().cpu(),
            "meta": dict(self._psc_basis_meta),
        }
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
        env_action_scale = copy.deepcopy(self.env_action_scale).to("cpu")
        env_action_bias = copy.deepcopy(self.env_action_bias).to("cpu")
        normalized_action_training = bool(self.normalized_action_training)

        class ActorWrapper(nn.Module):
            def __init__(
                self,
                actor,
                obs_normalizer,
                env_action_scale,
                env_action_bias,
                normalized_action_training,
            ):
                super().__init__()
                self.actor = actor
                self.obs_normalizer = obs_normalizer
                self.normalized_action_training = normalized_action_training
                self.register_buffer("env_action_scale", env_action_scale)
                self.register_buffer("env_action_bias", env_action_bias)

            def forward(self, actor_obs):
                if self.obs_normalizer is not None:
                    normalized_obs = self.obs_normalizer(actor_obs, update=False)
                else:
                    normalized_obs = actor_obs
                actions = self.actor(normalized_obs)[0]
                if not self.normalized_action_training:
                    return actions
                return actions * self.env_action_scale + self.env_action_bias

        return ActorWrapper(
            actor,
            obs_normalizer if self.obs_normalization else None,
            env_action_scale,
            env_action_bias,
            normalized_action_training,
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
