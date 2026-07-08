"""SYNC-QL: drift-gated CFCQL for BF-CQL.

SYNC-QL reuses BF-CQL's factorized actor, global twin critic, replay loader,
normalization, target networks, and optimizers.  Its conservative term keeps the
BF-CQL/CFCQL counterfactual structure unchanged, but replaces the full group sum
with a per-sample drift-gated sum:

    sum_g CQL_g(s, a_D)  ->  sum_{g: d_g(s,a_D) >= delta} CQL_g(s, a_D)

The inner logsumexp over random/current/next group counterfactuals, Lagrange
handling, and normalized action training are unchanged from BF-CQL.  Groups with
small actor-vs-dataset drift are skipped, and each sample is normalized by its
number of active groups instead of the total number of groups.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import tqdm
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.bf_cql.bf_cql import DoubleQCritic, FactorizedActor
from holosoma.agents.bf_cql.bf_cql_agent import BFCQLAgent, BFCQLEnv
from holosoma.agents.bf_cql.bf_cql_utils import save_params
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.config_types.algo import BFCQLConfig
from holosoma.data.hdf5_offline_dataset import GPUTransitionCache, HDF5BlockReader, RAMShuffleBuffer
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.average_meters import TensorAverageMeterDict
from holosoma.utils.safe_torch_import import F, TensorDict, TensorboardSummaryWriter, optim, torch


def build_group_to_action_mask(
    group_indices: list[tuple[int, ...]],
    action_dim: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Return a [G, A] boolean mask mapping action groups to action dimensions."""

    group_to_action = torch.zeros(len(group_indices), action_dim, dtype=torch.bool, device=device)
    for group_idx, indices in enumerate(group_indices):
        group_to_action[group_idx, list(indices)] = True
    return group_to_action


def counterfactual_actions_from_group_masks(
    dataset_actions: torch.Tensor,
    actor_actions: torch.Tensor,
    group_masks: torch.Tensor,
    group_to_action_mask: torch.Tensor,
) -> torch.Tensor:
    """Construct a_cf(M)=(a_pi^M, a_D^{-M}) for each per-sample group mask.

    Parameters
    ----------
    dataset_actions:
        Dataset action a_D with shape [B, A].
    actor_actions:
        Actor action a_pi with shape [B, A].
    group_masks:
        Boolean selected-group masks with shape [B, N, G].
    group_to_action_mask:
        Boolean group-to-action mask with shape [G, A].

    Returns
    -------
    torch.Tensor
        Counterfactual actions with shape [B, N, A].
    """

    action_masks = torch.matmul(
        group_masks.float(),
        group_to_action_mask.to(device=group_masks.device, dtype=torch.float32),
    )
    action_masks = action_masks.to(dtype=torch.bool)
    return torch.where(action_masks, actor_actions[:, None, :], dataset_actions[:, None, :])


def synergy_residual(
    q_data: torch.Tensor,
    q_block: torch.Tensor,
    q_singletons: torch.Tensor,
    singleton_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Delta(M)=v(M)-sum_{g in M}v({g}).

    q_data is detached by the caller when the residual is used as a conservative
    penalty.  That keeps this term focused on chosen counterfactual Q-values
    instead of re-weighting the dataset-anchor term already handled by CFCQL.
    """

    v_block = q_block - q_data
    v_singletons = (q_singletons.detach() - q_data[:, None]) * singleton_valid.to(q_singletons.dtype)
    v_singleton_sum = v_singletons.sum(dim=1)
    return v_block - v_singleton_sum, v_block, v_singleton_sum


class SyncCQLAgent(BaseAlgo):
    """Standalone SYNC-QL agent using BF-CQL-compatible modules without class inheritance."""

    config: BFCQLConfig
    env: BFCQLEnv  # type: ignore[assignment]
    actor: FactorizedActor
    qnet: DoubleQCritic
    qnet_target: DoubleQCritic

    _maybe_amp = BFCQLAgent._maybe_amp
    _synchronize_model_parameters = BFCQLAgent._synchronize_model_parameters
    _all_reduce_model_grads = BFCQLAgent._all_reduce_model_grads
    _soft_update_q_target = BFCQLAgent._soft_update_q_target
    _to_normalized_actions = BFCQLAgent._to_normalized_actions
    _to_env_actions = BFCQLAgent._to_env_actions
    _compute_action_ood_stats = BFCQLAgent._compute_action_ood_stats
    _counterfactual_group_actions = BFCQLAgent._counterfactual_group_actions
    _sample_actor_ood_group_mask = BFCQLAgent._sample_actor_ood_group_mask
    _counterfactual_actor_group_actions = BFCQLAgent._counterfactual_actor_group_actions
    _update_cql_lagrange = BFCQLAgent._update_cql_lagrange
    _sample_offline_batch = BFCQLAgent._sample_offline_batch
    __del__ = BFCQLAgent.__del__
    get_example_obs = BFCQLAgent.get_example_obs
    get_inference_policy = BFCQLAgent.get_inference_policy
    _eval_termination_reason_flags = BFCQLAgent._eval_termination_reason_flags
    _eval_stop_reason_for_env = BFCQLAgent._eval_stop_reason_for_env
    actor_onnx_wrapper = BFCQLAgent.actor_onnx_wrapper
    export = BFCQLAgent.export
    evaluate_policy = BFCQLAgent.evaluate_policy
    evaluate_one_episode = BFCQLAgent.evaluate_one_episode
    evaluate_vectorized_episodes = BFCQLAgent.evaluate_vectorized_episodes

    def __init__(
        self,
        env: BaseTask,
        config: BFCQLConfig,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict | None = None,
    ):
        wrapped_env = BFCQLEnv(env, config.actor_obs_keys, config.critic_obs_keys)
        BaseAlgo.__init__(self, wrapped_env, config, device, multi_gpu_cfg)  # type: ignore[arg-type]

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

        if config.cql_num_action_samples <= 0:
            raise ValueError(f"cql_num_action_samples must be > 0, got {config.cql_num_action_samples}")
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
        BFCQLAgent.setup(self)
        self._setup_sync_cql()

    def _setup_sync_cql(self) -> None:
        sync = self.config.sync_cql
        num_groups = len(self.bf_cql_group_indices)
        action_dim = int(self.env.robot_config.actions_dim)

        if sync.K < 0:
            raise ValueError(f"sync_cql.K must be >= 0, got {sync.K}")
        if sync.delta_threshold < 0.0:
            raise ValueError(f"sync_cql.delta_threshold must be >= 0, got {sync.delta_threshold}")
        if sync.alpha2 < 0.0:
            raise ValueError(f"sync_cql.alpha2 must be >= 0, got {sync.alpha2}")
        if sync.tau_syn < 0.0:
            raise ValueError(f"sync_cql.tau_syn must be >= 0, got {sync.tau_syn}")
        if sync.lambda_cf < 0.0 or sync.lambda_cf > 1.0:
            raise ValueError(f"sync_cql.lambda_cf must be in [0, 1], got {sync.lambda_cf}")
        if sync.drift_ema < 0.0 or sync.drift_ema >= 1.0:
            raise ValueError(f"sync_cql.drift_ema must be in [0, 1), got {sync.drift_ema}")
        if sync.drift_std_momentum < 0.0 or sync.drift_std_momentum >= 1.0:
            raise ValueError(
                f"sync_cql.drift_std_momentum must be in [0, 1), got {sync.drift_std_momentum}"
            )

        self.sync_group_to_action_mask = build_group_to_action_mask(
            self.bf_cql_group_indices,
            action_dim,
            device=self.device,
        )
        self.sync_action_std = torch.ones(action_dim, device=self.device)
        self.sync_group_drift_ema = torch.zeros(num_groups, device=self.device)
        self.log_sync_alpha2: torch.Tensor | None = None
        self.sync_alpha2_optimizer: optim.Optimizer | None = None
        if sync.alpha2_lagrange:
            init_alpha2 = max(float(sync.alpha2), 1e-8)
            self.log_sync_alpha2 = torch.tensor([math.log(init_alpha2)], requires_grad=True, device=self.device)
            self.sync_alpha2_optimizer = optim.AdamW(
                [self.log_sync_alpha2],
                lr=self.config.cql_lagrange_learning_rate,
                fused=True,
                betas=(0.9, 0.95),
            )

        logger.info(
            "SYNC-QL drift-gated CFCQL enabled: "
            f"delta_threshold={sync.delta_threshold}, selection_mode={sync.selection_mode}, "
            f"drift_mode={sync.drift_mode}, lambda_cf={sync.lambda_cf}"
        )

    def _sync_disabled(self) -> bool:
        sync = self.config.sync_cql
        return sync.selection_mode == "none" or sync.K == 0

    def _sync_alpha2_value(self, dtype: torch.dtype) -> torch.Tensor:
        if self.config.sync_cql.alpha2_lagrange and self.log_sync_alpha2 is not None:
            return self.log_sync_alpha2.exp().detach().to(dtype=dtype)
        return torch.tensor(float(self.config.sync_cql.alpha2), device=self.device, dtype=dtype)

    def _update_sync_alpha2_lagrange(self, sync_penalty: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sync = self.config.sync_cql
        if (
            not sync.alpha2_lagrange
            or self.log_sync_alpha2 is None
            or self.sync_alpha2_optimizer is None
        ):
            return self._sync_alpha2_value(sync_penalty.dtype).detach(), torch.zeros((), device=self.device)

        alpha2 = self.log_sync_alpha2.exp()
        target = torch.tensor(sync.tau_syn, device=self.device, dtype=sync_penalty.dtype)
        alpha2_loss = -alpha2 * (sync_penalty.detach() - target)

        self.sync_alpha2_optimizer.zero_grad(set_to_none=True)
        alpha2_loss.backward()
        if self.is_multi_gpu and self.log_sync_alpha2.grad is not None:
            torch.distributed.all_reduce(self.log_sync_alpha2.grad.data, op=torch.distributed.ReduceOp.SUM)
            self.log_sync_alpha2.grad.data.copy_(self.log_sync_alpha2.grad.data / self.gpu_world_size)
        self.sync_alpha2_optimizer.step()
        return self.log_sync_alpha2.exp().detach(), alpha2_loss.detach()

    @torch.no_grad()
    def _update_sync_action_std(self, dataset_actions: torch.Tensor) -> None:
        sync = self.config.sync_cql
        if sync.freeze_drift_stats:
            return
        batch_std = dataset_actions.detach().float().std(dim=0, unbiased=False).clamp_min(1e-3)
        self.sync_action_std.mul_(sync.drift_std_momentum).add_(batch_std, alpha=1.0 - sync.drift_std_momentum)

    @torch.no_grad()
    def _compute_group_drift(
        self,
        dataset_actions: torch.Tensor,
        actor_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute d_g(s,a_D) for every sample and action group."""

        sync = self.config.sync_cql
        if sync.drift_mode == "density":
            raise NotImplementedError(
                "sync_cql.drift_mode='density' requires a per-group behavior model "
                "mu_g(a_g|s). Plug in the CVAE density estimator before enabling it."
            )
        if sync.drift_mode != "rmse":
            raise ValueError(f"Unknown sync_cql.drift_mode={sync.drift_mode!r}")

        normalized_delta = (actor_actions.detach() - dataset_actions.detach()) / (
            self.sync_action_std.to(dataset_actions.device, dataset_actions.dtype) + 1e-6
        )
        drift_values = []
        for group_indices in self.bf_cql_group_indices:
            group_delta = normalized_delta[:, list(group_indices)]
            drift_values.append(torch.sqrt(group_delta.pow(2).mean(dim=1) + 1e-12))
        drift = torch.stack(drift_values, dim=1)

        if sync.drift_ema > 0.0:
            batch_mean = drift.mean(dim=0)
            self.sync_group_drift_ema.mul_(sync.drift_ema).add_(batch_mean, alpha=1.0 - sync.drift_ema)
            drift = (1.0 - sync.drift_ema) * drift + sync.drift_ema * self.sync_group_drift_ema[None, :]
        return drift

    @torch.no_grad()
    def _screen_sync_candidates(self, group_drift: torch.Tensor) -> torch.Tensor:
        sync = self.config.sync_cql
        batch_size, num_groups = group_drift.shape
        screen_k = min(max(2 * int(sync.K), 1), num_groups)
        candidate_mask = torch.zeros(batch_size, num_groups, device=group_drift.device, dtype=torch.bool)
        top_indices = group_drift.topk(screen_k, dim=1).indices
        candidate_mask.scatter_(1, top_indices, True)
        candidate_mask &= group_drift >= float(sync.delta_threshold)
        return candidate_mask

    @torch.no_grad()
    def _select_sync_groups(
        self,
        observations: torch.Tensor,
        critic_observations: torch.Tensor,
        dataset_actions: torch.Tensor,
        actor_actions: torch.Tensor,
        q_data_min: torch.Tensor,
        group_drift: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select M* per sample with detached tensors; no gradient flows through selection."""

        sync = self.config.sync_cql
        batch_size, num_groups = group_drift.shape
        max_k = min(int(sync.K), num_groups)
        selected_indices = torch.full((batch_size, max_k), -1, device=self.device, dtype=torch.long)
        selected_mask = torch.zeros(batch_size, num_groups, device=self.device, dtype=torch.bool)

        if max_k == 0 or sync.selection_mode == "none":
            return selected_mask.detach(), selected_indices.detach(), torch.zeros(batch_size, device=self.device, dtype=torch.bool)

        if sync.selection_mode == "random":
            scores = torch.rand(batch_size, num_groups, device=self.device)
            selected_indices = scores.topk(max_k, dim=1).indices
            selected_mask.scatter_(1, selected_indices, True)
            return selected_mask.detach(), selected_indices.detach(), torch.ones(batch_size, device=self.device, dtype=torch.bool)

        candidate_mask = self._screen_sync_candidates(group_drift)
        if sync.selection_mode == "topk":
            masked_drift = group_drift.masked_fill(~candidate_mask, -torch.inf)
            selected_values, selected_indices = masked_drift.topk(max_k, dim=1)
            valid_selected = torch.isfinite(selected_values)
            safe_indices = selected_indices.clamp_min(0)
            selected_mask.scatter_(1, safe_indices, valid_selected)
            selected_indices = torch.where(valid_selected, selected_indices, torch.full_like(selected_indices, -1))
            return selected_mask.detach(), selected_indices.detach(), selected_mask.any(dim=1).detach()

        if sync.selection_mode != "greedy":
            raise ValueError(f"Unknown sync_cql.selection_mode={sync.selection_mode!r}")

        screen_k = min(max(2 * max_k, 1), num_groups)
        candidate_scores = group_drift.masked_fill(~candidate_mask, -torch.inf)
        _, candidate_indices = candidate_scores.topk(screen_k, dim=1)
        candidate_valid = torch.isfinite(candidate_scores.gather(1, candidate_indices))
        current_v = torch.zeros(batch_size, device=self.device, dtype=q_data_min.dtype)

        for stage in range(max_k):
            available_valid = candidate_valid & ~selected_mask.gather(1, candidate_indices)
            trial_masks = selected_mask[:, None, :].expand(batch_size, screen_k, num_groups).clone()
            trial_masks.scatter_(2, candidate_indices[:, :, None], True)
            trial_actions = counterfactual_actions_from_group_masks(
                dataset_actions,
                actor_actions,
                trial_masks,
                self.sync_group_to_action_mask,
            )
            trial_critic_obs = critic_observations[:, None, :].expand(batch_size, screen_k, -1).reshape(
                batch_size * screen_k,
                -1,
            )
            q1_trial, q2_trial = self.qnet(trial_critic_obs, trial_actions.reshape(batch_size * screen_k, -1))
            trial_v = torch.minimum(q1_trial, q2_trial).view(batch_size, screen_k) - q_data_min[:, None]
            marginal = (trial_v - current_v[:, None]).masked_fill(~available_valid, -torch.inf)
            best_gain, best_pos = marginal.max(dim=1)
            add_mask = torch.isfinite(best_gain) & (best_gain >= float(sync.eps_gain))
            chosen_group = candidate_indices.gather(1, best_pos[:, None]).squeeze(1)
            safe_group = chosen_group.clamp_min(0)
            selected_mask[add_mask, safe_group[add_mask]] = True
            selected_indices[:, stage] = torch.where(add_mask, chosen_group, torch.full_like(chosen_group, -1))
            current_v = torch.where(add_mask, trial_v.gather(1, best_pos[:, None]).squeeze(1), current_v)

        return selected_mask.detach(), selected_indices.detach(), selected_mask.any(dim=1).detach()

    def _compute_sync_penalty(
        self,
        observations: torch.Tensor,
        critic_observations: torch.Tensor,
        dataset_actions: torch.Tensor,
        q1_data: torch.Tensor,
        q2_data: torch.Tensor,
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
    ]:
        """Compute alpha2 * E[relu(Delta(M*) - margin_m)].

        The hinge is on Delta(M*) rather than v(M*) because singleton CFCQL
        already suppresses one-group drift; this term should only target
        inter-group interaction residuals.
        """

        sync = self.config.sync_cql
        batch_size = dataset_actions.shape[0]
        num_groups = len(self.bf_cql_group_indices)
        max_k = min(max(int(sync.K), 0), num_groups)
        zero = torch.zeros((), device=self.device, dtype=q1_data.dtype)
        empty_selected = torch.zeros(batch_size, num_groups, device=self.device, dtype=torch.bool)
        empty_vec = torch.zeros(num_groups, device=self.device, dtype=q1_data.dtype)

        if self._sync_disabled():
            return (
                zero,
                zero,
                zero,
                zero,
                zero,
                self._sync_alpha2_value(q1_data.dtype),
                zero,
                empty_vec,
                empty_vec,
                empty_selected,
                zero,
            )

        with torch.no_grad():
            actor_actions = self.actor(observations)[0]
            self._update_sync_action_std(dataset_actions)
            group_drift = self._compute_group_drift(dataset_actions, actor_actions)
            q_data_min = torch.minimum(q1_data.detach(), q2_data.detach())
            selected_mask, selected_indices, active_mask = self._select_sync_groups(
                observations.detach(),
                critic_observations.detach(),
                dataset_actions.detach(),
                actor_actions.detach(),
                q_data_min,
                group_drift,
            )

        active_float = active_mask.to(dtype=q1_data.dtype)
        valid_count = active_float.sum().clamp_min(1.0)
        drift_means = group_drift.mean(dim=0).detach()
        selection_freq = selected_mask.to(dtype=q1_data.dtype).mean(dim=0).detach()

        block_actions = counterfactual_actions_from_group_masks(
            dataset_actions.detach(),
            actor_actions.detach(),
            selected_mask[:, None, :],
            self.sync_group_to_action_mask,
        ).squeeze(1)
        q1_block, q2_block = self.qnet(critic_observations, block_actions)

        if max_k > 0:
            singleton_masks = torch.zeros(
                batch_size,
                max_k,
                num_groups,
                device=self.device,
                dtype=torch.bool,
            )
            singleton_valid = selected_indices >= 0
            safe_indices = selected_indices.clamp_min(0)
            singleton_masks.scatter_(2, safe_indices[:, :, None], singleton_valid[:, :, None])
            singleton_actions = counterfactual_actions_from_group_masks(
                dataset_actions.detach(),
                actor_actions.detach(),
                singleton_masks,
                self.sync_group_to_action_mask,
            )
            singleton_critic_obs = critic_observations[:, None, :].expand(batch_size, max_k, -1).reshape(
                batch_size * max_k,
                -1,
            )
            q1_single, q2_single = self.qnet(singleton_critic_obs, singleton_actions.reshape(batch_size * max_k, -1))
            q1_single = q1_single.view(batch_size, max_k)
            q2_single = q2_single.view(batch_size, max_k)
        else:
            singleton_valid = torch.zeros(batch_size, 0, device=self.device, dtype=torch.bool)
            q1_single = torch.zeros(batch_size, 0, device=self.device, dtype=q1_data.dtype)
            q2_single = torch.zeros_like(q1_single)

        q1_data_anchor = q1_data.detach()
        q2_data_anchor = q2_data.detach()
        delta1, v_block1, v_single_sum1 = synergy_residual(q1_data_anchor, q1_block, q1_single, singleton_valid)
        delta2, v_block2, v_single_sum2 = synergy_residual(q2_data_anchor, q2_block, q2_single, singleton_valid)
        penalty1 = F.relu(delta1 - float(sync.margin_m)) * active_float
        penalty2 = F.relu(delta2 - float(sync.margin_m)) * active_float
        penalty_mean = 0.5 * ((penalty1.sum() / valid_count) + (penalty2.sum() / valid_count))
        alpha2 = self._sync_alpha2_value(q1_data.dtype)
        sync_loss = alpha2 * penalty_mean

        delta_mean = 0.5 * (((delta1 * active_float).sum() / valid_count) + ((delta2 * active_float).sum() / valid_count))
        v_block_mean = 0.5 * (
            ((v_block1 * active_float).sum() / valid_count) + ((v_block2 * active_float).sum() / valid_count)
        )
        v_single_sum_mean = 0.5 * (
            ((v_single_sum1 * active_float).sum() / valid_count)
            + ((v_single_sum2 * active_float).sum() / valid_count)
        )
        active_frac = active_float.mean()
        subset_hash = self._selected_subset_hash(selected_mask).to(dtype=q1_data.dtype)
        subset_hash_mean = (subset_hash * active_float).sum() / valid_count

        return (
            sync_loss,
            penalty_mean.detach(),
            delta_mean.detach(),
            v_block_mean.detach(),
            v_single_sum_mean.detach(),
            alpha2.detach().mean(),
            active_frac.detach(),
            drift_means,
            selection_freq,
            selected_mask.detach(),
            subset_hash_mean.detach(),
        )

    def _selected_subset_hash(self, selected_mask: torch.Tensor) -> torch.Tensor:
        powers = (2 ** torch.arange(selected_mask.shape[1], device=selected_mask.device)).to(torch.long)
        return (selected_mask.to(torch.long) * powers[None, :]).sum(dim=1)

    @torch.no_grad()
    def _selected_subset_composition_summary(self, selected_mask: torch.Tensor, max_items: int = 8) -> str:
        if selected_mask.numel() == 0:
            return "none"
        subset_hash = self._selected_subset_hash(selected_mask).detach().cpu()
        unique_hashes, counts = torch.unique(subset_hash, return_counts=True)
        order = counts.argsort(descending=True)
        total = max(int(subset_hash.numel()), 1)
        items: list[str] = []
        for idx in order[:max_items].tolist():
            hash_value = int(unique_hashes[idx].item())
            count = int(counts[idx].item())
            if hash_value == 0:
                name = "empty"
            else:
                selected_names = [
                    group_name
                    for group_idx, group_name in enumerate(self.bf_cql_group_names)
                    if hash_value & (1 << group_idx)
                ]
                name = "+".join(selected_names)
            items.append(f"{name}:{count / total:.3f}")
        return ", ".join(items)

    def _update_q(
        self,
        data: TensorDict,
    ):
        args = self.config
        scaler = self.scaler
        reward_scale = self.reward_scale
        with self._maybe_amp():
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
            dataset_actions = self._to_normalized_actions(data["actions"])
            rewards = reward_scale * data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            # Truncated ends (timeout / d3 segment_ends) are not true terminals: the
            # exporter stores the final pre-reset observation in next.observations, so
            # bootstrap through them. Required for cross-segment value stitching.
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
                    next_q1_target, next_q2_target = self.qnet_target(expanded_next_critic_obs, next_actions)
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
                    next_q1_target, next_q2_target = self.qnet_target(next_critic_observations, next_actions)
                    next_target_min_q = torch.minimum(next_q1_target, next_q2_target).view(-1)
                    next_log_probs = next_log_probs.view(-1)

                next_v = next_target_min_q - alpha * next_log_probs if args.backup_entropy else next_target_min_q
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
            sync_loss = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            sync_penalty = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            sync_delta = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            sync_v_block = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            sync_v_singleton_sum = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            sync_alpha2 = self._sync_alpha2_value(bellman_loss.dtype).detach().mean()
            sync_active_frac = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            sync_subset_hash_mean = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            num_groups_int = len(self.bf_cql_group_indices)
            sync_drift_means = torch.zeros(num_groups_int, device=self.device, dtype=bellman_loss.dtype)
            sync_selection_freq = torch.zeros(num_groups_int, device=self.device, dtype=bellman_loss.dtype)
            sync_selected_mask = torch.zeros(
                dataset_actions.shape[0],
                num_groups_int,
                device=self.device,
                dtype=torch.bool,
            )
            with torch.no_grad():
                pi_actions_det = self.actor(observations)[0]
                q1_pi_det, q2_pi_det = self.qnet(critic_observations, pi_actions_det)
                q_pi_minus_q_data = (
                    torch.minimum(q1_pi_det, q2_pi_det) - torch.minimum(q1.detach(), q2.detach())
                ).mean()
                if self.config.sync_cql.selection_mode != "none":
                    self._update_sync_action_std(dataset_actions)
                    group_drift = self._compute_group_drift(dataset_actions, pi_actions_det)
                    sync_selected_mask = group_drift >= float(self.config.sync_cql.delta_threshold)
                    sync_drift_means = group_drift.mean(dim=0).detach()
                    sync_selection_freq = sync_selected_mask.to(dtype=bellman_loss.dtype).mean(dim=0).detach()
                    sync_active_frac = sync_selected_mask.any(dim=1).to(dtype=bellman_loss.dtype).mean().detach()
                    sync_subset_hash = self._selected_subset_hash(sync_selected_mask).to(dtype=bellman_loss.dtype)
                    sync_subset_hash_mean = sync_subset_hash.mean().detach()
                else:
                    sync_selected_mask = torch.ones(
                        dataset_actions.shape[0],
                        num_groups_int,
                        device=self.device,
                        dtype=torch.bool,
                    )
                    sync_selection_freq = torch.ones(num_groups_int, device=self.device, dtype=bellman_loss.dtype)
                    sync_active_frac = torch.ones((), device=self.device, dtype=bellman_loss.dtype)

            if self._cql_weight > 0.0:
                batch_size = dataset_actions.shape[0]
                num_repeat = self._num_repeat_actions
                expanded_obs = observations[:, None, :].expand(batch_size, num_repeat, -1).reshape(
                    batch_size * num_repeat,
                    -1,
                )
                expanded_critic_obs = critic_observations[:, None, :].expand(batch_size, num_repeat, -1).reshape(
                    batch_size * num_repeat,
                    -1,
                )
                expanded_next_obs = next_observations[:, None, :].expand(batch_size, num_repeat, -1).reshape(
                    batch_size * num_repeat,
                    -1,
                )
                expanded_dataset_actions = dataset_actions[:, None, :].expand(batch_size, num_repeat, -1).reshape(
                    batch_size * num_repeat,
                    -1,
                )

                with torch.no_grad():
                    curr_actions, _, curr_group_logps = self.actor.get_actions_and_group_log_probs(expanded_obs)
                    next_actions_rep, _, next_group_logps = self.actor.get_actions_and_group_log_probs(
                        expanded_next_obs
                    )
                    curr_group_logps_stacked = torch.stack(curr_group_logps, dim=1)
                    next_group_logps_stacked = torch.stack(next_group_logps, dim=1)

                cql1_loss_total = torch.zeros(batch_size, device=self.device, dtype=bellman_loss.dtype)
                cql2_loss_total = torch.zeros(batch_size, device=self.device, dtype=bellman_loss.dtype)
                rand_q_total = torch.zeros(batch_size, device=self.device, dtype=bellman_loss.dtype)
                curr_q_total = torch.zeros(batch_size, device=self.device, dtype=bellman_loss.dtype)
                next_q_total = torch.zeros(batch_size, device=self.device, dtype=bellman_loss.dtype)
                curr_logp_total = torch.zeros(batch_size, device=self.device, dtype=bellman_loss.dtype)
                next_logp_total = torch.zeros(batch_size, device=self.device, dtype=bellman_loss.dtype)
                random_density_total = torch.zeros(batch_size, device=self.device, dtype=bellman_loss.dtype)
                active_group_count = sync_selected_mask.to(dtype=bellman_loss.dtype).sum(dim=1)
                active_group_denominator = active_group_count.clamp_min(1.0)
                active_sample_mask = active_group_count > 0
                active_sample_float = active_sample_mask.to(dtype=bellman_loss.dtype)
                active_sample_denominator = active_sample_float.sum().clamp_min(1.0)
                for group_idx, group_indices in enumerate(self.bf_cql_group_indices):
                    group_active = sync_selected_mask[:, group_idx].to(dtype=bellman_loss.dtype)
                    group_dim = len(group_indices)
                    rand_group_actions = torch.empty(
                        batch_size * num_repeat,
                        group_dim,
                        device=self.device,
                        dtype=dataset_actions.dtype,
                    ).uniform_(-1.0, 1.0)
                    random_density = math.log(0.5) * group_dim

                    rand_counterfactual_actions = self._counterfactual_group_actions(
                        expanded_dataset_actions,
                        group_indices,
                        rand_group_actions,
                    )
                    selected_group_mask = self._sample_actor_ood_group_mask(
                        num_rows=batch_size * num_repeat,
                        base_group_idx=group_idx,
                        num_groups=num_groups_int,
                        device=self.device,
                    )
                    curr_counterfactual_actions = self._counterfactual_actor_group_actions(
                        expanded_dataset_actions,
                        curr_actions,
                        selected_group_mask,
                    )
                    next_counterfactual_actions = self._counterfactual_actor_group_actions(
                        expanded_dataset_actions,
                        next_actions_rep,
                        selected_group_mask,
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
                    )
                    cql2_group_loss = (
                        torch.logsumexp(q2_terms / self._temperature, dim=1) * self._temperature - q2
                    )
                    cql1_loss_total = cql1_loss_total + cql1_group_loss * group_active
                    cql2_loss_total = cql2_loss_total + cql2_group_loss * group_active
                    rand_q_group = 0.5 * (
                        (q1_rand - random_density).mean(dim=1) + (q2_rand - random_density).mean(dim=1)
                    )
                    curr_q_group = 0.5 * (q1_curr.mean(dim=1) + q2_curr.mean(dim=1))
                    next_q_group = 0.5 * (q1_next.mean(dim=1) + q2_next.mean(dim=1))
                    rand_q_total = rand_q_total + rand_q_group * group_active
                    curr_q_total = curr_q_total + curr_q_group * group_active
                    next_q_total = next_q_total + next_q_group * group_active
                    curr_logp_total = curr_logp_total + curr_actor_logp.mean(dim=1) * group_active
                    next_logp_total = next_logp_total + next_actor_logp.mean(dim=1) * group_active
                    random_density_total = random_density_total + torch.full(
                        (batch_size,),
                        random_density,
                        device=self.device,
                        dtype=bellman_loss.dtype,
                    ) * group_active

                cql1_per_sample = cql1_loss_total / active_group_denominator
                cql2_per_sample = cql2_loss_total / active_group_denominator
                cql1_loss = (cql1_per_sample * active_sample_float).sum() / active_sample_denominator
                cql2_loss = (cql2_per_sample * active_sample_float).sum() / active_sample_denominator
                cql_gap = 0.5 * (cql1_loss + cql2_loss)
                rand_q_mean = (
                    (rand_q_total / active_group_denominator) * active_sample_float
                ).sum() / active_sample_denominator
                curr_q_mean = (
                    (curr_q_total / active_group_denominator) * active_sample_float
                ).sum() / active_sample_denominator
                next_q_mean = (
                    (next_q_total / active_group_denominator) * active_sample_float
                ).sum() / active_sample_denominator
                curr_logp_mean = (
                    (curr_logp_total / active_group_denominator) * active_sample_float
                ).sum() / active_sample_denominator
                next_logp_mean = (
                    (next_logp_total / active_group_denominator) * active_sample_float
                ).sum() / active_sample_denominator
                random_density_mean = (
                    (random_density_total / active_group_denominator) * active_sample_float
                ).sum() / active_sample_denominator

                if args.use_lagrange and self.log_cql_alpha is not None:
                    cql_alpha = self.log_cql_alpha.exp().detach().clamp(max=args.cql_lagrange_max)
                    target_gap = torch.tensor(args.cql_target_action_gap, device=self.device, dtype=bellman_loss.dtype)
                    has_active_sample = (active_sample_float.sum() > 0).to(dtype=bellman_loss.dtype)
                    conservative_loss = cql_alpha * self._cql_weight * 0.5 * (
                        (cql1_loss - target_gap) + (cql2_loss - target_gap)
                    ) * has_active_sample
                else:
                    conservative_loss = self._cql_weight * 0.5 * (cql1_loss + cql2_loss)
            else:
                conservative_loss = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                cql_gap = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)

            sync_loss = conservative_loss.detach()
            sync_penalty = cql_gap.detach()
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
            sync_loss.detach(),
            sync_penalty.detach(),
            sync_delta.detach(),
            sync_v_block.detach(),
            sync_v_singleton_sum.detach(),
            sync_alpha2.detach(),
            sync_active_frac.detach(),
            sync_drift_means.detach(),
            sync_selection_freq.detach(),
            sync_selected_mask.detach(),
            sync_subset_hash_mean.detach(),
        )

    def _update_actor(
        self,
        data: TensorDict,
        selected_group_mask: torch.Tensor | None = None,
    ):
        args = self.config
        scaler = self.scaler
        sync = args.sync_cql

        with self._maybe_amp():
            actor_observations = data["observations"]
            critic_observations = data["critic_observations"]

            _, _, log_std = self.actor(actor_observations)
            actions, log_probs = self.actor.get_actions_and_log_probs(actor_observations)
            with torch.no_grad():
                action_std = log_std.exp().mean()
                policy_entropy = -log_probs.mean()

            q1_pi, q2_pi = self.qnet(critic_observations, actions)
            qf_value = torch.minimum(q1_pi, q2_pi)
            actor_rl_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

            actor_cf_loss = torch.zeros((), device=self.device, dtype=actor_rl_loss.dtype)
            if (
                sync.lambda_cf > 0.0
                and selected_group_mask is not None
                and selected_group_mask.numel() > 0
            ):
                dataset_actions = self._to_normalized_actions(data["actions"]).detach()
                action_masks = torch.matmul(
                    selected_group_mask.float(),
                    self.sync_group_to_action_mask.to(device=selected_group_mask.device, dtype=torch.float32),
                ).to(dtype=torch.bool)
                cf_actions = torch.where(action_masks, actions, dataset_actions)
                q1_cf, q2_cf = self.qnet(critic_observations, cf_actions)
                active = selected_group_mask.any(dim=1)
                active_float = active.to(dtype=q1_cf.dtype)
                actor_cf_loss = -(torch.minimum(q1_cf, q2_cf) * active_float).sum() / active_float.sum().clamp_min(1.0)
                actor_loss = (1.0 - sync.lambda_cf) * actor_rl_loss + sync.lambda_cf * actor_cf_loss
            else:
                actor_loss = actor_rl_loss

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
            actor_cf_loss.detach(),
        )

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
                "Offline SYNC-QL gradient updates sample only the fixed dataset. "
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
        last_selected_mask: torch.Tensor | None = None
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
                        sync_loss,
                        sync_penalty,
                        sync_delta,
                        sync_v_block,
                        sync_v_singleton_sum,
                        sync_alpha2,
                        sync_active_frac,
                        sync_drift_means,
                        sync_selection_freq,
                        sync_selected_mask,
                        sync_subset_hash_mean,
                    ) = update_q(data)
                    last_selected_mask = sync_selected_mask

                    cql_alpha_value, cql_lagrange_loss = self._update_cql_lagrange(cql_gap)
                    sync_alpha2_value = sync_alpha2
                    sync_alpha2_lagrange_loss = torch.zeros((), device=self.device)

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
                            actor_cf_loss,
                        ) = update_actor(data, sync_selected_mask)
                    else:
                        actor_grad_norm = torch.tensor(0.0, device=self.device)
                        actor_loss = torch.tensor(0.0, device=self.device)
                        policy_entropy = torch.tensor(0.0, device=self.device)
                        action_std = torch.tensor(0.0, device=self.device)
                        actor_cf_loss = torch.tensor(0.0, device=self.device)

                    self._soft_update_q_target()

                    action_ood_stats = self._compute_action_ood_stats(data)
                    sync_metric_dict = {
                        "syncql/loss": sync_loss,
                        "syncql/penalty": sync_penalty,
                        "syncql/delta": sync_delta,
                        "syncql/v_block": sync_v_block,
                        "syncql/v_singleton_sum": sync_v_singleton_sum,
                        "syncql/alpha2": sync_alpha2_value,
                        "syncql/alpha2_lagrange_loss": sync_alpha2_lagrange_loss,
                        "syncql/active_frac": sync_active_frac,
                        "syncql/active_group_frac": sync_selection_freq.mean(),
                        "syncql/actor_cf_loss": actor_cf_loss,
                        "syncql/selected_subset_hash_mean": sync_subset_hash_mean,
                        "cfcql/penalty": conservative_loss,
                    }
                    for group_idx, group_name in enumerate(self.bf_cql_group_names):
                        sync_metric_dict[f"drift/d_g_{group_name}"] = sync_drift_means[group_idx]
                        sync_metric_dict[f"select/freq_{group_name}"] = sync_selection_freq[group_idx]

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
                            "bf_cql/ood_actor_num": torch.tensor(float(args.ood_actor_num), device=self.device),
                            "is_actor_warmup": float(is_actor_warmup),
                            "is_actor_update_step": float(is_actor_update_step),
                            **action_ood_stats,
                            **sync_metric_dict,
                            "current_logprob": curr_logp,
                            "next_logprob": next_logp,
                            "random_density": random_density,
                        }
                    )

            should_log = (self.global_step % args.logging_interval == 0) or (self.global_step <= 10)
            if should_log:
                if self.is_main_process and last_selected_mask is not None:
                    logger.info(
                        "[SYNC-QL subsets] "
                        + self._selected_subset_composition_summary(last_selected_mask)
                    )
                with torch.no_grad():
                    accumulated_metrics = self.training_metrics.mean_and_clear()
                    loss_dict = {
                        key: (value.item() if isinstance(value, torch.Tensor) else float(value))
                        for key, value in accumulated_metrics.items()
                    }
                self.logging_helper.post_epoch_logging(it=self.global_step, loss_dict=loss_dict, extra_log_dicts={})

            if args.save_interval > 0 and self.global_step % args.save_interval == 0:
                if self.is_main_process:
                    logger.info(f"Saving SYNC-QL model at global step {self.global_step}")
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
        metadata["action_space_mode"] = "normalized_action_training_v1"
        metadata["algo"] = "sync_cql"
        metadata["bf_cql_action_grouping"] = self.config.bf_cql_action_grouping
        metadata["sync_cql"] = {
            "K": self.config.sync_cql.K,
            "delta_threshold": self.config.sync_cql.delta_threshold,
            "selection_mode": self.config.sync_cql.selection_mode,
            "drift_mode": self.config.sync_cql.drift_mode,
            "eps_gain": self.config.sync_cql.eps_gain,
            "margin_m": self.config.sync_cql.margin_m,
            "alpha2": self.config.sync_cql.alpha2,
            "alpha2_lagrange": self.config.sync_cql.alpha2_lagrange,
            "tau_syn": self.config.sync_cql.tau_syn,
            "lambda_cf": self.config.sync_cql.lambda_cf,
            "drift_ema": self.config.sync_cql.drift_ema,
            "drift_std_momentum": self.config.sync_cql.drift_std_momentum,
            "freeze_drift_stats": self.config.sync_cql.freeze_drift_stats,
        }
        if self.log_sync_alpha2 is not None:
            metadata["sync_log_alpha2"] = self.log_sync_alpha2.detach().cpu()
        if self.sync_alpha2_optimizer is not None:
            metadata["sync_alpha2_optimizer_state_dict"] = self.sync_alpha2_optimizer.state_dict()
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

    def load(self, ckpt_path: str | None) -> None:
        BFCQLAgent.load(self, ckpt_path)
        if not ckpt_path:
            return
        if not self.config.sync_cql.alpha2_lagrange or self.log_sync_alpha2 is None:
            return
        torch_checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "sync_log_alpha2" in torch_checkpoint:
            self.log_sync_alpha2.data.copy_(torch_checkpoint["sync_log_alpha2"].to(self.device))
        if self.sync_alpha2_optimizer is not None and "sync_alpha2_optimizer_state_dict" in torch_checkpoint:
            self.sync_alpha2_optimizer.load_state_dict(torch_checkpoint["sync_alpha2_optimizer_state_dict"])
