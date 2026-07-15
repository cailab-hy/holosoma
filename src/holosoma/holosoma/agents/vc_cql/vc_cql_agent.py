"""Value-Contour Conservative Q-Learning.

VC-CQL preserves scalar CQL's dataset, Bellman target, conservative critic,
target-network, and evaluation implementation.  It replaces the diagonal actor
with a low-rank-plus-diagonal Gaussian.  At each actor update, antithetic local
actions are evaluated by the target twin critic, Q-drop weights define a local
near-optimal contour covariance, and the actor covariance is aligned to that
stop-gradient target.  The deterministic actor mean is optimized only through
the global critic, while covariance learns state-dependent coordinated joint
directions without predefined body groups or a fixed target entropy.
"""

from __future__ import annotations

from loguru import logger

from holosoma.agents.cql.cql_agent import CQLAgent
from holosoma.agents.vc_cql.vc_cql import (
    VCActor,
    covariance_kl,
    linearized_squashed_covariance,
    low_rank_covariance,
    weighted_contour_covariance,
)
from holosoma.config_types.algo import VCCQLConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.safe_torch_import import TensorDict, optim, torch


class VCCQLAgent(CQLAgent):
    """Scalar CQL with state-wise value-contour covariance adaptation."""

    config: VCCQLConfig
    actor: VCActor

    _VC_METRIC_KEYS = (
        "vc_cql/mean_q",
        "vc_cql/mean_loss",
        "vc_cql/contour_kl",
        "vc_cql/contour_loss",
        "vc_cql/q_drop_mean",
        "vc_cql/q_drop_p50",
        "vc_cql/q_drop_p90",
        "vc_cql/contour_mass",
        "vc_cql/weight_mass",
        "vc_cql/contour_cov_trace",
        "vc_cql/policy_cov_trace",
        "vc_cql/cov_trace_ratio",
        "vc_cql/contour_effective_rank",
        "vc_cql/policy_effective_rank",
        "vc_cql/contour_entropy",
        "vc_cql/contour_condition",
    )

    def __init__(
        self,
        env: BaseTask,
        config: VCCQLConfig,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict | None = None,
    ):
        self._validate_vc_config(config)
        self._vc_last_metrics: dict[str, torch.Tensor] = {}
        super().__init__(env, config, device, log_dir, multi_gpu_cfg)

    @staticmethod
    def _validate_vc_config(config: VCCQLConfig) -> None:
        if config.use_cnn_encoder:
            raise ValueError("VC-CQL currently supports the MLP actor only; use_cnn_encoder must be False.")
        if config.use_autotune:
            raise ValueError("VC-CQL replaces fixed target-entropy autotuning; set use_autotune=False.")
        if config.backup_entropy:
            raise ValueError("VC-CQL requires backup_entropy=False to avoid global entropy pressure.")
        if config.vc_rank <= 0:
            raise ValueError(f"vc_rank must be > 0, got {config.vc_rank}")
        if config.vc_num_probes < 2:
            raise ValueError(f"vc_num_probes must be >= 2, got {config.vc_num_probes}")
        if config.vc_probe_std <= 0.0:
            raise ValueError(f"vc_probe_std must be > 0, got {config.vc_probe_std}")
        if config.vc_q_temperature <= 0.0:
            raise ValueError(f"vc_q_temperature must be > 0, got {config.vc_q_temperature}")
        if config.vc_q_tolerance < 0.0:
            raise ValueError(f"vc_q_tolerance must be >= 0, got {config.vc_q_tolerance}")
        if config.vc_contour_weight < 0.0:
            raise ValueError(f"vc_contour_weight must be >= 0, got {config.vc_contour_weight}")
        if config.vc_cov_epsilon <= 0.0:
            raise ValueError(f"vc_cov_epsilon must be > 0, got {config.vc_cov_epsilon}")
        if not 0.0 <= config.vc_cov_shrinkage <= 1.0:
            raise ValueError(f"vc_cov_shrinkage must be in [0, 1], got {config.vc_cov_shrinkage}")
        if config.vc_factor_max <= 0.0:
            raise ValueError(f"vc_factor_max must be > 0, got {config.vc_factor_max}")

    def setup(self) -> None:
        super().setup()
        args = self.config
        n_act = self.env.robot_config.actions_dim
        if args.vc_rank > n_act:
            raise ValueError(f"vc_rank ({args.vc_rank}) cannot exceed action dimension ({n_act}).")

        self.actor = VCActor(
            obs_indices=self.actor_obs_indices,
            obs_keys=list(args.actor_obs_keys),
            n_act=n_act,
            num_envs=self.env.num_envs,
            hidden_dim=args.actor_hidden_dim,
            log_std_max=args.log_std_max,
            log_std_min=args.log_std_min,
            use_tanh=args.use_tanh,
            use_layer_norm=args.use_layer_norm,
            device=self.device,
            action_scale=self.env_action_scale,
            action_bias=self.env_action_bias,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
            covariance_rank=args.vc_rank,
            factor_max=args.vc_factor_max,
        )
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=args.actor_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )

        def _env_policy(
            obs: torch.Tensor,
            dones: torch.Tensor | None = None,
            deterministic: bool = False,
        ) -> torch.Tensor:
            return self._to_env_actions(self.actor.explore(obs, dones=dones, deterministic=deterministic))

        self.policy = _env_policy
        self.action_space_mode = "vc_cql_env_scaled_low_rank_v1"
        self._vc_last_metrics = self._zero_vc_metrics()
        if self.is_multi_gpu:
            self._synchronize_model_parameters()
        logger.info(
            "VC-CQL actor configured: "
            f"rank={args.vc_rank}, probes={args.vc_num_probes}, probe_std={args.vc_probe_std}, "
            f"q_temperature={args.vc_q_temperature}, contour_weight={args.vc_contour_weight}"
        )

    def _zero_vc_metrics(self) -> dict[str, torch.Tensor]:
        return {key: torch.zeros((), device=self.device) for key in self._VC_METRIC_KEYS}

    @torch.no_grad()
    def _estimate_value_contour(
        self,
        critic_observations: torch.Tensor,
        mean_raw_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Estimate a stop-gradient local Q-contour covariance in physical action coordinates."""

        args = self.config
        batch_size, action_dim = mean_raw_action.shape
        half_probes = (args.vc_num_probes + 1) // 2
        base_noise = torch.randn(
            batch_size,
            half_probes,
            action_dim,
            device=mean_raw_action.device,
            dtype=mean_raw_action.dtype,
        )
        probe_noise = torch.cat((base_noise, -base_noise), dim=1)[:, : args.vc_num_probes]
        raw_deltas = args.vc_probe_std * probe_noise
        raw_probe_actions = mean_raw_action[:, None, :] + raw_deltas
        probe_actions = self.actor.squash_raw_action(raw_probe_actions)
        mean_actions = self.actor.squash_raw_action(mean_raw_action)

        expanded_critic_obs = critic_observations[:, None, :].expand(
            batch_size,
            args.vc_num_probes,
            critic_observations.shape[-1],
        )
        q1_mean, q2_mean = self.qnet_target(critic_observations, mean_actions)
        q1_probe, q2_probe = self.qnet_target(
            expanded_critic_obs.reshape(batch_size * args.vc_num_probes, -1),
            probe_actions.reshape(batch_size * args.vc_num_probes, action_dim),
        )
        q_mean = torch.minimum(q1_mean, q2_mean).view(batch_size, 1)
        q_probe = torch.minimum(q1_probe, q2_probe).view(batch_size, args.vc_num_probes)
        q_drops = q_mean - q_probe
        action_deltas = probe_actions - mean_actions[:, None, :]
        contour_covariance, weights = weighted_contour_covariance(
            action_deltas,
            q_drops,
            temperature=args.vc_q_temperature,
            epsilon=args.vc_cov_epsilon,
            shrinkage=args.vc_cov_shrinkage,
        )

        eigenvalues = torch.linalg.eigvalsh(contour_covariance.float()).clamp_min(args.vc_cov_epsilon)
        trace = eigenvalues.sum(dim=-1)
        effective_rank = trace.square() / eigenvalues.square().sum(dim=-1).clamp_min(args.vc_cov_epsilon)
        contour_entropy = 0.5 * (
            action_dim * (1.0 + torch.log(torch.tensor(2.0 * torch.pi, device=self.device)))
            + torch.log(eigenvalues).sum(dim=-1)
        )
        metrics = {
            "vc_cql/q_drop_mean": q_drops.mean(),
            "vc_cql/q_drop_p50": torch.quantile(q_drops.float(), 0.50),
            "vc_cql/q_drop_p90": torch.quantile(q_drops.float(), 0.90),
            "vc_cql/contour_mass": (q_drops <= args.vc_q_tolerance).float().mean(),
            "vc_cql/weight_mass": weights.mean(),
            "vc_cql/contour_cov_trace": trace.mean(),
            "vc_cql/contour_effective_rank": effective_rank.mean(),
            "vc_cql/contour_entropy": contour_entropy.mean(),
            "vc_cql/contour_condition": (eigenvalues[:, -1] / eigenvalues[:, 0]).mean(),
        }
        return contour_covariance.detach(), {key: value.detach() for key, value in metrics.items()}

    def _update_actor(
        self,
        data: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        args = self.config
        scaler = self.scaler

        with self._maybe_amp():
            actor_observations = data["observations"]
            critic_observations = data["critic_observations"]
            mean_action, mean_raw, _ = self.actor(actor_observations)
            _, covariance_log_std, covariance_factor = self.actor.distribution_parameters(
                actor_observations,
                detach_features_for_covariance=True,
            )
            raw_policy_covariance = low_rank_covariance(covariance_log_std, covariance_factor)
            policy_covariance = linearized_squashed_covariance(
                raw_policy_covariance,
                mean_raw.detach(),
                self.actor.action_scale,
            )

            with torch.no_grad():
                contour_covariance, contour_metrics = self._estimate_value_contour(
                    critic_observations,
                    mean_raw.detach(),
                )
            contour_kl = covariance_kl(
                policy_covariance,
                contour_covariance,
                epsilon=args.vc_cov_epsilon,
            ).mean()
            contour_loss = args.vc_contour_weight * contour_kl

            q1_mean, q2_mean = self.qnet(critic_observations, mean_action)
            mean_q = torch.minimum(q1_mean, q2_mean)
            mean_loss = -mean_q.mean()
            actor_loss = mean_loss + contour_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        if self.is_multi_gpu:
            self._all_reduce_model_grads(self.actor)
        scaler.unscale_(self.actor_optimizer)
        if args.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), args.max_grad_norm)
        else:
            actor_grad_norm = torch.zeros((), device=self.device)
        scaler.step(self.actor_optimizer)
        scaler.update()

        with torch.no_grad():
            _, log_probs = self.actor.get_actions_and_log_probs(actor_observations)
            policy_entropy = -log_probs.mean()
            marginal_variance = torch.exp(2.0 * covariance_log_std) + covariance_factor.square().sum(dim=-1)
            action_std = marginal_variance.sqrt().mean()
            policy_eigenvalues = torch.linalg.eigvalsh(policy_covariance.float()).clamp_min(args.vc_cov_epsilon)
            policy_trace = policy_eigenvalues.sum(dim=-1)
            policy_effective_rank = policy_trace.square() / policy_eigenvalues.square().sum(dim=-1).clamp_min(
                args.vc_cov_epsilon
            )
            contour_trace = contour_metrics["vc_cql/contour_cov_trace"]
            self._vc_last_metrics = {
                **contour_metrics,
                "vc_cql/mean_q": mean_q.mean().detach(),
                "vc_cql/mean_loss": mean_loss.detach(),
                "vc_cql/contour_kl": contour_kl.detach(),
                "vc_cql/contour_loss": contour_loss.detach(),
                "vc_cql/policy_cov_trace": policy_trace.mean().detach(),
                "vc_cql/cov_trace_ratio": (policy_trace.mean() / contour_trace.clamp_min(args.vc_cov_epsilon)).detach(),
                "vc_cql/policy_effective_rank": policy_effective_rank.mean().detach(),
            }

        return actor_grad_norm.detach(), actor_loss.detach(), policy_entropy.detach(), action_std.detach()

    @torch.no_grad()
    def _compute_action_ood_stats(self, data: TensorDict) -> dict[str, torch.Tensor]:
        stats = super()._compute_action_ood_stats(data)
        stats.update(self._vc_last_metrics)
        self._vc_last_metrics = self._zero_vc_metrics()
        return stats
