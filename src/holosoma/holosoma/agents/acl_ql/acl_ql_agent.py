"""ACL-QL baseline on top of the existing scalar CQL trainer."""

from __future__ import annotations

import copy
import hashlib

import h5py
import numpy as np
from loguru import logger

from holosoma.agents.acl_ql.acl_weight_network import (
    ACLWeightNetwork,
    acl_action_distance,
    acl_quality_raw,
    clamp_acl_quality,
    compute_acl_monotonicity_loss,
    compute_acl_positivity_loss,
    compute_acl_surrogate_loss,
)
from holosoma.agents.cql.cql import Actor
from holosoma.agents.cql.cql_agent import CQLAgent
from holosoma.config_types.algo import ACLQLConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.safe_torch_import import F, TensorDict, optim, torch


class ACLQLAgent(CQLAgent):
    """Paper-faithful ACL-QL baseline with learned w_mu and w_pi_beta."""

    config: ACLQLConfig

    def __init__(self, env: BaseTask, config: ACLQLConfig, device: str, log_dir: str,
                 multi_gpu_cfg: dict | None = None):
        self.behavior_actor: Actor | None = None
        self.behavior_optimizer: optim.Optimizer | None = None
        self.acl_weight_net: ACLWeightNetwork | None = None
        self.acl_weight_optimizer: optim.Optimizer | None = None
        self._acl_quality_table: torch.Tensor | None = None
        self._acl_batch_quality: torch.Tensor | None = None
        self._acl_last_dataset_index: torch.Tensor | None = None
        self._acl_weight_update_context: dict[str, torch.Tensor] = {}
        self._acl_last_metrics: dict[str, torch.Tensor] = {}
        super().__init__(env, config, device, log_dir, multi_gpu_cfg)

    def setup(self) -> None:
        super().setup()
        args = self.config
        # ACL-QL uses cql_weight as the CQL-reference alpha in the learned-weight
        # surrogate (Eq. 21), not as an outer critic-loss multiplier.  Keep the
        # config value intact for the surrogate, but force the inherited CQL
        # critic prefactor to one so the critic term is exactly
        # E[w_mu Q_ood - w_beta Q_data] as in the paper's Eq. (4)/(25).
        acl_reference_alpha = float(args.cql_weight)
        self._cql_weight = 1.0
        if args.use_lagrange and self.log_cql_alpha is not None:
            logger.warning(
                "ACL-QL ignores use_lagrange for the critic conservative term: "
                "Eq. (4)/(25) has no outer lagrange/CQL-alpha prefactor."
            )
            self.log_cql_alpha = None
            self.cql_alpha_optimizer = None
        quality_path = args.acl_quality_path or f"{self._offline_dataset_path}.acl_quality.npz"
        with np.load(quality_path, allow_pickle=False) as sidecar:
            self._verify_quality_sidecar(sidecar, quality_path)
            quality = np.asarray(sidecar["quality"], dtype=np.float32)
            if quality.shape != (self._offline_num_samples,):
                raise ValueError(f"ACL quality sidecar has shape {quality.shape}; expected ({self._offline_num_samples},)")
            self._acl_quality_table = torch.as_tensor(quality, device=self.device)
            self._acl_r_max = float(sidecar["r_max"])

        self.behavior_actor = copy.deepcopy(self.actor).to(self.device)
        self.behavior_optimizer = optim.AdamW(
            self.behavior_actor.parameters(),
            lr=args.acl_behavior_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.acl_weight_net = ACLWeightNetwork(
            self.critic_obs_dim,
            self.env.robot_config.actions_dim,
            hidden_dim=args.acl_weight_hidden_dim,
            num_layers=args.acl_weight_num_layers,
        ).to(self.device)
        self.acl_weight_optimizer = optim.AdamW(
            self.acl_weight_net.parameters(),
            lr=args.acl_weight_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self._install_acl_index_capture()
        self._pretrain_behavior_policy()
        for param in self.behavior_actor.parameters():
            param.requires_grad_(False)
        self.behavior_actor.eval()
        logger.info(
            "ACL-QL setup complete: quality='{}', r_max={:.6f}, "
            "surrogate_reference_alpha={:.6f}, critic_outer_prefactor=1.0",
            quality_path,
            self._acl_r_max,
            acl_reference_alpha,
        )

    def _verify_quality_sidecar(self, sidecar, quality_path: str) -> None:
        required = ("quality", "n", "rhash", "h5")
        missing = [key for key in required if key not in sidecar]
        if missing:
            raise ValueError(f"ACL quality sidecar '{quality_path}' missing required keys: {missing}")
        actual_hash, h5_rows = self._h5_reward_fingerprint(self._offline_dataset_path)
        stored_hash = str(np.asarray(sidecar["rhash"]).item())
        stored_rows = int(np.asarray(sidecar["n"]).item())
        if stored_hash != actual_hash or stored_rows != h5_rows:
            raise ValueError(
                f"ACL quality sidecar/H5 mismatch: sidecar n={stored_rows} rhash={stored_hash}, "
                f"h5 n={h5_rows} rhash={actual_hash}. Re-run scripts/acl_precompute_quality.py."
            )

    @staticmethod
    def _h5_reward_fingerprint(h5_path) -> tuple[str, int]:
        with h5py.File(h5_path, "r") as h5_file:
            reward_key = "rewards" if "rewards" in h5_file else "reward"
            rewards = h5_file[reward_key]
            num_rows = int(h5_file.attrs.get("num_samples", rewards.shape[0]))
            first = np.asarray(rewards[: min(1000, num_rows)], dtype=np.float64).reshape(-1)
            last = np.asarray(rewards[max(0, num_rows - 1000): num_rows], dtype=np.float64).reshape(-1)
        rhash = hashlib.sha256(
            np.ascontiguousarray(first).tobytes()
            + np.ascontiguousarray(last).tobytes()
        ).hexdigest()[:16]
        return rhash, num_rows

    def _install_acl_index_capture(self) -> None:
        sampler = self._offline_gpu_cache if self._offline_gpu_cache is not None else self._offline_shuffle_buffer
        if sampler is None:
            raise RuntimeError("ACL-QL requires an offline sampler.")
        original_sample = sampler.sample

        def sample_with_index(batch_size: int):
            batch = original_sample(batch_size=batch_size)
            if "dataset_index" not in batch:
                raise RuntimeError("Offline sampler batch has no dataset_index; ACL-QL cannot map quality labels.")
            self._acl_last_dataset_index = batch["dataset_index"]
            return batch

        sampler.sample = sample_with_index  # type: ignore[method-assign]

    def _sample_offline_batch(self, batch_size: int, normalize_obs, normalize_critic_obs) -> TensorDict:
        data = super()._sample_offline_batch(batch_size, normalize_obs, normalize_critic_obs)
        assert self._acl_quality_table is not None and self._acl_last_dataset_index is not None
        quality = self._acl_quality_table[self._acl_last_dataset_index.to(self.device, torch.long)]
        effective_batch_size = int(data.batch_size[0])
        if effective_batch_size != quality.shape[0]:
            if effective_batch_size % quality.shape[0] != 0:
                raise RuntimeError("Cannot align ACL quality labels with the augmented batch")
            quality = quality.repeat(effective_batch_size // quality.shape[0])
        self._acl_batch_quality = quality
        data["acl_quality"] = quality
        return data

    def _pretrain_behavior_policy(self) -> None:
        if self.config.acl_behavior_pretrain_steps <= 0:
            logger.warning("ACL-QL behavior pretraining is disabled; this deviates from Algorithm 1.")
            return
        assert self.behavior_actor is not None and self.behavior_optimizer is not None
        logger.info(f"ACL-QL behavior policy BC pretrain: steps={self.config.acl_behavior_pretrain_steps}")
        for _ in range(self.config.acl_behavior_pretrain_steps):
            data = self._sample_offline_batch(
                self.config.batch_size,
                self.obs_normalizer,
                self.critic_obs_normalizer,
            )
            pred_actions = self.behavior_actor(data["observations"])[0]
            target_actions = self._to_critic_actions(data["actions"])
            bc_loss = F.mse_loss(pred_actions, target_actions)
            self.behavior_optimizer.zero_grad(set_to_none=True)
            bc_loss.backward()
            self.behavior_optimizer.step()
        self._acl_last_metrics["acl_behavior_loss"] = bc_loss.detach()

    def _normalized_action_delta(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Holosoma CQL sets actor/critic/dataset actions to the env-scaled action
        # coordinate. We convert all three actions to the shared normalized actor
        # coordinate before applying the ACL paper's L2 distance.
        scale = self.actor.action_scale.to(device=a.device, dtype=a.dtype).clamp_min(1e-6)
        return (a - b) / scale

    def _normalized_action_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return acl_action_distance(self._normalized_action_delta(a, b), self.config.acl_distance_mode)

    @staticmethod
    def _add_distribution_stats(stats: dict[str, torch.Tensor], prefix: str, values: torch.Tensor) -> None:
        values = values.detach().float().reshape(-1)
        quantiles = torch.quantile(
            values,
            torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99], device=values.device),
        )
        stats[f"{prefix}_min"] = values.min()
        stats[f"{prefix}_p01"] = quantiles[0]
        stats[f"{prefix}_p05"] = quantiles[1]
        stats[f"{prefix}_mean"] = values.mean()
        stats[f"{prefix}_p50"] = quantiles[2]
        stats[f"{prefix}_p95"] = quantiles[3]
        stats[f"{prefix}_p99"] = quantiles[4]
        stats[f"{prefix}_max"] = values.max()

    @staticmethod
    def _ess_frac(values: torch.Tensor) -> torch.Tensor:
        values = values.detach().float().reshape(-1)
        return values.sum().square() / (values.square().sum().clamp_min(1e-12) * values.numel())

    def _build_sampled_conservative_losses(
        self,
        data: TensorDict,
        dataset_actions: torch.Tensor,
        q1_data: torch.Tensor,
        q2_data: torch.Tensor,
        q1_lse: torch.Tensor,
        q2_lse: torch.Tensor,
        q1_rand: torch.Tensor,
        q2_rand: torch.Tensor,
        q1_curr: torch.Tensor,
        q2_curr: torch.Tensor,
        q1_next: torch.Tensor,
        q2_next: torch.Tensor,
        curr_actions: torch.Tensor,
        curr_logp: torch.Tensor,
        random_density: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.acl_weight_net is not None and self.behavior_actor is not None and self._acl_batch_quality is not None
        batch_size = dataset_actions.shape[0]
        num_repeat = curr_actions.shape[0] // batch_size
        critic_obs = data["critic_observations"]
        actor_obs = data["observations"]
        expanded_critic_obs = critic_obs[:, None, :].expand(batch_size, num_repeat, -1).reshape(batch_size * num_repeat, -1)

        with torch.no_grad():
            behavior_actions, behavior_logp = self.behavior_actor.get_actions_and_log_probs(actor_obs)
            log_mu = curr_logp.view(batch_size, num_repeat).mean(dim=1)
            log_beta = behavior_logp.view(-1)
            m_in = self._acl_batch_quality.view(-1)
            mu_distance = self._normalized_action_distance(
                curr_actions.view(batch_size, num_repeat, -1).mean(dim=1),
                dataset_actions,
            )
            beta_distance = self._normalized_action_distance(behavior_actions, dataset_actions)
            m_mu_raw = acl_quality_raw(m_in, mu_distance)
            m_beta_raw = acl_quality_raw(m_in, beta_distance)
            m_mu = clamp_acl_quality(m_mu_raw)
            m_beta = clamp_acl_quality(m_beta_raw)
            d_ord = (1.0 - m_mu) * self._acl_r_max
            d_cql = m_mu * self._acl_r_max

        w_mu_all, _ = self.acl_weight_net(expanded_critic_obs, curr_actions)
        _, w_beta = self.acl_weight_net(critic_obs, behavior_actions)
        w_mu = w_mu_all.view(batch_size, num_repeat).mean(dim=1)
        q1_mu = q1_curr.mean(dim=1)
        q2_mu = q2_curr.mean(dim=1)
        q1_beta, q2_beta = self.qnet(critic_obs, behavior_actions)

        if w_mu.shape != q1_mu.shape or w_beta.shape != q1_beta.view(-1).shape:
            raise RuntimeError(f"ACL shape mismatch: {w_mu.shape=} {q1_mu.shape=} {w_beta.shape=} {q1_beta.shape=}")

        # Critic update: detach learned weights so gradients flow only to Q.
        cql1 = w_mu.detach() * q1_mu - w_beta.detach() * q1_beta.view(-1)
        cql2 = w_mu.detach() * q2_mu - w_beta.detach() * q2_beta.view(-1)

        self._acl_weight_update_context = {
            "critic_obs": critic_obs.detach(),
            "curr_actions": curr_actions.detach(),
            "behavior_actions": behavior_actions.detach(),
            "m_mu": m_mu.detach(),
            "m_beta": m_beta.detach(),
            "log_mu": log_mu.detach(),
            "log_beta": log_beta.detach(),
            "d_ord": d_ord.detach(),
            "d_cql": d_cql.detach(),
        }

        metric_update: dict[str, torch.Tensor] = {
            "acl/behavior_logp_mean": behavior_logp.detach().mean(),
            "acl/behavior_logp_dataset_action_mean": self.behavior_actor.log_prob_dataset_actions(
                actor_obs,
                dataset_actions,
            ).detach().mean(),
            "acl/m_mu_clamp_low_frac": (m_mu_raw < 1e-6).float().mean(),
            "acl/m_mu_clamp_high_frac": (m_mu_raw > 1.0 - 1e-6).float().mean(),
            "acl/m_beta_clamp_low_frac": (m_beta_raw < 1e-6).float().mean(),
            "acl/m_beta_clamp_high_frac": (m_beta_raw > 1.0 - 1e-6).float().mean(),
            "acl/w_mu_ess_frac": self._ess_frac(w_mu),
            "acl/w_beta_ess_frac": self._ess_frac(w_beta),
            "acl/weight_mass_diff": w_mu.detach().mean() - w_beta.detach().mean(),
            "acl/distance_mode_is_rms": torch.as_tensor(
                float(self.config.acl_distance_mode == "rms"),
                device=self.device,
            ),
        }
        for name, values in (
            ("acl/m_mu_raw", m_mu_raw),
            ("acl/m_beta_raw", m_beta_raw),
            ("acl/m_mu", m_mu),
            ("acl/m_beta", m_beta),
            ("acl/mu_distance", mu_distance),
            ("acl/beta_distance", beta_distance),
            ("acl/w_mu", w_mu),
            ("acl/w_beta", w_beta),
            ("acl/dataset_action", dataset_actions),
            ("acl/policy_action", curr_actions.view(batch_size, num_repeat, -1).mean(dim=1)),
            ("acl/behavior_action", behavior_actions),
            ("acl/action_scale", self.actor.action_scale),
        ):
            self._add_distribution_stats(metric_update, name, values)
        if "motion_phase" in data:
            phase_bins = torch.clamp((data["motion_phase"].view(-1)[:batch_size] * 20).long(), 0, 19)
            for bin_idx in (4, 5, 12, 13):
                mask = phase_bins == bin_idx
                if mask.any():
                    metric_update[f"acl/phase_bin_{bin_idx}/m_kappa"] = w_mu.detach()[mask].mean() - w_beta.detach()[mask].mean()
        self._acl_last_metrics.update(metric_update)
        return cql1, cql2

    def _after_q_update(self, data: TensorDict) -> None:
        if self._acl_weight_update_context:
            self._update_acl_weight_network(**self._acl_weight_update_context)
            self._acl_weight_update_context = {}

    def _update_acl_weight_network(
        self,
        critic_obs: torch.Tensor,
        curr_actions: torch.Tensor,
        behavior_actions: torch.Tensor,
        m_mu: torch.Tensor,
        m_beta: torch.Tensor,
        log_mu: torch.Tensor,
        log_beta: torch.Tensor,
        d_ord: torch.Tensor,
        d_cql: torch.Tensor,
    ) -> None:
        assert self.acl_weight_net is not None and self.acl_weight_optimizer is not None
        batch_size = critic_obs.shape[0]
        num_repeat = curr_actions.shape[0] // batch_size
        expanded_critic_obs = critic_obs[:, None, :].expand(batch_size, num_repeat, -1).reshape(batch_size * num_repeat, -1)
        w_mu_all, _ = self.acl_weight_net(expanded_critic_obs, curr_actions)
        _, w_beta = self.acl_weight_net(critic_obs, behavior_actions)
        w_mu = w_mu_all.view(batch_size, num_repeat).mean(dim=1)

        mono = compute_acl_monotonicity_loss(w_mu, w_beta, m_mu, m_beta)
        ord_loss, cql_loss = compute_acl_surrogate_loss(
            w_mu,
            w_beta,
            log_mu,
            log_beta,
            d_ord,
            d_cql,
            alpha=self.config.cql_weight,
        )
        pos = compute_acl_positivity_loss(w_mu, w_beta)
        total = mono + ord_loss + cql_loss + pos
        self.acl_weight_optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.acl_weight_optimizer.step()
        self._acl_last_metrics.update({
            "acl_weight_loss_1": ord_loss.detach(),
            "acl_weight_loss_2": cql_loss.detach(),
            "acl_monotonicity_loss": mono.detach(),
            "acl_surrogate_loss": (ord_loss + cql_loss).detach(),
            "acl_positivity_loss": pos.detach(),
            "acl_total_weight_loss": total.detach(),
        })

    @torch.no_grad()
    def _compute_action_ood_stats(self, data: TensorDict) -> dict[str, torch.Tensor]:
        stats = super()._compute_action_ood_stats(data)
        stats.update(self._acl_last_metrics)
        return stats
