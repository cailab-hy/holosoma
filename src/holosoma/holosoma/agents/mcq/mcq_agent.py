"""Mildly Conservative Q-learning (MCQ) on top of the scalar CQL scaffolding.

MCQ (Lyu et al., NeurIPS 2022) replaces CQL's push-down regularizer with the
Mildly Conservative Bellman (MCB) operator: OOD actions are neither left
unconstrained nor pushed down without bound — they are *anchored* to an
in-support pseudo target

    y(s) = max_{a_i ~ beta_hat(.|s), i=1..N} min_j Q_target_j(s, a_i)

via an MSE term on policy actions u ~ pi at both s and s'. The critic loss is

    L = lambda * bellman + (1 - lambda) * ood_anchor (+ dr3)

so lambda -> 1 recovers plain SAC-style TD learning and smaller lambda anchors
harder (the paper's ablation shows lambda <= 0.5 collapses Q — the same
underestimation failure mode this repo has been diagnosing).

Everything else (actor update, entropy tuning, target smoothing, dataset
pipeline, eval, export, probe compatibility) is inherited from CQLAgent.
Behavior actions come from a BCQ-style conditional VAE trained concurrently;
its state rides along in the checkpoint (loading a plain CQL checkpoint works
and simply starts the VAE fresh).
"""

from __future__ import annotations

from loguru import logger

from holosoma.agents.cql.cql_agent import CQLAgent
from holosoma.agents.mcq.mcq import BehaviorVAE
from holosoma.config_types.algo import MCQConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.safe_torch_import import F, TensorDict, optim, torch


class MCQAgent(CQLAgent):
    config: MCQConfig

    def __init__(
        self,
        env: BaseTask,
        config: MCQConfig,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(env=env, config=config, device=device, log_dir=log_dir, multi_gpu_cfg=multi_gpu_cfg)

        if not (0.0 < config.mcq_lambda <= 1.0):
            raise ValueError(f"mcq_lambda must be in (0, 1], got {config.mcq_lambda}")
        if config.mcq_num_action_samples <= 0:
            raise ValueError(f"mcq_num_action_samples must be > 0, got {config.mcq_num_action_samples}")
        if config.mcq_num_ood_samples <= 0:
            raise ValueError(f"mcq_num_ood_samples must be > 0, got {config.mcq_num_ood_samples}")
        if config.vae_hidden_dim <= 0:
            raise ValueError(f"vae_hidden_dim must be > 0, got {config.vae_hidden_dim}")
        if config.vae_latent_dim < 0:
            raise ValueError(f"vae_latent_dim must be >= 0 (0 = auto), got {config.vae_latent_dim}")
        if config.vae_learning_rate <= 0.0:
            raise ValueError(f"vae_learning_rate must be > 0, got {config.vae_learning_rate}")
        if config.vae_kl_weight < 0.0:
            raise ValueError(f"vae_kl_weight must be >= 0, got {config.vae_kl_weight}")

        self._mcq_lambda = config.mcq_lambda
        self._mcq_num_action_samples = config.mcq_num_action_samples
        self._mcq_num_ood_samples = config.mcq_num_ood_samples

    def setup(self) -> None:
        super().setup()
        args = self.config
        n_act = self.env.robot_config.actions_dim
        latent_dim = args.vae_latent_dim if args.vae_latent_dim > 0 else 2 * n_act
        self._vae = BehaviorVAE(
            obs_dim=self.actor_obs_dim,
            action_dim=n_act,
            latent_dim=latent_dim,
            hidden_dim=args.vae_hidden_dim,
            action_scale=self.env_action_scale,
            action_bias=self.env_action_bias,
            device=self.device,
        )
        self._vae_optimizer = optim.AdamW(
            self._vae.parameters(),
            lr=args.vae_learning_rate,
            betas=(0.9, 0.95),
        )
        if self.is_multi_gpu:
            # Ranks seed differently; grads are all-reduced every step, so the
            # initial VAE parameters must match too (actor/qnet get this in
            # super().setup() via _synchronize_model_parameters).
            for param in self._vae.parameters():
                torch.distributed.broadcast(param.data, src=0)
        logger.info(
            "MCQ setup: lambda={} behavior_samples={} ood_samples={} vae(latent={}, hidden={})",
            self._mcq_lambda,
            self._mcq_num_action_samples,
            self._mcq_num_ood_samples,
            latent_dim,
            args.vae_hidden_dim,
        )

    @staticmethod
    def _expand_rows(x: torch.Tensor, repeats: int) -> torch.Tensor:
        batch_size = x.shape[0]
        return x[:, None, :].expand(batch_size, repeats, -1).reshape(batch_size * repeats, -1)

    @torch.no_grad()
    def _mcb_pseudo_targets(self, observations: torch.Tensor, critic_observations: torch.Tensor) -> torch.Tensor:
        """y(s) = max over N behavior-VAE samples of min-twin target Q. Shape [B]."""
        batch_size = observations.shape[0]
        num_samples = self._mcq_num_action_samples
        rep_obs = self._expand_rows(observations, num_samples)
        rep_cobs = self._expand_rows(critic_observations, num_samples)
        beta_actions = self._vae.decode(rep_obs)
        q1_beta, q2_beta = self.qnet_target(rep_cobs, self._to_critic_actions(beta_actions))
        q_min = torch.minimum(q1_beta, q2_beta).view(batch_size, num_samples)
        return q_min.max(dim=1).values

    def _update_vae(self, observations: torch.Tensor, dataset_actions_env: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vae_loss, recon_loss, kl_loss = self._vae.loss(
            observations.detach(),
            dataset_actions_env.detach(),
            kl_weight=self.config.vae_kl_weight,
        )
        self._vae_optimizer.zero_grad(set_to_none=True)
        vae_loss.backward()
        if self.is_multi_gpu:
            self._all_reduce_model_grads(self._vae)
        self._vae_optimizer.step()
        return vae_loss.detach(), recon_loss.detach(), kl_loss.detach()

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

        # Behavior model step (fp32, own optimizer) before the critic step.
        vae_loss, vae_recon_loss, vae_kl_loss = self._update_vae(data["observations"], data["actions"])

        with self._maybe_amp():
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
            dataset_actions = self._to_critic_actions(data["actions"])
            rewards = data["next"]["rewards"]
            rewards = reward_scale * rewards
            dones = data["next"]["dones"].bool()
            # Same truncation semantics as CQLAgent._update_q (see comment there).
            if args.bootstrap_truncations:
                truncations = data["next"]["truncations"].bool()
                bootstrap = (truncations | ~dones).float()
            else:
                bootstrap = (~dones).float()
            alpha = self.log_alpha.exp().detach()

            with torch.no_grad():
                discount = args.gamma ** data["next"]["effective_n_steps"]

                rewards_ = rewards.view(-1)
                bootstrap_ = bootstrap.view(-1)
                discount_ = discount.view(-1)

                if args.cql_max_target_backup:
                    batch_size = next_observations.shape[0]
                    num_backup_actions = args.cql_max_target_backup_samples
                    expanded_next_obs = self._expand_rows(next_observations, num_backup_actions)
                    expanded_next_critic_obs = self._expand_rows(next_critic_observations, num_backup_actions)

                    next_actions, next_log_probs = self.actor.get_actions_and_log_probs(expanded_next_obs)
                    next_q1_target, next_q2_target = self.qnet_target(expanded_next_critic_obs, next_actions)
                    next_q1_target = next_q1_target.view(batch_size, num_backup_actions)
                    next_q2_target = next_q2_target.view(batch_size, num_backup_actions)
                    next_log_probs = next_log_probs.view(batch_size, num_backup_actions)

                    next_target_min_q_all = torch.minimum(next_q1_target, next_q2_target)
                    next_target_min_q, max_target_indices = next_target_min_q_all.max(dim=1)
                    next_log_probs = next_log_probs.gather(dim=1, index=max_target_indices.unsqueeze(1)).squeeze(1)
                else:
                    next_actions, next_log_probs = self.actor.get_actions_and_log_probs(next_observations)
                    next_q1_target, next_q2_target = self.qnet_target(next_critic_observations, next_actions)
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
                q_target = q_target.clamp(min=-10000.0, max=10000.0)
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

            with torch.no_grad():
                pi_actions_det = self.actor(observations)[0]
                q1_pi_det, q2_pi_det = self.qnet(critic_observations, pi_actions_det)
                q_pi_minus_q_data = (
                    torch.minimum(q1_pi_det, q2_pi_det) - torch.minimum(q1.detach(), q2.detach())
                ).mean()

            # --- MCB anchoring: pull Q(s, u~pi) toward the best in-support value. ---
            batch_size = dataset_actions.shape[0]
            num_ood = self._mcq_num_ood_samples

            with torch.no_grad():
                pseudo_target_s = self._mcb_pseudo_targets(observations, critic_observations)
                pseudo_target_next = self._mcb_pseudo_targets(next_observations, next_critic_observations)

                ood_obs = self._expand_rows(observations, num_ood)
                ood_next_obs = self._expand_rows(next_observations, num_ood)
                ood_actions_s, ood_logp_s = self.actor.get_actions_and_log_probs(ood_obs)
                ood_actions_next, ood_logp_next = self.actor.get_actions_and_log_probs(ood_next_obs)

                pseudo_s_rep = pseudo_target_s[:, None].expand(batch_size, num_ood).reshape(-1)
                pseudo_next_rep = pseudo_target_next[:, None].expand(batch_size, num_ood).reshape(-1)

            ood_critic_obs = self._expand_rows(critic_observations, num_ood)
            ood_next_critic_obs = self._expand_rows(next_critic_observations, num_ood)
            q1_ood_s, q2_ood_s = self.qnet(ood_critic_obs, ood_actions_s.detach())
            q1_ood_next, q2_ood_next = self.qnet(ood_next_critic_obs, ood_actions_next.detach())

            # Average over the two anchored states so the raw term matches the
            # two-MSE scale of bellman_loss.
            ood_loss_raw = 0.5 * (
                F.mse_loss(q1_ood_s, pseudo_s_rep)
                + F.mse_loss(q2_ood_s, pseudo_s_rep)
                + F.mse_loss(q1_ood_next, pseudo_next_rep)
                + F.mse_loss(q2_ood_next, pseudo_next_rep)
            )
            conservative_loss = (1.0 - self._mcq_lambda) * ood_loss_raw
            cql_gap = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)

            rand_q_mean = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
            curr_q_mean = 0.5 * (q1_ood_s.mean() + q2_ood_s.mean())
            next_q_mean = 0.5 * (q1_ood_next.mean() + q2_ood_next.mean())
            random_density = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)

            q_loss = self._mcq_lambda * bellman_loss + conservative_loss + dr3_loss

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

        with torch.no_grad():
            q_ood_min_s = torch.minimum(q1_ood_s, q2_ood_s).view(batch_size, num_ood).mean(dim=1)
            self.training_metrics.add(
                {
                    "mcq/vae_loss": vae_loss,
                    "mcq/vae_recon_loss": vae_recon_loss,
                    "mcq/vae_kl_loss": vae_kl_loss,
                    "mcq/ood_loss_raw": ood_loss_raw.detach(),
                    "mcq/pseudo_target_s": pseudo_target_s.mean(),
                    "mcq/pseudo_target_next": pseudo_target_next.mean(),
                    "mcq/anchor_minus_q_pi_s": (pseudo_target_s - q_ood_min_s).mean(),
                    "mcq/anchor_minus_q_data": (pseudo_target_s - torch.minimum(q1, q2)).mean(),
                    "mcq/lambda": torch.tensor(self._mcq_lambda, device=self.device),
                }
            )

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
            ood_logp_s.mean().detach(),
            ood_logp_next.mean().detach(),
            random_density,
        )

    def save(self, path: str) -> None:  # type: ignore[override]
        super().save(path)
        # save_params uploads the artifact inside super().save(), so the wandb
        # copy lacks the VAE keys; the local file gets them appended here and
        # load() tolerates their absence.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["vae_state_dict"] = {k: v.detach().cpu() for k, v in self._vae.state_dict().items()}
        checkpoint["vae_optimizer_state_dict"] = self._vae_optimizer.state_dict()
        torch.save(checkpoint, path)

    def load(self, ckpt_path: str | None) -> None:
        super().load(ckpt_path)
        if not ckpt_path:
            return
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        vae_state = checkpoint.get("vae_state_dict")
        if vae_state:
            self._vae.load_state_dict(vae_state)
            if "vae_optimizer_state_dict" in checkpoint:
                self._vae_optimizer.load_state_dict(checkpoint["vae_optimizer_state_dict"])
            logger.info("Restored MCQ behavior VAE state from checkpoint.")
        else:
            logger.warning(
                "Checkpoint has no vae_state_dict (e.g. a plain CQL checkpoint); MCQ behavior VAE starts fresh."
            )
