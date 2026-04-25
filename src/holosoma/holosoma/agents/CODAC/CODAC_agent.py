from __future__ import annotations

"""CODAC offline agent (Stage-1 CODAC approximation).

Important notes:
1. This is not a full CODAC implementation yet. It is a Stage-1 CODAC
   approximation built on top of the stable offline_sac baseline.
2. Distributional critics are preserved end-to-end, but conservative
   regularization is currently computed on mean-Q expectation E[Z].
3. This staged design is intentional:
   - keep the strong stability of existing offline_sac,
   - add conservative pressure incrementally,
   - avoid regressing existing FastSAC HDF5 compatibility.

Dataset semantics:
- FastSAC-exported HDF5 is consumed as-is.
- Terminal next observations are trusted from dataset fields and are not
  reinterpreted in the loader path.
"""

import math
import os
from typing import Any

import tqdm
from loguru import logger

from holosoma.agents.CODAC.CODAC_utils import save_params
from holosoma.agents.offline_sac.offline_sac_agent import OfflineSACAgent, OfflineSACEnv
from holosoma.config_types.algo import CODACConfig
from holosoma.utils.safe_torch_import import F, TensorDict, autocast, optim, torch


class CODACAgent(OfflineSACAgent):
    """Conservative Offline Distributional Actor-Critic (Stage-1)."""

    config: CODACConfig

    def __init__(
        self,
        env,
        config: CODACConfig,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict | None = None,
    ):
        if config.conservative_weight < 0.0:
            raise ValueError(f"conservative_weight must be >= 0, got {config.conservative_weight}")
        if config.conservative_temperature <= 0.0:
            raise ValueError(
                f"conservative_temperature must be > 0, got {config.conservative_temperature}"
            )
        if config.num_action_samples <= 0:
            raise ValueError(f"num_action_samples must be > 0, got {config.num_action_samples}")
        if config.actor_q_aggregation not in ("mean", "min"):
            raise ValueError(
                f"actor_q_aggregation must be one of ['mean', 'min'], got {config.actor_q_aggregation}"
            )
        if config.critic_conservative_mode not in ("mean_q_stage1", "upper_quantile", "cvar", "atom_level"):
            raise ValueError(
                "critic_conservative_mode must be one of "
                "['mean_q_stage1', 'upper_quantile', 'cvar', 'atom_level'], "
                f"got {config.critic_conservative_mode}"
            )
        if config.use_lagrange:
            if config.cql_lagrange_init <= 0.0:
                raise ValueError(
                    f"cql_lagrange_init must be > 0 when use_lagrange=True, got {config.cql_lagrange_init}"
                )
            if config.cql_lagrange_learning_rate <= 0.0:
                raise ValueError(
                    "cql_lagrange_learning_rate must be > 0 when use_lagrange=True, "
                    f"got {config.cql_lagrange_learning_rate}"
                )
            if config.cql_lagrange_max <= 0.0:
                raise ValueError(f"cql_lagrange_max must be > 0, got {config.cql_lagrange_max}")

        super().__init__(env=env, config=config, device=device, log_dir=log_dir, multi_gpu_cfg=multi_gpu_cfg)

        self.log_cql_alpha: torch.Tensor | None = None
        self.cql_alpha_optimizer: torch.optim.Optimizer | None = None

    def setup(self) -> None:
        super().setup()

        args = self.config
        if args.use_lagrange:
            init = max(float(args.cql_lagrange_init), 1e-6)
            self.log_cql_alpha = torch.tensor([math.log(init)], requires_grad=True, device=self.device)
            fused = str(self.device).startswith("cuda")
            self.cql_alpha_optimizer = optim.AdamW(
                [self.log_cql_alpha],
                lr=args.cql_lagrange_learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.0,
                fused=fused,
            )
            if self.is_multi_gpu:
                torch.distributed.broadcast(self.log_cql_alpha.data, src=0)
        else:
            self.log_cql_alpha = None
            self.cql_alpha_optimizer = None

        logger.info(
            "CODAC Stage-1 conservative mode configured: "
            f"mode={args.critic_conservative_mode}, "
            f"actor_q_aggregation={args.actor_q_aggregation}, "
            f"use_lagrange={args.use_lagrange}, "
            f"num_action_samples={args.num_action_samples}, "
            f"temperature={args.conservative_temperature:.4f}, "
            f"weight={args.conservative_weight:.4f}"
        )

    def _maybe_amp(self):
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        is_cuda = str(self.device).startswith("cuda")
        device_type = "cuda" if is_cuda else "cpu"
        return autocast(device_type=device_type, dtype=amp_dtype, enabled=self.config.amp and is_cuda)

    @staticmethod
    def _repeat_obs_for_action_samples(obs: torch.Tensor, num_action_samples: int) -> torch.Tensor:
        return obs.unsqueeze(1).expand(-1, num_action_samples, -1).reshape(-1, obs.shape[-1])

    def _sample_random_actions(
        self,
        batch_size: int,
        num_action_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_act = self.env.robot_config.actions_dim
        total = batch_size * num_action_samples

        if hasattr(self.actor, "action_scale") and hasattr(self.actor, "action_bias"):
            action_scale = self.actor.action_scale
            action_bias = self.actor.action_bias
        else:
            action_scale = torch.ones(n_act, device=self.device, dtype=torch.float32)
            action_bias = torch.zeros(n_act, device=self.device, dtype=torch.float32)

        random_u = torch.rand((total, n_act), device=self.device, dtype=torch.float32) * 2.0 - 1.0
        random_actions = random_u * action_scale + action_bias

        # Uniform density under action bounds:
        # p(a) = 1 / prod_i (2 * action_scale_i)
        # log p(a) is constant across sampled actions.
        random_log_prob_const = -torch.log((2.0 * action_scale).clamp_min(1e-6)).sum()
        random_log_prob = random_log_prob_const.expand(total)
        return random_actions, random_log_prob

    def _reduce_actor_q(self, q_values: torch.Tensor) -> torch.Tensor:
        if self.config.actor_q_aggregation == "mean":
            return q_values.mean(dim=0)
        # Offline setting default uses min aggregation to reduce overestimation risk.
        # TODO: compare mean/min ensemble aggregation for actor update in offline setting.
        return q_values.min(dim=0).values

    def _compute_stage1_conservative_regularizer(
        self,
        data: TensorDict,
        q_data_values: torch.Tensor,
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
        """Stage-1 CODAC approximation on mean-Q expectation E[Z].

        Conservative objective:
            cat_q = [Q_rand - log p_rand, Q_curr - log p_curr, Q_next - log p_next]
            L_cons = temperature * logsumexp(cat_q / temperature) - Q_data

        where all Q values are distribution expectations E[Z] from the
        distributional critic.

        TODO: replace mean-Q conservative term with quantile-wise conservative regularizer.
        TODO: add upper-tail-only conservative penalty.
        TODO: add CVaR-based conservative backup.
        """
        args = self.config
        num_action_samples = int(args.num_action_samples)
        batch_size = int(data["critic_observations"].shape[0])

        actor_obs = data["observations"]
        critic_obs = data["critic_observations"]
        next_actor_obs = data["next"]["observations"]
        next_critic_obs = data["next"]["critic_observations"]

        actor_obs_rep = self._repeat_obs_for_action_samples(actor_obs, num_action_samples)
        critic_obs_rep = self._repeat_obs_for_action_samples(critic_obs, num_action_samples)
        next_actor_obs_rep = self._repeat_obs_for_action_samples(next_actor_obs, num_action_samples)
        next_critic_obs_rep = self._repeat_obs_for_action_samples(next_critic_obs, num_action_samples)

        with torch.no_grad(), self._maybe_amp():
            current_actions, current_log_probs = self.actor.get_actions_and_log_probs(actor_obs_rep)
            next_actions, next_log_probs = self.actor.get_actions_and_log_probs(next_actor_obs_rep)
            random_actions, random_log_probs = self._sample_random_actions(batch_size, num_action_samples)

        with self._maybe_amp():
            q_rand_logits = self.qnet(critic_obs_rep, random_actions)
            q_curr_logits = self.qnet(critic_obs_rep, current_actions)
            q_next_logits = self.qnet(next_critic_obs_rep, next_actions)

            q_rand_values = self.qnet.get_value(F.softmax(q_rand_logits, dim=-1))
            q_curr_values = self.qnet.get_value(F.softmax(q_curr_logits, dim=-1))
            q_next_values = self.qnet.get_value(F.softmax(q_next_logits, dim=-1))

            num_q = int(q_data_values.shape[0])
            q_rand_values = q_rand_values.reshape(num_q, batch_size, num_action_samples)
            q_curr_values = q_curr_values.reshape(num_q, batch_size, num_action_samples)
            q_next_values = q_next_values.reshape(num_q, batch_size, num_action_samples)

            random_log_probs = random_log_probs.reshape(batch_size, num_action_samples)
            current_log_probs = current_log_probs.reshape(batch_size, num_action_samples)
            next_log_probs = next_log_probs.reshape(batch_size, num_action_samples)

            q_rand_adjusted = q_rand_values - random_log_probs.unsqueeze(0)
            q_curr_adjusted = q_curr_values - current_log_probs.unsqueeze(0)
            q_next_adjusted = q_next_values - next_log_probs.unsqueeze(0)

            cat_q = torch.cat([q_rand_adjusted, q_curr_adjusted, q_next_adjusted], dim=2)

            temperature = max(float(args.conservative_temperature), 1e-6)
            conservative_lse = torch.logsumexp(cat_q / temperature, dim=2) * temperature
            conservative_per_q = conservative_lse - q_data_values
            conservative_unweighted = conservative_per_q.mean(dim=1).sum(dim=0)

        return (
            conservative_unweighted,
            q_data_values.mean().detach(),
            q_rand_values.mean().detach(),
            q_curr_values.mean().detach(),
            q_next_values.mean().detach(),
            q_rand_adjusted.mean().detach(),
            q_curr_adjusted.mean().detach(),
            q_next_adjusted.mean().detach(),
        )

    def _update_critic(
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
    ]:
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
            bellman_loss = critic_losses.mean(dim=1).sum(dim=0)

            q_probs = F.softmax(q_outputs, dim=-1)
            q_data_values = self.qnet.get_value(q_probs)

            if args.critic_conservative_mode != "mean_q_stage1":
                raise NotImplementedError(
                    "Only critic_conservative_mode='mean_q_stage1' is implemented in Stage-1 CODAC. "
                    f"Got {args.critic_conservative_mode}."
                )

            (
                conservative_unweighted,
                codac_q_data_mean,
                codac_q_rand_mean,
                codac_q_curr_mean,
                codac_q_next_mean,
                codac_rand_contrib,
                codac_curr_contrib,
                codac_next_contrib,
            ) = self._compute_stage1_conservative_regularizer(data, q_data_values)

            codac_alpha_value = torch.tensor(1.0, device=self.device)
            lagrange_loss = torch.tensor(0.0, device=self.device)

            if args.use_lagrange and self.log_cql_alpha is not None:
                cql_alpha = torch.clamp(self.log_cql_alpha.exp(), min=0.0, max=args.cql_lagrange_max)
                codac_alpha_value = cql_alpha.detach()
                conservative_loss = args.conservative_weight * cql_alpha.detach() * (
                    conservative_unweighted - args.target_action_gap
                )
            else:
                conservative_loss = args.conservative_weight * conservative_unweighted

            total_q_loss = bellman_loss + conservative_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_q_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(self.qnet)

        scaler.unscale_(self.q_optimizer)
        if args.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), args.max_grad_norm)
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)

        scaler.step(self.q_optimizer)
        scaler.update()

        if (
            args.use_lagrange
            and self.log_cql_alpha is not None
            and self.cql_alpha_optimizer is not None
        ):
            cql_alpha = torch.clamp(self.log_cql_alpha.exp(), min=0.0, max=args.cql_lagrange_max)
            lagrange_loss = -(cql_alpha * (conservative_unweighted.detach() - args.target_action_gap))

            self.cql_alpha_optimizer.zero_grad(set_to_none=True)
            lagrange_loss.backward()

            if self.is_multi_gpu and self.log_cql_alpha.grad is not None:
                torch.distributed.all_reduce(self.log_cql_alpha.grad.data, op=torch.distributed.ReduceOp.SUM)
                self.log_cql_alpha.grad.data.copy_(self.log_cql_alpha.grad.data / self.gpu_world_size)

            self.cql_alpha_optimizer.step()
            codac_alpha_value = cql_alpha.detach()

        return (
            rewards.mean().detach(),
            critic_grad_norm.detach(),
            bellman_loss.detach(),
            conservative_loss.detach(),
            total_q_loss.detach(),
            conservative_unweighted.detach(),
            codac_q_data_mean,
            codac_q_rand_mean,
            codac_q_curr_mean,
            codac_q_next_mean,
            conservative_unweighted.detach(),
            codac_alpha_value.detach(),
            lagrange_loss.detach(),
            codac_rand_contrib,
            codac_curr_contrib,
            codac_next_contrib,
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
            qf_value = self._reduce_actor_q(q_values)
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

    def load(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return

        super().load(ckpt_path)

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if self.log_cql_alpha is not None and "log_cql_alpha" in ckpt:
            self.log_cql_alpha.data.copy_(ckpt["log_cql_alpha"].to(self.device))
        if self.cql_alpha_optimizer is not None and "cql_alpha_optimizer_state_dict" in ckpt:
            self.cql_alpha_optimizer.load_state_dict(ckpt["cql_alpha_optimizer_state_dict"])

    def save(self, path: str) -> None:  # type: ignore[override]
        env_state = self._collect_env_state()
        metadata = self._checkpoint_metadata(iteration=self.global_step)
        metadata["action_space_mode"] = "env_scaled_action_training_v1"

        extra_state: dict[str, Any] = {
            "critic_conservative_mode": self.config.critic_conservative_mode,
            "actor_q_aggregation": self.config.actor_q_aggregation,
        }
        if self.log_cql_alpha is not None:
            extra_state["log_cql_alpha"] = self.log_cql_alpha.detach().cpu()
        if self.cql_alpha_optimizer is not None:
            extra_state["cql_alpha_optimizer_state_dict"] = self.cql_alpha_optimizer.state_dict()

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
            extra_state=extra_state,
        )

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
            update_alpha = self._update_alpha
        else:
            update_critic = self._update_critic
            update_actor = self._update_actor
            update_alpha = self._update_alpha

        if self.env.num_envs > 1 and self.is_main_process:
            logger.warning(
                "CODAC offline training does not use vectorized rollout collection. "
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

                    (
                        reward_mean,
                        critic_grad_norm,
                        codac_bellman_loss,
                        codac_conservative_loss,
                        codac_total_q_loss,
                        codac_conservative_unweighted,
                        codac_q_data_mean,
                        codac_q_rand_mean,
                        codac_q_curr_mean,
                        codac_q_next_mean,
                        codac_gap,
                        codac_alpha_value,
                        codac_lagrange_loss,
                        codac_rand_contrib,
                        codac_curr_contrib,
                        codac_next_contrib,
                    ) = update_critic(data)
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
                            "codac_bellman_loss": codac_bellman_loss,
                            "codac_conservative_loss": codac_conservative_loss,
                            "codac_total_q_loss": codac_total_q_loss,
                            "codac_conservative_unweighted": codac_conservative_unweighted,
                            "codac_q_data_mean": codac_q_data_mean,
                            "codac_q_rand_mean": codac_q_rand_mean,
                            "codac_q_curr_mean": codac_q_curr_mean,
                            "codac_q_next_mean": codac_q_next_mean,
                            "codac_gap": codac_gap,
                            "codac_alpha_value": codac_alpha_value,
                            "codac_lagrange_loss": codac_lagrange_loss,
                            "codac_rand_contrib": codac_rand_contrib,
                            "codac_curr_contrib": codac_curr_contrib,
                            "codac_next_contrib": codac_next_contrib,
                            "critic_grad_norm": critic_grad_norm,
                            "alpha_loss": alpha_loss,
                            "alpha_value": self.log_alpha.exp().detach().mean(),
                            "actor_loss": actor_loss,
                            "actor_grad_norm": actor_grad_norm,
                            "policy_entropy": policy_entropy,
                            "action_std": action_std,
                            "is_actor_update_step": float(is_actor_update_step),
                            "actor_q_aggregation_is_min": float(args.actor_q_aggregation == "min"),
                            "actor_q_aggregation_is_mean": float(args.actor_q_aggregation == "mean"),
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
                    logger.info(f"Saving CODAC model at global step {self.global_step}")
                    self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
                    self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))

            pbar.update(1)

        pbar.close()

        if self.is_main_process and self.global_step >= args.num_learning_iterations:
            self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
            self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))


CODACEnv = OfflineSACEnv


# Backward-compatible aliases for older class naming styles.
CODAC_AGENT = CODACAgent
CODAC_ENV = CODACEnv


# -----------------------------------------------------------------------------
# CODAC Stage-1 configuration/expansion notes
# -----------------------------------------------------------------------------
# Added config fields for this stage:
# - conservative_weight
# - conservative_temperature
# - num_action_samples
# - use_lagrange
# - target_action_gap
# - cql_lagrange_init
# - cql_lagrange_learning_rate
# - cql_lagrange_max
# - actor_q_aggregation                # "mean" or "min"
# - critic_conservative_mode           # stage-1 implements "mean_q_stage1"
#
# Full-CODAC expansion points:
# - TODO: replace mean-Q conservative term with quantile-wise conservative regularizer
# - TODO: add upper-tail-only conservative penalty
# - TODO: add CVaR-based conservative backup
# - TODO: add atom-level conservative regularization
# - TODO: compare mean/min ensemble aggregation for actor update in offline setting
