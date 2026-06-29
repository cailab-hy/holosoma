from __future__ import annotations

import math

from loguru import logger

from holosoma.agents.bf_cql.bf_cql import FactorizedActor, resolve_action_groups
from holosoma.agents.cql.cql_agent import CQLAgent
from holosoma.utils.safe_torch_import import F, TensorDict, optim, torch


class BFCQLAgent(CQLAgent):
    def setup(self) -> None:
        if self.config.use_cnn_encoder:
            raise ValueError("BF-CQL currently supports vector observations only; set use_cnn_encoder=False.")

        super().setup()

        args = self.config
        device = self.device
        env = self.env
        n_act = env.robot_config.actions_dim
        group_names, group_indices = resolve_action_groups(args.bf_cql_action_grouping, env.robot_config.dof_names)

        actor_obs_keys = list(args.actor_obs_keys)
        action_scale = torch.ones(n_act, device=device)
        action_bias = torch.zeros(n_act, device=device)
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
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=args.actor_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.bf_cql_group_names = group_names
        self.bf_cql_group_indices = group_indices

        def _env_policy(obs: torch.Tensor, dones: torch.Tensor | None = None, deterministic: bool = False) -> torch.Tensor:
            return self._to_env_actions(self.actor.explore(obs, dones=dones, deterministic=deterministic))

        self.policy = _env_policy
        if self.is_multi_gpu:
            self._synchronize_model_parameters()

        logger.info(
            "BF-CQL action groups: "
            + ", ".join(
                f"{name}:{list(indices)}" for name, indices in zip(group_names, group_indices, strict=True)
            )
        )

    def _counterfactual_group_actions(
        self,
        base_actions: torch.Tensor,
        group_indices: tuple[int, ...],
        group_actions: torch.Tensor,
    ) -> torch.Tensor:
        counterfactual_actions = base_actions.clone()
        counterfactual_actions[:, list(group_indices)] = group_actions
        return counterfactual_actions

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
    ]:
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

                with torch.no_grad():
                    curr_actions, _, curr_group_logps = self.actor.get_actions_and_group_log_probs(expanded_obs)
                    next_actions_rep, _, next_group_logps = self.actor.get_actions_and_group_log_probs(
                        expanded_next_obs
                    )

                cql1_loss_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                cql2_loss_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                rand_q_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                curr_q_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                next_q_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                curr_logp_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                next_logp_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)
                random_density_total = torch.zeros((), device=self.device, dtype=bellman_loss.dtype)

                for group_indices, curr_group_logp, next_group_logp in zip(
                    self.bf_cql_group_indices,
                    curr_group_logps,
                    next_group_logps,
                    strict=True,
                ):
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
                    curr_counterfactual_actions = self._counterfactual_group_actions(
                        expanded_dataset_actions,
                        group_indices,
                        curr_actions[:, list(group_indices)],
                    )
                    next_counterfactual_actions = self._counterfactual_group_actions(
                        expanded_dataset_actions,
                        group_indices,
                        next_actions_rep[:, list(group_indices)],
                    )

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
                    curr_group_logp = curr_group_logp.view(batch_size, num_repeat)
                    next_group_logp = next_group_logp.view(batch_size, num_repeat)

                    q1_terms = torch.cat(
                        [
                            q1_rand - random_density,
                            q1_curr - curr_group_logp,
                            q1_next - next_group_logp,
                        ],
                        dim=1,
                    )
                    q2_terms = torch.cat(
                        [
                            q2_rand - random_density,
                            q2_curr - curr_group_logp,
                            q2_next - next_group_logp,
                        ],
                        dim=1,
                    )

                    cql1_loss_total = cql1_loss_total + (
                        torch.logsumexp(q1_terms / self._temperature, dim=1) * self._temperature - q1
                    ).mean()
                    cql2_loss_total = cql2_loss_total + (
                        torch.logsumexp(q2_terms / self._temperature, dim=1) * self._temperature - q2
                    ).mean()
                    rand_q_total = rand_q_total + 0.5 * (
                        (q1_rand - random_density).mean() + (q2_rand - random_density).mean()
                    )
                    curr_q_total = curr_q_total + 0.5 * (q1_curr.mean() + q2_curr.mean())
                    next_q_total = next_q_total + 0.5 * (q1_next.mean() + q2_next.mean())
                    curr_logp_total = curr_logp_total + curr_group_logp.mean()
                    next_logp_total = next_logp_total + next_group_logp.mean()
                    random_density_total = random_density_total + torch.tensor(
                        random_density,
                        device=self.device,
                        dtype=bellman_loss.dtype,
                    )

                num_groups = float(len(self.bf_cql_group_indices))
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
        )


__all__ = ["BFCQLAgent"]
