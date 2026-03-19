"""Offline CODAC agent — conservative offline distributional actor-critic.

Inherits **everything** from ``OfflineFastSACAgent`` and overrides only the
two conservative-loss hooks.  The training loop, dataset loading, normalizer
initialization, checkpoint I/O, target-network update, and evaluation path
are all inherited without modification.
"""

from __future__ import annotations

from holosoma.agents.offline_fast_sac.offline_fast_sac_agent import OfflineFastSACAgent
from holosoma.agents.offline_CODAC.offline_codac_config import CODACConfig
from holosoma.agents.offline_CODAC.offline_codac_losses import (
    compute_conservative_penalty,
    compute_q_values_for_actions,
    sample_ood_actions,
)
from holosoma.config_types.algo import FastSACConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.safe_torch_import import F, TensorDict, torch


class OfflineCODACAgent(OfflineFastSACAgent):
    """Conservative Offline Distributional Actor-Critic.

    Extends ``OfflineFastSACAgent`` with CODAC-style conservative
    regularization on the distributional critic.  All other components
    (actor architecture, critic architecture, dataset loading, observation
    normalization, training skeleton, evaluation, checkpoint I/O) are
    inherited as-is.

    The two extension hooks provided by the parent class are overridden:

    * ``_compute_conservative_critic_loss`` — adds a distributional
      conservative penalty (logsumexp over OOD Q minus dataset Q).
    * ``_compute_conservative_actor_loss`` — returns 0 by default
      (standard SAC actor objective is sufficient under CODAC).
    """

    def __init__(
        self,
        config: FastSACConfig,
        device: str,
        log_dir: str,
        codac_config: CODACConfig | None = None,
        env: BaseTask | None = None,
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(config, device, log_dir, env, multi_gpu_cfg)
        self.codac_config = codac_config or CODACConfig()

    # ------------------------------------------------------------------
    # Conservative loss hooks (CODAC-specific)
    # ------------------------------------------------------------------

    def _compute_conservative_critic_loss(
        self,
        data: TensorDict,
        q_outputs: torch.Tensor,
        critic_observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """CODAC conservative critic penalty.

        Computes a CQL-style distributional conservative regularization:

        .. math::

            \\mathcal{L}_{conservative} = \\alpha_c \\Big(
                \\tau \\cdot \\text{logsumexp}\\big(Q_{ood} / \\tau\\big)
                - Q_{dataset}
            \\Big)

        where OOD actions come from uniform-random and/or the current policy,
        and Q-values are the scalar expectations E[Z] from the distributional
        critic.

        This method is called inside the parent's ``_update_main`` within the
        ``_maybe_amp()`` context.  The returned scalar is multiplied by
        ``config.conservative_weight`` before being added to the total critic
        loss.

        Parameters are identical to the parent signature — see
        ``OfflineFastSACAgent._compute_conservative_critic_loss`` for docs.

        Returns
        -------
        torch.Tensor
            Scalar conservative penalty (already multiplied by
            ``codac_config.conservative_coef``).
        """
        codac = self.codac_config
        batch_size = critic_observations.shape[0]
        action_dim = actions.shape[1]

        # =============================================================
        # Step 1: Determine which OOD action sources to use
        # =============================================================
        mode = codac.conservative_action_sample_mode
        num_act = codac.num_conservative_actions

        use_random = mode in ("random", "random_policy", "random_policy_next")
        use_policy = mode in ("policy", "random_policy", "random_policy_next")
        use_next_policy = mode == "random_policy_next"

        # =============================================================
        # Step 2: Sample OOD actions
        # =============================================================
        ood_action_chunks: list[torch.Tensor] = []
        ood_counts: list[int] = []  # num_samples per chunk, for Q computation

        if use_random or use_policy:
            random_actions, policy_actions, _policy_lp = sample_ood_actions(
                actor=self.actor,
                observations=data["observations"],
                num_random=num_act if use_random else 0,
                num_policy=num_act if use_policy else 0,
                action_scale=self.actor.action_scale,
                action_bias=self.actor.action_bias,
                action_dim=action_dim,
            )

        if use_random:
            ood_action_chunks.append(random_actions)
            ood_counts.append(num_act)

        if use_policy:
            ood_action_chunks.append(policy_actions)
            ood_counts.append(num_act)

        if use_next_policy:
            # Sample actions from policy conditioned on *next* observations
            _, next_policy_actions, _ = sample_ood_actions(
                actor=self.actor,
                observations=data["next"]["observations"],
                num_random=0,
                num_policy=num_act,
                action_scale=self.actor.action_scale,
                action_bias=self.actor.action_bias,
                action_dim=action_dim,
            )
            ood_action_chunks.append(next_policy_actions)
            ood_counts.append(num_act)

        # =============================================================
        # Step 3: Compute Q-values for each OOD action source
        # =============================================================
        # Each chunk is [batch * N_k, action_dim].  We evaluate Q for each
        # chunk separately so that obs is repeated by the correct N_k, then
        # concatenate the Q-values.
        q_ood_chunks: list[torch.Tensor] = []
        for chunk, n_k in zip(ood_action_chunks, ood_counts):
            # compute_q_values_for_actions returns [num_q, batch, n_k]
            q_chunk = compute_q_values_for_actions(
                qnet=self.qnet,
                critic_observations=critic_observations,
                actions=chunk,
                num_samples=n_k,
            )
            q_ood_chunks.append(q_chunk)

        # Concatenate along the samples dim: [num_q, batch, total_ood]
        q_ood = torch.cat(q_ood_chunks, dim=-1)
        total_ood = sum(ood_counts)
        num_q = q_ood.shape[0]
        assert q_ood.shape == (num_q, batch_size, total_ood), (
            f"q_ood shape {q_ood.shape} != expected ({num_q}, {batch_size}, {total_ood})"
        )

        # =============================================================
        # Step 4: Compute Q-values for dataset actions
        # =============================================================
        # q_outputs is [num_q, batch, num_atoms] — raw logits from parent.
        q_dataset_probs = F.softmax(q_outputs, dim=-1)
        q_dataset = self.qnet.get_value(q_dataset_probs)  # [num_q, batch]
        assert q_dataset.shape == (num_q, batch_size), (
            f"q_dataset shape {q_dataset.shape} != expected ({num_q}, {batch_size})"
        )

        # =============================================================
        # Step 5: Conservative penalty = logsumexp(Q_ood/τ)·τ − Q_dataset
        # =============================================================
        raw_penalty = compute_conservative_penalty(
            q_ood=q_ood,
            q_dataset=q_dataset,
            temperature=codac.conservative_temp,
        )

        # Scale by CODAC-specific coefficient.
        # (Parent will additionally multiply by config.conservative_weight.)
        conservative_loss = codac.conservative_coef * raw_penalty

        # =============================================================
        # Step 6: Diagnostic metrics (optional)
        # =============================================================
        if codac.log_codac_debug_metrics:
            with torch.no_grad():
                q_dataset_mean = q_dataset.mean()
                q_ood_mean = q_ood.mean()

                # Per-source diagnostics
                offset = 0
                diag: dict[str, torch.Tensor] = {
                    "codac/conservative_penalty": raw_penalty.detach(),
                    "codac/conservative_loss": conservative_loss.detach(),
                    "codac/q_dataset_mean": q_dataset_mean,
                }
                if use_random:
                    q_rand_slice = q_ood[:, :, offset : offset + num_act]
                    diag["codac/q_ood_random_mean"] = q_rand_slice.mean()
                    offset += num_act
                if use_policy:
                    q_pol_slice = q_ood[:, :, offset : offset + num_act]
                    diag["codac/q_ood_policy_mean"] = q_pol_slice.mean()
                    offset += num_act
                if use_next_policy:
                    q_next_slice = q_ood[:, :, offset : offset + num_act]
                    diag["codac/q_ood_next_policy_mean"] = q_next_slice.mean()
                    offset += num_act

                diag["codac/q_ood_mean"] = q_ood_mean
                diag["codac/conservative_gap"] = q_ood_mean - q_dataset_mean

                self.training_metrics.add(diag)

        return conservative_loss

    def _compute_conservative_actor_loss(
        self,
        data: TensorDict,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Conservative actor loss (CODAC).

        Default CODAC uses the standard SAC actor objective — the
        conservative critic already pushes the policy toward in-distribution
        actions.

        When ``codac_config.actor_bc_coef > 0``, adds a behavior-cloning
        MSE penalty: ``bc_coef * MSE(policy_action, dataset_action)``.

        Returns 0.0 when ``actor_bc_coef == 0``.
        """
        # TODO: If codac.actor_bc_coef > 0, compute BC regularization:
        #   bc_loss = codac.actor_bc_coef * F.mse_loss(actions, data["actions"])
        return torch.tensor(0.0, device=self.device)
