"""SMQRAgent — Step 7-C extraction of the SMQR anchor ``q_times_g`` path.

Standalone agent for the **SMQR continuous-action anchor-only ``q_times_g``**
variant of Conservative Q-Learning.  The code was extracted from the
historical monolithic ``OfflineCQLAgent`` source before the ``offline_cql``
package was removed.

Design contract (Step 7-C)
--------------------------
This class inherits from
:class:`holosoma.agents.offline_rl.algorithms.cql.agent.CQLAgent` and
overrides the four algorithm-specific hooks:

* :meth:`_resolve_algo_mode`        — emits the SMQR-anchor marker
* :meth:`_validate_algo_config`     — accepts the anchor-only invariant
* :meth:`_compute_conservative_penalty` — calls the Step 4-B helper
* :meth:`_update_critic`            — verbatim port of the anchor
  ``q_times_g`` branch of the legacy ``_update_critic``

All other methods (``setup`` / ``learn`` / ``_update_actor`` /
``_update_alpha`` / ``evaluate_policy`` / hook delegations) are inherited
unchanged from :class:`CQLAgent`.  The actor / alpha / eval paths are
algorithm-neutral in offline CQL family and are reused verbatim.

Anchor-only invariant (rejected at config-validation time)
----------------------------------------------------------
* ``algo_mode``               ∈ {``'smqr_anchor'``, ``'auto'``}    (no learned-τ)
* ``smqr_anchor_objective``   == ``'vanilla'``                     (no Phase-E stabilised)
* ``sc_tau_res_scale``        == 0.0                               (no learned residual)
* ``critic_penalty_mode``     == ``'smqr_cont_self'``
* ``smqr_lse_mode``           ∈ {``'q_times_g'``, ``'q_times_detached_g'``}
  (SG modes — ``sg_weighted_lse`` / ``sg_blend`` — are routed to
  ``SMQRSGAgent`` in Step 7-D and rejected here.)
* All F1/G1/H1/B2/V1/Phase-E sub-flags must be **off**.

Out-of-scope (deferred or rejected)
-----------------------------------
* SMQR-SG (``sg_weighted_lse`` / ``sg_blend``)               → Step 7-D
* SMQR learned-τ (``algo_mode='smqr_learned'``)              → later
* SC-CQL (``critic_penalty_mode='sc_cql'``)                  → not planned

Import boundary
---------------
This module **must not** import from ``holosoma.agents.offline_cql``.  Direct
legacy imports are unsupported after Step 14; old target compatibility is
metadata-only in ``target_compat``.  All shared utilities are sourced through
:mod:`holosoma.agents.offline_rl.common.*` and the SMQR loss helper at
:mod:`holosoma.agents.offline_rl.algorithms.smqr.losses`.  The base
:class:`CQLAgent` (which is itself import-boundary clean) provides the
algorithm-neutral training scaffolding.
"""

from __future__ import annotations

import math
from typing import Any

from holosoma.agents.offline_rl.algorithms.cql.agent import CQLAgent
from holosoma.agents.offline_rl.algorithms.smqr.losses import (
    compute_smqr_q_times_g_penalty,
)
from holosoma.utils.safe_torch_import import F, TensorDict, torch


# ---------------------------------------------------------------------------
# Algo-mode marker (SMQR-anchor specific)
# ---------------------------------------------------------------------------
class _SMQRAlgoMode:
    """Marker returned by :meth:`SMQRAgent._resolve_algo_mode`.

    Mirrors the field set of
    :class:`holosoma.agents.offline_rl.common.algo_mode.ResolvedAlgoMode` for
    the anchor-only ``q_times_g`` branch (no learned residual, no
    Phase-E stabilised objective).
    """

    mode = "smqr_anchor"
    tau_source = "anchor"
    legacy_critic_penalty_mode = "smqr_cont_self"
    tau_res_scale = 0.0
    learned_variant = "none"
    explicit = True
    logging_prefix = "smqranc"


# ---------------------------------------------------------------------------
# SMQRAgent
# ---------------------------------------------------------------------------
class SMQRAgent(CQLAgent):
    """SMQR anchor ``q_times_g`` agent (Step 7-C extraction)."""

    # ------------------------------------------------------------------
    # Algorithm hooks
    # ------------------------------------------------------------------
    def _resolve_algo_mode(self, args: Any) -> _SMQRAlgoMode:  # type: ignore[override]
        return _SMQRAlgoMode()

    def _validate_algo_config(self, args: Any) -> None:  # type: ignore[override]
        """Enforce the SMQR anchor ``q_times_g`` invariant.

        Accepts:
            * ``algo_mode``           ∈ {``'smqr_anchor'``, ``'auto'``}
            * ``critic_penalty_mode`` == ``'smqr_cont_self'``
            * ``smqr_anchor_objective`` == ``'vanilla'``
            * ``sc_tau_res_scale``    == 0.0
            * ``smqr_lse_mode``       ∈ {``'q_times_g'``, ``'q_times_detached_g'``}

        Rejects every other knob configuration with a precise message
        pointing to the correct sibling agent (``CQLAgent`` for
        vanilla CQL; ``SMQRSGAgent`` — Step 7-D — for SG modes).
        """
        # ── 1. algo_mode ──────────────────────────────────────────
        algo_mode = str(getattr(args, "algo_mode", "auto")).strip().lower()
        if algo_mode not in ("smqr_anchor", "auto"):
            raise RuntimeError(
                f"SMQRAgent requires algo_mode in {{'smqr_anchor','auto'}}, "
                f"got {algo_mode!r}.  algo_mode='cql' → CQLAgent; "
                f"algo_mode='smqr_learned' is not supported in this agent."
            )

        # ── 2. critic_penalty_mode ────────────────────────────────
        cpm = str(getattr(args, "critic_penalty_mode", "vanilla_cql")).strip().lower()
        if cpm != "smqr_cont_self":
            raise RuntimeError(
                f"SMQRAgent requires critic_penalty_mode='smqr_cont_self', "
                f"got {cpm!r}.  critic_penalty_mode='vanilla_cql' → CQLAgent; "
                f"critic_penalty_mode='sc_cql' is not supported."
            )

        # ── 3. smqr_lse_mode ──────────────────────────────────────
        lse_mode = str(
            getattr(args, "smqr_lse_mode", "q_times_g")
        ).strip().lower()
        if lse_mode in ("sg_weighted_lse", "sg_blend"):
            raise RuntimeError(
                f"SMQRAgent rejects smqr_lse_mode={lse_mode!r}; "
                "use SMQRSGAgent (Step 7-D) for SG-mode runs."
            )
        if lse_mode not in ("q_times_g", "q_times_detached_g"):
            raise RuntimeError(
                f"SMQRAgent requires smqr_lse_mode in "
                f"{{'q_times_g','q_times_detached_g'}}, got {lse_mode!r}."
            )

        # ── 4. anchor-only invariant ──────────────────────────────
        anchor_obj = str(
            getattr(args, "smqr_anchor_objective", "vanilla")
        ).strip().lower()
        if anchor_obj != "vanilla":
            raise RuntimeError(
                f"SMQRAgent requires smqr_anchor_objective='vanilla', "
                f"got {anchor_obj!r}.  Stabilised Phase-E objective is "
                "out of scope for Step 7-C."
            )

        tau_res_scale = float(getattr(args, "sc_tau_res_scale", 0.0))
        if tau_res_scale != 0.0:
            raise RuntimeError(
                f"SMQRAgent requires sc_tau_res_scale=0.0 (anchor-only "
                f"invariant), got {tau_res_scale}.  Learned residual is "
                "out of scope for the canonical SMQRAgent."
            )

        # ── 5. F1/G1/H1/B2/V1/Phase-E sub-flags must be off ───────
        forbidden_truthy = (
            "smqr_anchor_phase_e_optin",
            "smqr_f1_random_full_grad",
        )
        for key in forbidden_truthy:
            if bool(getattr(args, key, False)):
                raise RuntimeError(
                    f"SMQRAgent rejects {key}=True; "
                    "Phase-E / F1 / G1 / H1 / B2 sub-flags are out of "
                    "scope for the anchor q_times_g extraction."
                )
        for key in ("smqr_h1_alpha_floor", "smqr_b2_alpha_floor"):
            if float(getattr(args, key, 0.0)) != 0.0:
                raise RuntimeError(
                    f"SMQRAgent rejects {key}!=0.0; "
                    "H1 / B2 sub-flags are out of scope."
                )

    # ------------------------------------------------------------------
    # Conservative penalty (SMQR q_times_g)
    # ------------------------------------------------------------------
    def _compute_conservative_penalty(
        self,
        *,
        q_cat_raw: "torch.Tensor",
        log_p_cat: "torch.Tensor",
        q_data: "torch.Tensor",
        tau: "torch.Tensor",
        beta: float,
        q_clip: float,
        num_random: int,
        num_policy: int,
    ) -> dict[str, "torch.Tensor"]:
        """Thin wrapper around the Step 4-B SMQR loss helper."""
        return compute_smqr_q_times_g_penalty(
            q_cat_raw=q_cat_raw,
            log_p_cat=log_p_cat,
            q_data=q_data,
            tau=tau,
            beta=beta,
            q_clip=q_clip,
            num_random=num_random,
            num_policy=num_policy,
        )

    # ------------------------------------------------------------------
    # Critic update (SMQR anchor q_times_g)
    # ------------------------------------------------------------------
    def _update_critic(  # type: ignore[override]
        self, data: "TensorDict"
    ) -> dict[str, "torch.Tensor"]:
        """Critic gradient step: TD loss + SMQR anchor q_times_g penalty.

        Bit-exact port of the legacy ``_update_critic`` branch entered
        under::

            critic_penalty_mode    == 'smqr_cont_self'
            algo_mode              == 'smqr_anchor'
            smqr_anchor_objective  == 'vanilla'
            sc_tau_res_scale       == 0.0
            smqr_lse_mode          ∈ {'q_times_g', 'q_times_detached_g'}

        Shared post-dispatch tail (clamp → logsumexp → per_state_penalty
        → ``cql_penalty_per_q.clamp(min=-10)`` → α_cql dispatch →
        ``cql_loss_scale``) is verbatim with
        :meth:`CQLAgent._update_critic`.
        """
        args = self.config
        scaler = self.scaler
        actor = self.actor
        qnet = self.qnet
        qnet_target = self.qnet_target

        with self._maybe_amp():
            observations = data["observations"]
            critic_obs = data["critic_observations"]
            actions = data["actions"]
            next_obs = data["next"]["observations"]
            next_critic_obs = data["next"]["critic_observations"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()
            discount = args.gamma ** data["next"]["effective_n_steps"]

            # ── TD target ─────────────────────────────────────────
            _q_clip: float = getattr(args, "q_clip", 1e4)
            with torch.no_grad():
                next_actions, next_log_probs = actor.get_actions_and_log_probs(
                    next_obs
                )
                target_q = qnet_target.min_q(next_critic_obs, next_actions)
                target_q = target_q.squeeze(-1)
                td_target = rewards.float() + discount.float() * bootstrap.float() * (
                    target_q.float()
                    - self.log_alpha.exp().detach().float()
                    * next_log_probs.float()
                )
                td_target = td_target.clamp(-_q_clip, _q_clip)

            # ── TD loss ───────────────────────────────────────────
            q_pred_all = qnet(critic_obs, actions).squeeze(-1)
            td_loss = 0.5 * F.mse_loss(
                q_pred_all.float(),
                td_target.unsqueeze(0).expand_as(q_pred_all),
            )

            # ── Sample random + policy actions (CQL IS estimator) ─
            B = observations.shape[0]
            n_act = actions.shape[-1]
            num_random = args.cql_num_random_actions
            num_policy = args.cql_num_policy_actions

            critic_obs_processed = qnet.process_obs(critic_obs)

            rand_actions = (
                torch.rand(B, num_random, n_act, device=observations.device) * 2.0
                - 1.0
            ) * actor.action_scale.unsqueeze(0).unsqueeze(0) + actor.action_bias.unsqueeze(
                0
            ).unsqueeze(0)
            rand_log_density = -torch.log(
                2.0 * actor.action_scale + 1e-6
            ).sum().detach()

            q_rand = qnet.q_values_for_actions(
                critic_obs_processed, rand_actions
            ).squeeze(-1)

            obs_repeat = (
                observations.unsqueeze(1)
                .expand(B, num_policy, -1)
                .reshape(B * num_policy, -1)
            )
            pi_actions, pi_log_probs = actor.get_actions_and_log_probs(obs_repeat)
            pi_actions = pi_actions.view(B, num_policy, n_act)
            pi_log_probs = pi_log_probs.view(B, num_policy).detach()

            q_pi = qnet.q_values_for_actions(
                critic_obs_processed, pi_actions
            ).squeeze(-1)

            # ── q_data (raw Q on dataset actions) ─────────────────
            q_data = q_pred_all.float()  # [num_q, B]

            # ── SMQR anchor τ ─────────────────────────────────────
            # τ(s) = Q_data_min(s).detach() + 0.0 · tanh(τ_raw(s))
            #
            # The residual term is multiplied by ``sc_tau_res_scale == 0.0``
            # (anchor-only invariant), so τ_raw does NOT enter the loss
            # nor the autograd graph.  We still call ``tau_from_processed``
            # to keep the tau_head parameters present in the checkpoint
            # state_dict (bit-exact key-stable contract with the legacy
            # SMQR-anchor schema, even though they receive no gradient).
            _tau_res_scale = float(getattr(args, "sc_tau_res_scale", 0.0))
            _tau_anchor = q_data.detach().min(dim=0).values  # [B]
            _tau_raw_residual = qnet.tau_from_processed(critic_obs_processed)
            _tau_residual = _tau_res_scale * torch.tanh(_tau_raw_residual)
            _tau = _tau_anchor + _tau_residual  # [B]

            # ── β ────────────────────────────────────────────────
            _beta = max(
                float(getattr(args, "sc_tau_beta", 1.0)),
                float(getattr(args, "sc_tau_eps", 1e-6)),
            )

            # ── Raw Q stack + IS log-densities (no detach on Q) ───
            Q_cat_raw = torch.cat([q_rand, q_pi], dim=-1).float()  # [num_q, B, K]
            log_p_rand = rand_log_density.expand(B, num_random)  # [B, N_rand]
            log_p_cat = torch.cat([log_p_rand, pi_log_probs], dim=-1)  # [B, K]
            log_p_cat = log_p_cat.unsqueeze(0).float()  # [1, B, K]

            # ── SMQR conservative penalty ─────────────────────────
            _lse_mode = str(
                getattr(args, "smqr_lse_mode", "q_times_g")
            ).strip().lower()
            if _lse_mode == "q_times_g":
                _smqr_out = self._compute_conservative_penalty(
                    q_cat_raw=Q_cat_raw,
                    log_p_cat=log_p_cat,
                    q_data=q_data,
                    tau=_tau,
                    beta=_beta,
                    q_clip=_q_clip,
                    num_random=num_random,
                    num_policy=num_policy,
                )
                cql_logsumexp = _smqr_out["cql_logsumexp"]
                per_state_penalty = _smqr_out["per_state_penalty"]
            else:  # q_times_detached_g  (backward-only ablation)
                _tau_b = _tau.view(1, B, 1)
                _g = torch.sigmoid((Q_cat_raw - _tau_b) / _beta)
                _weighted_logits_preclip = (
                    Q_cat_raw * _g.detach() - log_p_cat
                )
                _weighted_logits = _weighted_logits_preclip.clamp(
                    -_q_clip, _q_clip
                )
                N_total = num_random + num_policy
                cql_logsumexp = (
                    torch.logsumexp(_weighted_logits, dim=-1)
                    - math.log(N_total)
                )
                per_state_penalty = cql_logsumexp - q_data

            # ── Per-Q penalty + clamp (shared with vanilla CQL) ───
            cql_penalty_per_q = per_state_penalty.mean(dim=1)
            cql_penalty = cql_penalty_per_q.clamp(min=-10).sum()
            cql_penalty_raw = cql_penalty_per_q.sum()

            # ── Phase P1b — penalty floor (opt-in; default no-op) ──
            _penalty_floor_optin = bool(
                getattr(args, "cql_penalty_floor_optin", False)
            )
            if _penalty_floor_optin:
                _penalty_for_loss = cql_penalty.clamp(min=0.0)
            else:
                _penalty_for_loss = cql_penalty

            # ── α_cql dispatch (verbatim from CQLAgent) ───────────
            alpha_cql = self.log_alpha_cql.exp().detach().squeeze()
            _cql_td_ratio = getattr(args, "cql_td_ratio", None)
            _cql_alpha_mode = getattr(args, "cql_alpha_mode", "td_relative")
            _cql_alpha_cap = float(getattr(args, "cql_effective_alpha_cap", 0.0))

            if (
                _cql_td_ratio is not None
                and _cql_alpha_mode == "fixed_effective"
            ):
                _fixed_val = getattr(args, "cql_fixed_effective_alpha", 0.015)
                _effective_alpha_pre_cap = _fixed_val
                _effective_alpha = (
                    min(_effective_alpha_pre_cap, _cql_alpha_cap)
                    if _cql_alpha_cap > 0.0
                    else _effective_alpha_pre_cap
                )
                cql_loss = _effective_alpha * _penalty_for_loss
            elif _cql_td_ratio is not None:
                _cql_floor = getattr(args, "cql_alpha_floor", 0.0)
                _cql_raw_alpha = (
                    _cql_td_ratio
                    * td_loss.detach().item()
                    / max(abs(cql_penalty.detach().item()), 1e-8)
                )
                _effective_alpha_pre_cap = max(_cql_raw_alpha, _cql_floor)
                _effective_alpha = (
                    min(_effective_alpha_pre_cap, _cql_alpha_cap)
                    if _cql_alpha_cap > 0.0
                    else _effective_alpha_pre_cap
                )
                cql_loss = _effective_alpha * _penalty_for_loss
            else:
                cql_loss = alpha_cql * cql_penalty
                _effective_alpha = alpha_cql.item()

            # Stage R1 uniform conservative-loss scale (default 1.0 = no-op)
            _cql_loss_scale = float(getattr(args, "cql_loss_scale", 1.0))
            cql_loss = _cql_loss_scale * cql_loss

            # Total critic loss (anchor q_times_g has no V1 shrink term)
            critic_loss = td_loss + cql_loss

        # ── Backward + optimise ───────────────────────────────────
        self.q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(critic_loss).backward()
        scaler.unscale_(self.q_optimizer)
        if args.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(), max_norm=args.max_grad_norm,
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(self.q_optimizer)
        scaler.update()

        # ── Canonical TB scalars (mirrors legacy SMQR-anchor schema) ──
        with torch.no_grad():
            q_data_mean = q_data.mean()
            q_data_max = q_data.max()
            q_data_min = q_data.min()
            td_target_mean = td_target.mean()
            cql_q_rand_mean = q_rand.mean()
            q_overestimation_gap = q_data_mean - td_target_mean

            # SMQR-shared τ telemetry (anchor-only → residual is zero)
            _tau_anchor_mean = _tau_anchor.mean()
            _tau_mean = _tau.mean()
            _tau_std = _tau.std() if _tau.numel() >= 2 else torch.tensor(
                0.0, device=self.device
            )

        return {
            "td_loss": td_loss.detach(),
            "cql_penalty": cql_penalty.detach(),
            "cql_penalty_per_q_mean": cql_penalty_per_q.mean().detach(),
            "cql_loss": cql_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "q_data_mean": q_data_mean.detach(),
            "q_data_max": q_data_max.detach(),
            "q_data_min": q_data_min.detach(),
            "td_target_mean": td_target_mean.detach(),
            "cql_q_rand_mean": cql_q_rand_mean.detach(),
            "q_overestimation_gap": q_overestimation_gap.detach(),
            "cql_alpha": alpha_cql.detach(),
            "cql_effective_alpha": torch.tensor(
                _effective_alpha, device=self.device
            ),
            "cql_loss_scale": torch.tensor(
                _cql_loss_scale, device=self.device
            ),
            "_cql_penalty_raw": cql_penalty_raw.detach(),  # consumed by learn()
            # ── SMQR-shared τ telemetry (anchor-only) ────────────
            "smqr/shared/tau_mean": _tau_mean.detach(),
            "smqr/shared/tau_std": _tau_std.detach(),
            "smqr/shared/tau_anchor_mean": _tau_anchor_mean.detach(),
            "smqr/shared/beta": torch.tensor(_beta, device=self.device),
            "smqr/shared/tau_res_scale": torch.tensor(
                _tau_res_scale, device=self.device
            ),
        }


__all__ = ["SMQRAgent"]
