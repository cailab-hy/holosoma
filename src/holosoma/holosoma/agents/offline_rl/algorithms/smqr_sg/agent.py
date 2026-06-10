"""SMQRSGAgent — Step 7-D extraction of the SMQR-SG family.

Standalone agent for the **SMQR Stop-Gradient** family of conservative
penalties.  The code was extracted from the historical monolithic
``OfflineCQLAgent`` source before the ``offline_cql`` package was removed.

Scope (Step 7-D)
----------------
Supports the **anchor-only** stop-gradient family:

* ``smqr_lse_mode = 'sg_weighted_lse'`` — gate enters detached, as
  additive ``log(g.detach().clamp_min(eps))``; softmax is gate-weighted
  over ``Q − log p``.  ``∂lse/∂Q_i = w_i`` (β-independent gradient
  structure; no ``Q·g'`` amplification).
* ``smqr_lse_mode = 'sg_blend'`` — LOSS-level convex blend of
  ``q_times_g`` and ``sg_weighted_lse`` per-state penalties with a
  schedule-resolved λ(t).  S6 golden uses ``schedule='fixed'``,
  ``λ_start=λ_end=0.5``.

Schedules supported (via :func:`compute_smqr_blend_lambda`):
    ``fixed``, ``linear``, ``delayed_linear``, ``piecewise``.

Out-of-scope (rejected at config-validation)
--------------------------------------------
* ``q_times_g`` / ``q_times_detached_g`` → :class:`SMQRAgent`
* ``vanilla_cql``                        → :class:`CQLAgent`
* ``algo_mode='smqr_learned'``           → unsupported in canonical agents
* F1 / G1 / H1 / B2 / P1 / P1b sub-flags → not planned
* Phase-E stabilised anchor objective    → not planned
* Learned-τ residual is available for ablation when
    ``sc_tau_res_scale > 0.0``.

Inheritance
-----------
Subclasses :class:`SMQRAgent` (which itself subclasses
:class:`CQLAgent`).  This grants verbatim reuse of:

* ``setup`` / ``learn`` outer loop                  (from CQLAgent)
* ``_update_actor`` / ``_update_alpha``             (from CQLAgent)
* ``evaluate_policy`` + 5 eval hook delegations     (from CQLAgent
                                                     → eval_utils shim)
* SMQR τ-anchor construction logic                  (mirrored inline
                                                     here; the SG family
                                                     uses the same
                                                     ``q_data.detach().min``
                                                     anchor)

We override only the four algorithm-specific hooks (algo-mode marker,
config validator, conservative penalty, critic update).

Import boundary
---------------
This module **must not** import from ``holosoma.agents.offline_cql``.  Direct
legacy imports are unsupported after Step 14; old target compatibility is
metadata-only in ``target_compat``.  All shared utilities flow through
:mod:`offline_rl.common.*` and the algorithm-specific loss / schedule helpers
under :mod:`offline_rl.algorithms.smqr_sg.*`.
"""

from __future__ import annotations

import math
from typing import Any

from holosoma.agents.offline_rl.algorithms.cql.agent import CQLAgent
from holosoma.agents.offline_rl.algorithms.smqr.agent import SMQRAgent
from holosoma.agents.offline_rl.algorithms.smqr_sg.losses import (
    compute_sg_blend_penalty,
    compute_sg_weighted_lse_penalty,
)
from holosoma.agents.offline_rl.algorithms.smqr_sg.schedules import (
    compute_smqr_blend_lambda,
)
from holosoma.utils.safe_torch_import import F, TensorDict, torch


# ---------------------------------------------------------------------------
# Algo-mode marker
# ---------------------------------------------------------------------------
class _SMQRSGAlgoMode:
    """Marker returned by :meth:`SMQRSGAgent._resolve_algo_mode`.

    The SMQR-SG family is anchor-only (no learned residual; no Phase-E
    stabilised objective).  ``mode == 'smqr_anchor'`` matches the
    legacy ``ResolvedAlgoMode`` mapping for sg_weighted_lse / sg_blend
    runs.
    """

    mode = "smqr_anchor"
    tau_source = "anchor"
    legacy_critic_penalty_mode = "smqr_cont_self"
    tau_res_scale = "config"
    learned_variant = "none"
    explicit = True
    logging_prefix = "smqranc"


# ---------------------------------------------------------------------------
# SMQRSGAgent
# ---------------------------------------------------------------------------
class SMQRSGAgent(SMQRAgent):
    """SMQR Stop-Gradient agent (Step 7-D extraction)."""

    # ------------------------------------------------------------------
    # Setup — extends CQLAgent.setup with sg_blend schedule cache
    # ------------------------------------------------------------------
    def setup(
        self,
        *,
        eval_only: bool = False,
        checkpoint_path: str | None = None,
    ) -> None:
        """Defer to :meth:`CQLAgent.setup`, then cache sg_blend schedule.

        The schedule cache attributes
        (``_smqr_blend_{schedule,lambda_start,lambda_end,
        warmup_steps,ramp_steps,hold_steps}``) mirror the legacy
        ``offline_cql_agent.py:426-430`` attribute set bit-exactly so
        that :meth:`_update_critic` can call
        :func:`compute_smqr_blend_lambda` with the same numeric inputs
        and produce identical λ(t) trajectories.

        For ``smqr_lse_mode != 'sg_blend'`` we still populate the cache
        with the documented defaults (``schedule='fixed'``,
        ``λ_start=λ_end=0.5``) — the attributes are simply never read
        by the critic update under that branch.
        """
        super().setup(
            eval_only=eval_only,
            checkpoint_path=checkpoint_path,
        )
        args = self.config
        self._smqr_blend_schedule = str(
            getattr(args, "smqr_blend_schedule", "fixed")
        ).strip().lower()
        self._smqr_blend_lambda_start = float(
            getattr(args, "smqr_blend_lambda_start", 0.5)
        )
        self._smqr_blend_lambda_end = float(
            getattr(args, "smqr_blend_lambda_end", 0.5)
        )
        self._smqr_blend_warmup_steps = int(
            getattr(args, "smqr_blend_warmup_steps", 0)
        )
        self._smqr_blend_ramp_steps = int(
            getattr(args, "smqr_blend_ramp_steps", 1)
        )
        self._smqr_blend_hold_steps = int(
            getattr(args, "smqr_blend_hold_steps", 0)
        )

    # ------------------------------------------------------------------
    # Algorithm hooks
    # ------------------------------------------------------------------
    def _resolve_algo_mode(self, args: Any) -> _SMQRSGAlgoMode:  # type: ignore[override]
        return _SMQRSGAlgoMode()

    def _validate_algo_config(self, args: Any) -> None:  # type: ignore[override]
        """Enforce the SMQR-SG family invariant.

        Accepts:
            * ``algo_mode``           ∈ {``'smqr_anchor'``, ``'auto'``}
            * ``critic_penalty_mode`` == ``'smqr_cont_self'``
            * ``smqr_anchor_objective`` == ``'vanilla'``
            * ``sc_tau_res_scale``    >= 0.0
            * ``smqr_lse_mode``       ∈ {``'sg_weighted_lse'``, ``'sg_blend'``}

        Plus, for ``sg_blend`` mode:
            * ``smqr_blend_schedule`` ∈ {``'fixed'``, ``'linear'``,
              ``'delayed_linear'``, ``'piecewise'``}
            * ``smqr_blend_lambda_{start,end}`` ∈ [0, 1]
            * ``smqr_blend_warmup_steps`` ≥ 0
            * ``smqr_blend_ramp_steps`` ≥ 1
        """
        # ── 1. algo_mode ──────────────────────────────────────────
        algo_mode = str(getattr(args, "algo_mode", "auto")).strip().lower()
        if algo_mode not in ("smqr_anchor", "auto"):
            raise RuntimeError(
                f"SMQRSGAgent requires algo_mode in {{'smqr_anchor','auto'}}, "
                f"got {algo_mode!r}.  algo_mode='cql' → CQLAgent; "
                "algo_mode='smqr_learned' is not supported in this agent."
            )

        # ── 2. critic_penalty_mode ────────────────────────────────
        cpm = str(getattr(args, "critic_penalty_mode", "vanilla_cql")).strip().lower()
        if cpm != "smqr_cont_self":
            raise RuntimeError(
                f"SMQRSGAgent requires critic_penalty_mode='smqr_cont_self', "
                f"got {cpm!r}.  critic_penalty_mode='vanilla_cql' → CQLAgent; "
                "critic_penalty_mode='sc_cql' is not supported."
            )

        # ── 3. smqr_lse_mode — accept ONLY SG modes ───────────────
        lse_mode = str(
            getattr(args, "smqr_lse_mode", "q_times_g")
        ).strip().lower()
        if lse_mode in ("q_times_g", "q_times_detached_g"):
            raise RuntimeError(
                f"SMQRSGAgent rejects smqr_lse_mode={lse_mode!r}; "
                "use SMQRAgent for the anchor q_times_g family."
            )
        if lse_mode not in ("sg_weighted_lse", "sg_blend"):
            raise RuntimeError(
                f"SMQRSGAgent requires smqr_lse_mode in "
                f"{{'sg_weighted_lse','sg_blend'}}, got {lse_mode!r}."
            )

        # ── 4. Anchor objective invariant + tau residual range check ─
        anchor_obj = str(
            getattr(args, "smqr_anchor_objective", "vanilla")
        ).strip().lower()
        if anchor_obj != "vanilla":
            raise RuntimeError(
                f"SMQRSGAgent requires smqr_anchor_objective='vanilla', "
                f"got {anchor_obj!r}.  Stabilised Phase-E objective is "
                "out of scope for Step 7-D."
            )
        tau_res_scale = float(getattr(args, "sc_tau_res_scale", 0.0))
        if tau_res_scale < 0.0:
            raise RuntimeError(
                f"SMQRSGAgent requires sc_tau_res_scale >= 0.0, "
                f"got {tau_res_scale}."
            )

        # ── 5. F1/G1/H1/B2/Phase-E sub-flags must be off ──────────
        for key in ("smqr_anchor_phase_e_optin", "smqr_f1_random_full_grad"):
            if bool(getattr(args, key, False)):
                raise RuntimeError(
                    f"SMQRSGAgent rejects {key}=True; "
                    "Phase-E / F1 / G1 / H1 / B2 sub-flags are out of "
                    "scope for Step 7-D."
                )
        for key in ("smqr_h1_alpha_floor", "smqr_b2_alpha_floor"):
            if float(getattr(args, key, 0.0)) != 0.0:
                raise RuntimeError(
                    f"SMQRSGAgent rejects {key}!=0.0; H1 / B2 sub-flags "
                    "are out of scope for Step 7-D."
                )

        # ── 6. smqr_sg_eps must be positive ───────────────────────
        sg_eps = float(getattr(args, "smqr_sg_eps", 1e-6))
        if sg_eps <= 0.0:
            raise RuntimeError(
                f"SMQRSGAgent requires smqr_sg_eps > 0, got {sg_eps}."
            )

        # ── 7. sg_blend schedule validation ───────────────────────
        if lse_mode == "sg_blend":
            self._validate_blend_schedule(args)

    @staticmethod
    def _validate_blend_schedule(args: Any) -> None:
        """Mirror of the legacy schedule validation block (L385-426)."""
        schedule = str(
            getattr(args, "smqr_blend_schedule", "fixed")
        ).strip().lower()
        allowed = ("fixed", "linear", "delayed_linear", "piecewise")
        if schedule not in allowed:
            raise RuntimeError(
                f"Unknown smqr_blend_schedule={schedule!r}. "
                f"Allowed: {allowed}."
            )
        bls = float(getattr(args, "smqr_blend_lambda_start", 0.5))
        ble = float(getattr(args, "smqr_blend_lambda_end", 0.5))
        blw = int(getattr(args, "smqr_blend_warmup_steps", 0))
        blr = int(getattr(args, "smqr_blend_ramp_steps", 1))
        blh = int(getattr(args, "smqr_blend_hold_steps", 0))
        if not (0.0 <= bls <= 1.0):
            raise RuntimeError(
                f"smqr_blend_lambda_start={bls} outside [0, 1]."
            )
        if not (0.0 <= ble <= 1.0):
            raise RuntimeError(
                f"smqr_blend_lambda_end={ble} outside [0, 1]."
            )
        if blw < 0:
            raise RuntimeError(
                f"smqr_blend_warmup_steps={blw} must be ≥ 0."
            )
        if blr < 1:
            raise RuntimeError(
                f"smqr_blend_ramp_steps={blr} must be ≥ 1."
            )
        if blh < 0:
            raise RuntimeError(
                f"smqr_blend_hold_steps={blh} must be ≥ 0."
            )

    # ------------------------------------------------------------------
    # Conservative penalty dispatch
    # ------------------------------------------------------------------
    def _compute_conservative_penalty(  # type: ignore[override]
        self,
        *,
        q_cat_raw: "torch.Tensor",
        log_p_cat: "torch.Tensor",
        q_data: "torch.Tensor",
        g: "torch.Tensor",
        lse_mode: str,
        lambda_active: float,
        eps: float,
        q_clip: float,
        n_total: int,
    ) -> dict[str, "torch.Tensor"]:
        """Dispatch to the appropriate SMQR-SG loss helper.

        The helper's return dict is forwarded verbatim — the caller
        consumes ``per_state_penalty`` only.  This split keeps the
        per-mode formula encapsulated in a Step 4-C helper that has
        independent loss-equivalence test coverage.
        """
        if lse_mode == "sg_weighted_lse":
            return compute_sg_weighted_lse_penalty(
                q_cat_raw, log_p_cat, q_data, g,
                eps=eps, q_clip=q_clip, n_total=n_total,
            )
        elif lse_mode == "sg_blend":
            return compute_sg_blend_penalty(
                q_cat_raw, log_p_cat, q_data, g,
                lambda_active=lambda_active,
                eps=eps, q_clip=q_clip, n_total=n_total,
            )
        else:  # unreachable — validated upstream
            raise RuntimeError(f"unknown SMQR-SG lse_mode={lse_mode!r}")

    # ------------------------------------------------------------------
    # Critic update (SMQR-SG: sg_weighted_lse / sg_blend)
    # ------------------------------------------------------------------
    def _update_critic(  # type: ignore[override]
        self, data: "TensorDict"
    ) -> dict[str, "torch.Tensor"]:
        """Critic gradient step: TD loss + SMQR-SG conservative penalty.

        Bit-exact port of the legacy ``_update_critic`` branch entered
        under::

            critic_penalty_mode    == 'smqr_cont_self'
            algo_mode              == 'smqr_anchor'
            smqr_anchor_objective  == 'vanilla'
            sc_tau_res_scale       >= 0.0
            smqr_lse_mode          ∈ {'sg_weighted_lse', 'sg_blend'}

        The shared post-dispatch tail (clamp-min(-10) on ``cql_penalty_per_q``
        → α_cql dispatch → ``cql_loss_scale``) is reused verbatim with
        :meth:`CQLAgent._update_critic` and :meth:`SMQRAgent._update_critic`.
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
            q_data = q_pred_all.float()

            # ── SMQR anchor τ (+ optional learned residual via sc_tau_res_scale) ───
            _tau_res_scale = float(getattr(args, "sc_tau_res_scale", 0.0))
            _tau_anchor = q_data.detach().min(dim=0).values  # [B]
            _tau_raw_residual = qnet.tau_from_processed(critic_obs_processed)
            _tau_residual = _tau_res_scale * torch.tanh(_tau_raw_residual)
            _tau = _tau_anchor + _tau_residual  # [B]
            _tau_b1 = _tau.view(1, B, 1)

            # ── β ────────────────────────────────────────────────
            _beta = max(
                float(getattr(args, "sc_tau_beta", 1.0)),
                float(getattr(args, "sc_tau_eps", 1e-6)),
            )

            # ── Raw Q stack + IS log-densities ────────────────────
            Q_cat_raw = torch.cat([q_rand, q_pi], dim=-1).float()  # [num_q,B,K]
            log_p_rand = rand_log_density.expand(B, num_random)
            log_p_cat = torch.cat([log_p_rand, pi_log_probs], dim=-1)
            log_p_cat = log_p_cat.unsqueeze(0).float()  # [1, B, K]

            # ── Per-critic Δ + sigmoid gate g (NO detach) ─────────
            _g = torch.sigmoid((Q_cat_raw - _tau_b1) / _beta)

            # ── SMQR-SG dispatch ──────────────────────────────────
            _lse_mode = str(
                getattr(args, "smqr_lse_mode", "sg_weighted_lse")
            ).strip().lower()
            _sg_eps = float(getattr(args, "smqr_sg_eps", 1e-6))
            N_total = num_random + num_policy

            _smqr_blend_lambda_active = 0.0
            if _lse_mode == "sg_blend":
                _smqr_blend_lambda_active = compute_smqr_blend_lambda(
                    int(self.global_step),
                    schedule=self._smqr_blend_schedule,
                    lambda_start=self._smqr_blend_lambda_start,
                    lambda_end=self._smqr_blend_lambda_end,
                    warmup_steps=self._smqr_blend_warmup_steps,
                    ramp_steps=self._smqr_blend_ramp_steps,
                )

            _sg_out = self._compute_conservative_penalty(
                q_cat_raw=Q_cat_raw,
                log_p_cat=log_p_cat,
                q_data=q_data,
                g=_g,
                lse_mode=_lse_mode,
                lambda_active=_smqr_blend_lambda_active,
                eps=_sg_eps,
                q_clip=_q_clip,
                n_total=N_total,
            )
            per_state_penalty = _sg_out["per_state_penalty"]

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
            _cql_alpha_cap = float(
                getattr(args, "cql_effective_alpha_cap", 0.0)
            )

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

            # Stage R1 uniform conservative-loss scale (default 1.0)
            _cql_loss_scale = float(getattr(args, "cql_loss_scale", 1.0))
            cql_loss = _cql_loss_scale * cql_loss

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

        # ── Canonical TB scalars ──────────────────────────────────
        with torch.no_grad():
            q_data_mean = q_data.mean()
            q_data_max = q_data.max()
            q_data_min = q_data.min()
            td_target_mean = td_target.mean()
            cql_q_rand_mean = q_rand.mean()
            q_overestimation_gap = q_data_mean - td_target_mean
            _tau_anchor_mean = _tau_anchor.mean()
            _tau_mean = _tau.mean()
            _tau_std = (
                _tau.std() if _tau.numel() >= 2
                else torch.tensor(0.0, device=self.device)
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
            # ── SMQR-shared τ telemetry ─────────────────────────
            "smqr/shared/tau_mean": _tau_mean.detach(),
            "smqr/shared/tau_std": _tau_std.detach(),
            "smqr/shared/tau_anchor_mean": _tau_anchor_mean.detach(),
            "smqr/shared/beta": torch.tensor(_beta, device=self.device),
            "smqr/shared/tau_res_scale": torch.tensor(
                _tau_res_scale, device=self.device
            ),
            # ── SMQR-SG specific telemetry (lambda + lse-mode tag) ──
            "smqr_blend_lambda_active": torch.tensor(
                float(_smqr_blend_lambda_active), device=self.device
            ),
            "smqr/sg/sg_eps": torch.tensor(_sg_eps, device=self.device),
        }


__all__ = ["SMQRSGAgent"]
