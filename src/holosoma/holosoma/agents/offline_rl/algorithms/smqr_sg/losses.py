"""SMQR-SG conservative penalties (Step 4-C extraction).

Bit-exact lift of the inline ``sg_weighted_lse`` and ``sg_blend``
branches at ``offline_cql_agent.py:2340-2465``.

Formulas
--------

**sg_weighted_lse** (agent L2340-2356)::

    log_g_sg              = log(g.detach().clamp_min(eps))   # [num_q,B,K]
    weighted_logits_pre   = Q_cat_raw + log_g_sg - log_p_cat
    weighted_logits       = weighted_logits_pre.clamp(-q_clip, +q_clip)
    cql_logsumexp         = logsumexp(weighted_logits, dim=-1) - log(N_total)
    per_state_penalty     = cql_logsumexp - q_data

**sg_blend** (agent L2357-2370 ∪ L2400-2465)::

    log_g_sg              = log(g.detach().clamp_min(eps))
    preclip_qg            = Q_cat_raw * g       - log_p_cat   # uses g (NOT detached)
    preclip_sg            = Q_cat_raw + log_g_sg - log_p_cat
    weighted_logits_qg    = preclip_qg.clamp(-q_clip, +q_clip)
    weighted_logits_sg    = preclip_sg.clamp(-q_clip, +q_clip)
    lse_qg                = logsumexp(weighted_logits_qg, dim=-1) - log(N_total)
    lse_sg                = logsumexp(weighted_logits_sg, dim=-1) - log(N_total)
    P_qg                  = lse_qg - q_data
    P_sg                  = lse_sg - q_data
    per_state_penalty     = (1 - λ) · P_qg + λ · P_sg

Key invariants preserved verbatim (any deviation = FAIL):

* ``g.detach()`` inside the ``log`` (gate gradient must NOT flow to
  Q through the ``log_g_sg`` term — only through the bare ``Q`` term
  in sg_weighted_lse, and through ``Q*g`` in the qg side of sg_blend).
* ``clamp_min(eps)`` applied to the detached gate BEFORE ``log``
  (not ``g + eps`` — these differ near g ≈ 0).
* Clamp applied to preclip BEFORE ``logsumexp``.
* ``- math.log(N_total)`` applied AFTER ``logsumexp``.
* ``per_state_penalty = cql_logsumexp − q_data`` (subtraction, not
  ratio).
* In ``sg_blend``: both sides are clamped/logsumexp'd/q_data-subtracted
  **independently** before the convex combine — blending happens at
  the per-state-penalty stage, NOT at the logits stage.
"""

from __future__ import annotations

import math
from typing import TypedDict, Union

import torch


# ────────────────────────────────────────────────────────────────────
# Return types
# ────────────────────────────────────────────────────────────────────
class SMQRSGWeightedLSEOutputs(TypedDict):
    log_g_sg: torch.Tensor              # [num_q, B, K]   detached
    weighted_logits_preclip: torch.Tensor  # [num_q, B, K]
    weighted_logits: torch.Tensor       # [num_q, B, K]
    cql_logsumexp: torch.Tensor         # [num_q, B]
    per_state_penalty: torch.Tensor     # [num_q, B]
    n_total: int


class SMQRSGBlendOutputs(TypedDict):
    log_g_sg: torch.Tensor              # [num_q, B, K]   detached
    weighted_logits_preclip_qg: torch.Tensor  # Q*g − log_p
    weighted_logits_preclip_sg: torch.Tensor  # Q + log_g_sg − log_p
    weighted_logits_qg: torch.Tensor          # clamped
    weighted_logits_sg: torch.Tensor          # clamped
    cql_logsumexp_qg: torch.Tensor      # [num_q, B]
    cql_logsumexp_sg: torch.Tensor      # [num_q, B]
    per_state_penalty_qg: torch.Tensor  # [num_q, B]
    per_state_penalty_sg: torch.Tensor  # [num_q, B]
    per_state_penalty: torch.Tensor     # blended  (1-λ)·qg + λ·sg
    lambda_active: float
    n_total: int


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
def _compute_log_g_sg(g: torch.Tensor, eps: float) -> torch.Tensor:
    """``log(g.detach().clamp_min(eps))`` — mirrors agent L2348/L2371."""
    return torch.log(g.detach().clamp_min(eps))


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────
def compute_sg_weighted_lse_penalty(
    q_cat_raw: torch.Tensor,
    log_p_cat: torch.Tensor,
    q_data: torch.Tensor,
    g: torch.Tensor,
    *,
    eps: float,
    q_clip: float,
    n_total: int,
) -> SMQRSGWeightedLSEOutputs:
    """SMQR-SG ``sg_weighted_lse`` conservative penalty.

    Parameters
    ----------
    q_cat_raw : ``[num_q, B, K]``
        Raw critic Q on concat(random, policy) actions.
    log_p_cat : ``[1, B, K]`` (broadcastable)
        Concatenated detached importance-sampling log-densities.
    q_data    : ``[num_q, B]``
        Critic Q on the dataset action.
    g         : ``[num_q, B, K]``
        Sigmoid gate from anchor τ.  This function detaches it
        internally for the ``log_g_sg`` term — the caller may pass
        a gradient-bearing tensor without affecting bit-exactness.
    eps       : float
        ``smqr_sg_eps``; lower bound for ``log(g)`` stability.
    q_clip    : float
        Symmetric clamp for the pre-clip weighted logits.
    n_total   : int
        ``num_random + num_policy``.

    Returns
    -------
    SMQRSGWeightedLSEOutputs
    """
    log_g_sg = _compute_log_g_sg(g, eps)
    weighted_logits_preclip = q_cat_raw + log_g_sg - log_p_cat
    weighted_logits = weighted_logits_preclip.clamp(-q_clip, q_clip)
    cql_logsumexp = torch.logsumexp(weighted_logits, dim=-1) - math.log(n_total)
    per_state_penalty = cql_logsumexp - q_data
    return {
        "log_g_sg": log_g_sg,
        "weighted_logits_preclip": weighted_logits_preclip,
        "weighted_logits": weighted_logits,
        "cql_logsumexp": cql_logsumexp,
        "per_state_penalty": per_state_penalty,
        "n_total": n_total,
    }


def compute_sg_blend_penalty(
    q_cat_raw: torch.Tensor,
    log_p_cat: torch.Tensor,
    q_data: torch.Tensor,
    g: torch.Tensor,
    *,
    lambda_active: Union[float, torch.Tensor],
    eps: float,
    q_clip: float,
    n_total: int,
) -> SMQRSGBlendOutputs:
    """SMQR-SG ``sg_blend`` LOSS-level blend of qg and sg per-state penalties.

    Bit-exact mirror of agent L2357-2370 ∪ L2400-2465.  Both sides
    are clamped/logsumexp'd/q_data-subtracted independently; the
    convex combine happens at the per-state-penalty stage.

    Parameters
    ----------
    lambda_active
        Pre-resolved λ(t) from
        :func:`compute_smqr_blend_lambda`.  Python ``float`` for
        S6 (``fixed`` schedule); a 0-D tensor is accepted for
        future-proofing but currently the agent always passes a
        ``float`` (matches inline ``float(_lam)`` cast at L2461).
    eps, q_clip, n_total
        See :func:`compute_sg_weighted_lse_penalty`.

    Returns
    -------
    SMQRSGBlendOutputs
    """
    log_g_sg = _compute_log_g_sg(g, eps)

    # ── qg side (uses non-detached g, mirrors agent L2374) ─────
    weighted_logits_preclip_qg = q_cat_raw * g - log_p_cat
    # ── sg side (uses detached log(g)+eps, mirrors agent L2377) ─
    weighted_logits_preclip_sg = q_cat_raw + log_g_sg - log_p_cat

    # ── Independent clamp + logsumexp + q_data subtraction ─────
    weighted_logits_qg = weighted_logits_preclip_qg.clamp(-q_clip, q_clip)
    weighted_logits_sg = weighted_logits_preclip_sg.clamp(-q_clip, q_clip)

    log_n = math.log(n_total)
    cql_logsumexp_qg = torch.logsumexp(weighted_logits_qg, dim=-1) - log_n
    cql_logsumexp_sg = torch.logsumexp(weighted_logits_sg, dim=-1) - log_n

    per_state_penalty_qg = cql_logsumexp_qg - q_data
    per_state_penalty_sg = cql_logsumexp_sg - q_data

    # ── LOSS-level convex blend (agent L2462-2464) ─────────────
    lam_f = float(lambda_active) if not isinstance(lambda_active, torch.Tensor) \
        else float(lambda_active.item())
    per_state_penalty = (
        (1.0 - lam_f) * per_state_penalty_qg + lam_f * per_state_penalty_sg
    )

    return {
        "log_g_sg": log_g_sg,
        "weighted_logits_preclip_qg": weighted_logits_preclip_qg,
        "weighted_logits_preclip_sg": weighted_logits_preclip_sg,
        "weighted_logits_qg": weighted_logits_qg,
        "weighted_logits_sg": weighted_logits_sg,
        "cql_logsumexp_qg": cql_logsumexp_qg,
        "cql_logsumexp_sg": cql_logsumexp_sg,
        "per_state_penalty_qg": per_state_penalty_qg,
        "per_state_penalty_sg": per_state_penalty_sg,
        "per_state_penalty": per_state_penalty,
        "lambda_active": lam_f,
        "n_total": n_total,
    }


__all__ = [
    "SMQRSGBlendOutputs",
    "SMQRSGWeightedLSEOutputs",
    "compute_sg_blend_penalty",
    "compute_sg_weighted_lse_penalty",
]
