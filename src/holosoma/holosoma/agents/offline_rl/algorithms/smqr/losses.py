"""SMQR anchor q_times_g conservative penalty (Step 4-B extraction).

This module is a **bit-exact lift** of the inline SMQR-anchor /
``q_times_g`` penalty currently in
``holosoma.agents.offline_cql.offline_cql_agent._update_critic``.

Pipeline (anchor-only, vanilla, ``smqr_lse_mode='q_times_g'``)::

    q_minus_tau          = Q_cat_raw - tau.view(1, B, 1)
    g                    = sigmoid(q_minus_tau / beta)
    weighted_logits_pre  = Q_cat_raw * g - log_p_cat
    weighted_logits      = weighted_logits_pre.clamp(-q_clip, +q_clip)
    cql_logsumexp        = logsumexp(weighted_logits, dim=-1) - log(N_total)
    per_state_penalty    = cql_logsumexp - q_data
    N_total              = num_random + num_policy

All intermediate tensors are exposed via the return dict so the
equivalence test in ``tests/offline_rl/test_smqr_loss_equivalence.py``
can validate every stage against the verbatim inline reference.

Design rationale (matches Step 4-A pattern):
  * Inputs already computed by the agent (``Q_cat_raw``, ``log_p_cat``,
    ``q_data``, ``tau``, ``beta``, ``q_clip``, ``num_random``,
    ``num_policy``) are passed in by keyword so name-collisions with
    the agent's local ``_g`` / ``_weighted_logits`` are impossible.
  * The helper recomputes ``g`` internally from ``tau`` and ``beta``
    so it can be exercised stand-alone in unit tests.  Because the
    formula is deterministic (``sigmoid((Q - tau)/beta)``) this is
    mathematically and bit-exactly equal to the agent's pre-computed
    ``_g``, even though they live in different scopes.
  * The agent-side post-dispatch shared block at L2348-2358 of
    ``offline_cql_agent.py`` (``.clamp(...)`` + ``logsumexp(...) -
    log(N_total)`` + ``- q_data``) is **not** removed; the agent only
    consumes ``weighted_logits_preclip`` from the helper.  This keeps
    the diff to a single branch and the bit-exactness of the other
    three SMQR sub-modes is preserved by construction.
"""

from __future__ import annotations

import math
from typing import TypedDict

import torch


class SMQRQTimesGOutputs(TypedDict):
    """All tensors produced by :func:`compute_smqr_q_times_g_penalty`."""

    q_minus_tau: torch.Tensor          # [num_q, B, K]
    g: torch.Tensor                    # [num_q, B, K]
    weighted_logits_preclip: torch.Tensor  # [num_q, B, K]
    weighted_logits: torch.Tensor      # [num_q, B, K]
    cql_logsumexp: torch.Tensor        # [num_q, B]
    per_state_penalty: torch.Tensor    # [num_q, B]
    tau: torch.Tensor                  # [num_q or 1, B, 1] broadcast view
    n_total: int


def _broadcast_tau(tau: torch.Tensor) -> torch.Tensor:
    """Reshape ``tau`` to ``[1, B, 1]`` if it arrives as ``[B]``.

    Accepts:
      * ``[B]``         → reshaped to ``[1, B, 1]``
      * ``[1, B, 1]``   → returned as-is
      * any other ndim  → returned as-is (assumed broadcast-compatible)
    """
    if tau.ndim == 1:
        return tau.view(1, -1, 1)
    return tau


def compute_smqr_q_times_g_penalty(
    *,
    q_cat_raw: torch.Tensor,
    log_p_cat: torch.Tensor,
    q_data: torch.Tensor,
    tau: torch.Tensor,
    beta: float,
    q_clip: float,
    num_random: int,
    num_policy: int,
) -> SMQRQTimesGOutputs:
    """Compute the SMQR anchor ``q_times_g`` conservative penalty.

    Parameters
    ----------
    q_cat_raw
        Raw Q-values on concat(random, policy) actions, shape
        ``[num_q, B, K]`` (K = num_random + num_policy).  Float dtype
        matching the critic forward (typically ``torch.float32``).
    log_p_cat
        Concatenated importance-sampling log-densities, shape
        ``[1, B, K]`` (broadcastable to ``q_cat_raw``).  In the agent
        this is built by concatenating the uniform random log-density
        (expanded to the policy-action count) and the squashed-Normal
        log-probability of the policy actions, ``.unsqueeze(0)`` and
        ``.float()`` applied.  Must be detached.
    q_data
        Critic Q on the dataset action, shape ``[num_q, B]``,
        ``float32``.  Subtracted from ``cql_logsumexp`` to form the
        per-state conservative penalty.
    tau
        Anchor tau, shape ``[B]`` or ``[1, B, 1]``.  Detached.
    beta
        Temperature for the sigmoid gate (``sc_tau_beta``, lower-
        bounded by ``sc_tau_eps`` in the caller).  Python ``float``.
    q_clip
        Symmetric clamp applied to the pre-clip weighted logits.
        Python ``float``.
    num_random
        Number of uniform random actions sampled for the CQL IS
        estimator.
    num_policy
        Number of policy actions sampled for the CQL IS estimator.

    Returns
    -------
    SMQRQTimesGOutputs
        Dict with all intermediates.  See class docstring for shapes.

    Notes
    -----
    Bit-exact lift of the inline branch at
    ``offline_cql_agent.py:2280-2286`` plus the shared post-dispatch
    block at ``offline_cql_agent.py:2348-2358``.  No gradient breaks;
    no in-place ops on inputs.
    """
    tau_b = _broadcast_tau(tau)

    # ── Anchor-only gate ────────────────────────────────────────────
    # ``_g = sigmoid((Q - tau)/beta)`` — same formula as agent L2098,
    # but recomputed here for self-containment.  Deterministic →
    # bit-exact match.
    q_minus_tau = q_cat_raw - tau_b
    g = torch.sigmoid(q_minus_tau / beta)

    # ── Pre-clip weighted logits (the *only* line that lives in the
    # ``q_times_g`` branch in the agent) ──────────────────────────
    weighted_logits_preclip = q_cat_raw * g - log_p_cat

    # ── Shared post-dispatch tail (agent L2348-2358) ────────────────
    weighted_logits = weighted_logits_preclip.clamp(-q_clip, q_clip)

    n_total = num_random + num_policy
    cql_logsumexp = torch.logsumexp(weighted_logits, dim=-1) - math.log(n_total)
    per_state_penalty = cql_logsumexp - q_data

    return {
        "q_minus_tau": q_minus_tau,
        "g": g,
        "weighted_logits_preclip": weighted_logits_preclip,
        "weighted_logits": weighted_logits,
        "cql_logsumexp": cql_logsumexp,
        "per_state_penalty": per_state_penalty,
        "tau": tau_b,
        "n_total": n_total,
    }
