"""Vanilla CQL conservative-penalty loss — Step 4-A extraction.

This module factors out the importance-weighted logsumexp penalty used by
vanilla CQL from
``holosoma.agents.offline_cql.offline_cql_agent.OfflineCQLAgent._update_critic``.

The implementation is a **bit-exact** lift of the original inline block
(the "logsumexp with importance weights" section of ``_update_critic``).
Every tensor operation, dtype cast, clamp, ``math.log(N_total)`` adjustment
and broadcasting rule is preserved verbatim so that:

* CQL goldens (e.g. ``logs/hv-g1-manager/exp_80_perf_cql_seed1_bs4096_300k``)
  remain reproducible.
* The SMQR / SMQR-SG branches that overwrite ``cql_logsumexp`` and
  ``per_state_penalty`` later in ``_update_critic`` see exactly the same
  upstream values as before.

No new detach, ``.item()`` graph-break, or dtype change is introduced.
"""

from __future__ import annotations

import math
from typing import TypedDict

import torch

__all__ = [
    "compute_cql_logsumexp_penalty",
    "CQLPenaltyOutputs",
]


class CQLPenaltyOutputs(TypedDict):
    """Outputs of :func:`compute_cql_logsumexp_penalty`.

    Tensor shapes (with ``num_q`` = number of critics, ``B`` = batch size,
    ``N_total`` = ``num_random + num_policy``):

    * ``cql_logsumexp``           — ``[num_q, B]``, float32.
    * ``per_state_penalty``       — ``[num_q, B]``, float32.
    * ``q_data``                  — ``[num_q, B]``, float32. Just
      ``q_pred_all.float()``; returned for callers that need the float32
      view of the dataset Q-values without having to recompute the cast.
    * ``weighted_logits_preclip`` — ``[num_q, B, N_total]``. The raw
      concatenated importance-corrected logits prior to the upcast/clamp.
      Same dtype as the input ``q_rand`` / ``q_pi`` (typically bf16 or
      fp32 depending on AMP context).
    * ``weighted_logits``         — ``[num_q, B, N_total]``, float32.
      Upcast and clamped to ``[-q_clip, +q_clip]``; this is the tensor
      that feeds ``torch.logsumexp``.
    * ``n_total``                 — ``int``. Sample count
      ``num_random + num_policy``; convenient for downstream telemetry.
    """

    cql_logsumexp: torch.Tensor
    per_state_penalty: torch.Tensor
    q_data: torch.Tensor
    weighted_logits_preclip: torch.Tensor
    weighted_logits: torch.Tensor
    n_total: int


def compute_cql_logsumexp_penalty(
    *,
    q_rand: torch.Tensor,
    q_pi: torch.Tensor,
    rand_log_density: torch.Tensor,
    pi_log_probs: torch.Tensor,
    q_pred_all: torch.Tensor,
    num_random: int,
    num_policy: int,
    q_clip: float,
) -> CQLPenaltyOutputs:
    """Compute the vanilla-CQL importance-weighted logsumexp penalty.

    This is a bit-exact extraction of the following inline block from
    ``OfflineCQLAgent._update_critic`` (the "logsumexp with importance
    weights" section, immediately after the random / policy Q-value
    rollouts have been computed)::

        N_total = num_random + num_policy
        q_cat = torch.cat([
            q_rand - rand_log_density,
            q_pi - pi_log_probs.unsqueeze(0),
        ], dim=-1)
        q_cat_f32 = q_cat.float().clamp(-_q_clip, _q_clip)
        cql_logsumexp = (
            torch.logsumexp(q_cat_f32, dim=-1) - math.log(N_total)
        )
        q_data = q_pred_all.float()
        per_state_penalty = cql_logsumexp - q_data

    Parameters
    ----------
    q_rand
        Q-values for the uniformly-random action candidates, shape
        ``[num_q, B, num_random]``. The IS correction
        ``-rand_log_density`` is applied inside this function — pass the
        raw Q-values.
    q_pi
        Q-values for the on-policy action candidates, shape
        ``[num_q, B, num_policy]``. The IS correction
        ``-pi_log_probs.unsqueeze(0)`` is applied inside this function.
    rand_log_density
        Scalar log-density of the uniform sampler. Must already be
        detached; broadcasts against ``q_rand``.
    pi_log_probs
        Per-state per-candidate policy log-probabilities, shape
        ``[B, num_policy]``. Must already be detached (matches the
        legacy code path where ``pi_log_probs`` is detached on
        construction).
    q_pred_all
        Dataset Q-predictions, shape ``[num_q, B]``. Cast to float32
        internally to match ``cql_logsumexp``.
    num_random
        Number of uniform-random action samples (``cql_num_random_actions``).
    num_policy
        Number of on-policy action samples (``cql_num_policy_actions``).
    q_clip
        Symmetric clamp applied to ``q_cat`` after the float32 upcast
        and before the logsumexp. Matches the legacy ``args.q_clip``
        default of ``1e4``.

    Returns
    -------
    CQLPenaltyOutputs
        See :class:`CQLPenaltyOutputs` for the field semantics.
    """
    # NOTE: `N_total` mirrors the legacy variable name. Kept as a Python
    # int to match `math.log(N_total)` exactly (the legacy code uses the
    # Python-`math` log, not a tensor op).
    N_total = num_random + num_policy

    # Concatenate IS-corrected candidate Q-values along the K axis.
    # Shape: [num_q, B, N_total].  rand_log_density is a scalar so the
    # broadcast across [num_q, B, num_random] matches the legacy code
    # exactly; pi_log_probs.unsqueeze(0) gives [1, B, num_policy] which
    # broadcasts against [num_q, B, num_policy].
    q_cat = torch.cat(
        [
            q_rand - rand_log_density,
            q_pi - pi_log_probs.unsqueeze(0),
        ],
        dim=-1,
    )

    # ⚡ STABILITY (P3): upcast to float32 and clamp before logsumexp.
    # Identical to the legacy implementation.
    q_cat_f32 = q_cat.float().clamp(-q_clip, q_clip)

    # logsumexp over action samples, then subtract log(N_total) to
    # normalise: log(1/N · Σ exp(Q - log_density)).
    cql_logsumexp = torch.logsumexp(q_cat_f32, dim=-1) - math.log(N_total)

    # Dataset Q-values upcast to float32 to match cql_logsumexp.
    q_data = q_pred_all.float()

    # CQL penalty per Q-network: E_s[logsumexp] - E_{s,a~D}[Q].
    per_state_penalty = cql_logsumexp - q_data

    return CQLPenaltyOutputs(
        cql_logsumexp=cql_logsumexp,
        per_state_penalty=per_state_penalty,
        q_data=q_data,
        weighted_logits_preclip=q_cat,
        weighted_logits=q_cat_f32,
        n_total=N_total,
    )
