"""SMQR-SG (Stop-Gradient) algorithm package — Step 4-C extraction.

Scope (this phase only):
  * ``sg_weighted_lse`` mode  — gate enters as detached additive
    ``log(g.detach().clamp_min(eps))``; softmax is gate-weighted over
    ``Q − log p``.
  * ``sg_blend`` mode         — LOSS-level convex blend of
    ``q_times_g`` and ``sg_weighted_lse`` per-state penalties with a
    schedule-driven λ(t).
  * ``compute_smqr_blend_lambda`` — pure schedule helper for the four
    supported schedules (``fixed``, ``linear``, ``delayed_linear``,
    ``piecewise``).

**Bit-exact** lift of the inline branches at
``offline_cql_agent.py:2340-2465``.  Out-of-scope (untouched by
Step 4-C): CQL vanilla penalty (Step 4-A), SMQR-anchor ``q_times_g``
(Step 4-B), ``q_times_detached_g`` ablation, learned-tau residual,
F1/G1/H1/B2/P1/P1b stabilised G-variants, actor update,
``cql_loss_scale`` application site (kept verbatim in the agent).

Golden reference: ``exp_S6_1m_fixed05_cqls05_bc30_seed1_bs4096``
(``smqr_lse_mode='sg_blend'``, fixed λ=0.5, ``cql_loss_scale=0.5``,
``bc_weight=3.0``, batch=4096).
"""

from holosoma.agents.offline_rl.algorithms.smqr_sg.losses import (
    SMQRSGBlendOutputs,
    SMQRSGWeightedLSEOutputs,
    compute_sg_blend_penalty,
    compute_sg_weighted_lse_penalty,
)
from holosoma.agents.offline_rl.algorithms.smqr_sg.schedules import (
    compute_smqr_blend_lambda,
)

__all__ = [
    "SMQRSGBlendOutputs",
    "SMQRSGWeightedLSEOutputs",
    "compute_sg_blend_penalty",
    "compute_sg_weighted_lse_penalty",
    "compute_smqr_blend_lambda",
]
