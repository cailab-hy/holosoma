"""SMQR (Soft-Max Q-Regularization) algorithm package.

Step 4-B scope: extracts ONLY the **anchor-only ``q_times_g`` variant** of
the SMQR conservative penalty from
``holosoma.agents.offline_cql.offline_cql_agent._update_critic``.

Out-of-scope (DO NOT touch in this step):
  * ``q_times_detached_g``
  * ``sg_weighted_lse``
  * ``sg_blend`` (loss-level blend schedule)
  * Stabilised G-variants (F1 / G1 / H1 / B2 / P1 / P1b)
  * Learned-tau (sc_tau_res_scale > 0) residual head
  * Actor update (SMQR-actor cache / log_g_stab cache)

The extraction is a **bit-exact lift** of the inline formulae the agent
runs when

  critic_penalty_mode  == 'smqr_cont_self'
  algo_mode            == 'smqr_anchor'
  smqr_anchor_objective == 'vanilla'
  sc_tau_res_scale     == 0.0
  smqr_lse_mode        == 'q_times_g'

i.e. the configuration pinned in the Step 0 golden manifest under
``golden/exp_81_perf_smqr_qtimesg_seed1_bs4096_300k``.
"""

from holosoma.agents.offline_rl.algorithms.smqr.losses import (
    SMQRQTimesGOutputs,
    compute_smqr_q_times_g_penalty,
)

# NOTE: ``SMQRAgent`` is intentionally NOT re-exported from this
# package ``__init__`` to avoid a circular import.  The legacy
# :mod:`holosoma.agents.offline_cql.offline_cql_agent` imports
# ``algorithms.smqr.losses`` at module load, which triggers this
# package ``__init__`` — re-exporting ``SMQRAgent`` here would create a
# package-initialisation cycle while the legacy module is still loading.
# Import SMQRAgent directly from
# :mod:`holosoma.agents.offline_rl.algorithms.smqr.agent` instead.

__all__ = [
    "SMQRQTimesGOutputs",
    "compute_smqr_q_times_g_penalty",
]
