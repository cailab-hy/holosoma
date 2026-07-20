"""One-sided Advantage-Weighted Conservative Q-Learning (OS-AW-CQL).

OS-AW-CQL reuses AW-CQL's precomputed per-transition sidecar weight and all
training machinery, but applies the weight only to the dataset-action anchor:

    conservative_i = logsumexp_a Q(s_i, a) - w_i Q(s_i, a_D,i)

The OOD logsumexp term, Bellman loss, actor update, alpha/Lagrange path,
sampling sources, and hyperparameters are unchanged from AW-CQL.
"""

from __future__ import annotations

from holosoma.agents.aw_cql.aw_cql_agent import AWCQLAgent
from holosoma.utils.safe_torch_import import torch


class OSAWCQLAgent(AWCQLAgent):
    """AW-CQL variant that weights only the in-dataset Q anchor."""

    def _build_cql_per_sample_losses(
        self,
        q1_lse: torch.Tensor,
        q2_lse: torch.Tensor,
        q1_data: torch.Tensor,
        q2_data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._aw_batch_weight is not None
        weight = self._aw_batch_weight.to(dtype=q1_lse.dtype)
        return q1_lse - weight * q1_data, q2_lse - weight * q2_data

    def _transform_cql_per_sample_losses(
        self,
        q1_gap: torch.Tensor,
        q2_gap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q1_gap, q2_gap
