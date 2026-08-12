"""LogSumExp-only Advantage-Weighted Conservative Q-Learning (LSE-AW-CQL).

LSE-AW-CQL reuses AW-CQL's fixed per-transition sidecar weight and all
training machinery, but applies the weight only to the OOD logsumexp term:

    conservative_i = w_i logsumexp_a Q(s_i, a) - Q(s_i, a_D,i)

The dataset-action anchor, Bellman loss, actor update, alpha/Lagrange path,
sampling sources, and hyperparameters are unchanged from AW-CQL. This is the
placement counterpart to OS-AW-CQL, which weights only the dataset anchor.
"""

from __future__ import annotations

from holosoma.agents.aw_cql.aw_cql_agent import AWCQLAgent
from holosoma.utils.safe_torch_import import torch


class LSEAWCQLAgent(AWCQLAgent):
    """AW-CQL variant that weights only the OOD logsumexp term."""

    def _build_cql_per_sample_losses(
        self,
        q1_lse: torch.Tensor,
        q2_lse: torch.Tensor,
        q1_data: torch.Tensor,
        q2_data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._aw_batch_weight is not None
        weight = self._aw_batch_weight.to(dtype=q1_lse.dtype)
        return weight * q1_lse - q1_data, weight * q2_lse - q2_data

    def _transform_cql_per_sample_losses(
        self,
        q1_gap: torch.Tensor,
        q2_gap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q1_gap, q2_gap
