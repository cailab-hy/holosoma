from __future__ import annotations

import torch

from holosoma.agents.aw_cql.aw_cql_agent import AWCQLAgent
from holosoma.agents.os_aw_cql.os_aw_cql_agent import OSAWCQLAgent


def test_os_aw_cql_weights_only_dataset_anchor() -> None:
    weight = torch.tensor([0.5, 2.0])
    q1_lse = torch.tensor([3.0, 5.0])
    q2_lse = torch.tensor([4.0, 6.0])
    q1_data = torch.tensor([1.0, 2.0])
    q2_data = torch.tensor([2.0, 3.0])

    agent = OSAWCQLAgent.__new__(OSAWCQLAgent)
    agent._aw_batch_weight = weight

    q1_loss, q2_loss = agent._build_cql_per_sample_losses(q1_lse, q2_lse, q1_data, q2_data)
    q1_loss, q2_loss = agent._transform_cql_per_sample_losses(q1_loss, q2_loss)

    torch.testing.assert_close(q1_loss, q1_lse - weight * q1_data)
    torch.testing.assert_close(q2_loss, q2_lse - weight * q2_data)


def test_os_aw_cql_differs_from_full_bracket_weighting() -> None:
    weight = torch.tensor([0.5, 2.0])
    lse = torch.tensor([3.0, 5.0])
    q_data = torch.tensor([1.0, 2.0])

    aw_agent = AWCQLAgent.__new__(AWCQLAgent)
    aw_agent._aw_batch_weight = weight
    aw_loss, _ = aw_agent._transform_cql_per_sample_losses(lse - q_data, lse - q_data)

    os_agent = OSAWCQLAgent.__new__(OSAWCQLAgent)
    os_agent._aw_batch_weight = weight
    os_loss, _ = os_agent._build_cql_per_sample_losses(lse, lse, q_data, q_data)
    os_loss, _ = os_agent._transform_cql_per_sample_losses(os_loss, os_loss)

    torch.testing.assert_close(aw_loss, weight * (lse - q_data))
    torch.testing.assert_close(os_loss, lse - weight * q_data)
    assert not torch.allclose(aw_loss, os_loss)
