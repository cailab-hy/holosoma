from __future__ import annotations

import torch

from holosoma.agents.aw_cql.aw_cql_agent import AWCQLAgent
from holosoma.agents.lse_aw_cql.lse_aw_cql_agent import LSEAWCQLAgent
from holosoma.agents.os_aw_cql.os_aw_cql_agent import OSAWCQLAgent
from holosoma.config_values.experiment import DEFAULTS


def test_lse_aw_cql_weights_only_logsumexp() -> None:
    weight = torch.tensor([0.5, 2.0])
    q1_lse = torch.tensor([3.0, 5.0])
    q2_lse = torch.tensor([4.0, 6.0])
    q1_data = torch.tensor([1.0, 2.0])
    q2_data = torch.tensor([2.0, 3.0])

    agent = LSEAWCQLAgent.__new__(LSEAWCQLAgent)
    agent._aw_batch_weight = weight

    q1_loss, q2_loss = agent._build_cql_per_sample_losses(q1_lse, q2_lse, q1_data, q2_data)
    q1_loss, q2_loss = agent._transform_cql_per_sample_losses(q1_loss, q2_loss)

    torch.testing.assert_close(q1_loss, weight * q1_lse - q1_data)
    torch.testing.assert_close(q2_loss, weight * q2_lse - q2_data)


def test_aw_weight_placements_are_distinct() -> None:
    weight = torch.tensor([0.5, 2.0])
    lse = torch.tensor([3.0, 5.0])
    q_data = torch.tensor([1.0, 2.0])

    aw_agent = AWCQLAgent.__new__(AWCQLAgent)
    aw_agent._aw_batch_weight = weight
    aw_loss, _ = aw_agent._transform_cql_per_sample_losses(lse - q_data, lse - q_data)

    os_agent = OSAWCQLAgent.__new__(OSAWCQLAgent)
    os_agent._aw_batch_weight = weight
    os_loss, _ = os_agent._build_cql_per_sample_losses(lse, lse, q_data, q_data)

    lse_agent = LSEAWCQLAgent.__new__(LSEAWCQLAgent)
    lse_agent._aw_batch_weight = weight
    lse_loss, _ = lse_agent._build_cql_per_sample_losses(lse, lse, q_data, q_data)

    torch.testing.assert_close(aw_loss, weight * (lse - q_data))
    torch.testing.assert_close(os_loss, lse - weight * q_data)
    torch.testing.assert_close(lse_loss, weight * lse - q_data)
    assert not torch.allclose(lse_loss, aw_loss)
    assert not torch.allclose(lse_loss, os_loss)


def test_lse_aw_cql_experiments_are_registered_and_paired() -> None:
    pairs = (
        ("g1_29dof_wbt_aw_cql", "g1_29dof_wbt_lse_aw_cql"),
        ("g1_29dof_wbt_aw_cql_w_object", "g1_29dof_wbt_lse_aw_cql_w_object"),
        ("g1_29dof_wbt_fall_and_getup_aw_cql", "g1_29dof_wbt_fall_and_getup_lse_aw_cql"),
    )

    for aw_name, lse_name in pairs:
        aw_experiment = DEFAULTS[aw_name]
        lse_experiment = DEFAULTS[lse_name]
        assert lse_experiment.algo._target_ == (
            "holosoma.agents.lse_aw_cql.lse_aw_cql_agent.LSEAWCQLAgent"
        )
        assert lse_experiment.algo.config == aw_experiment.algo.config
