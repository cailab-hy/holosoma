from __future__ import annotations

import torch

from holosoma.agents.aw_cql.aw_cql_agent import AWCQLAgent
from holosoma.agents.b_arm.b_arm_agent import BArmAgent
from holosoma.agents.c_arm.c_arm_agent import CArmAgent
from holosoma.config_values.experiment import DEFAULTS


def test_c_arm_weights_only_logsumexp() -> None:
    weight = torch.tensor([0.5, 2.0])
    q1_lse = torch.tensor([3.0, 5.0])
    q2_lse = torch.tensor([4.0, 6.0])
    q1_data = torch.tensor([1.0, 2.0])
    q2_data = torch.tensor([2.0, 3.0])

    agent = CArmAgent.__new__(CArmAgent)
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

    os_agent = BArmAgent.__new__(BArmAgent)
    os_agent._aw_batch_weight = weight
    os_loss, _ = os_agent._build_cql_per_sample_losses(lse, lse, q_data, q_data)

    lse_agent = CArmAgent.__new__(CArmAgent)
    lse_agent._aw_batch_weight = weight
    lse_loss, _ = lse_agent._build_cql_per_sample_losses(lse, lse, q_data, q_data)

    torch.testing.assert_close(aw_loss, weight * (lse - q_data))
    torch.testing.assert_close(os_loss, lse - weight * q_data)
    torch.testing.assert_close(lse_loss, weight * lse - q_data)
    assert not torch.allclose(lse_loss, aw_loss)
    assert not torch.allclose(lse_loss, os_loss)


def test_c_arm_experiments_are_registered_and_paired() -> None:
    pairs = (
        ("g1_29dof_wbt_aw_cql", "g1_29dof_wbt_c_arm"),
        ("g1_29dof_wbt_aw_cql_w_object", "g1_29dof_wbt_c_arm_w_object"),
        ("g1_29dof_wbt_lafan_dance1_aw_cql", "g1_29dof_wbt_lafan_dance1_c_arm"),
    )

    for aw_name, lse_name in pairs:
        aw_experiment = DEFAULTS[aw_name]
        lse_experiment = DEFAULTS[lse_name]
        assert lse_experiment.algo._target_ == ("holosoma.agents.c_arm.c_arm_agent.CArmAgent")
        assert lse_experiment.algo.config == aw_experiment.algo.config
