from __future__ import annotations

import torch

from holosoma.agents.aw_cql.aw_cql_agent import AWCQLAgent
from holosoma.agents.cql.cql_agent import CQLAgent
from holosoma.agents.odpr_cql.odpr_cql_agent import ODPRCQLAgent
from holosoma.config_values.algo import DEFAULTS as ALGO_DEFAULTS
from holosoma.config_values.experiment import DEFAULTS


def test_odpr_cql_is_independent_of_aw_cql() -> None:
    assert issubclass(ODPRCQLAgent, CQLAgent)
    assert not issubclass(ODPRCQLAgent, AWCQLAgent)


def test_odpr_cql_weights_the_full_bracket() -> None:
    weight = torch.tensor([0.5, 2.0])
    lse = torch.tensor([3.0, 5.0])
    q_data = torch.tensor([1.0, 2.0])

    agent = ODPRCQLAgent.__new__(ODPRCQLAgent)
    agent._odpr_batch_weight = weight
    loss1, loss2 = agent._transform_cql_per_sample_losses(lse - q_data, lse - q_data)
    torch.testing.assert_close(loss1, weight * (lse - q_data))
    torch.testing.assert_close(loss2, weight * (lse - q_data))

    # identical placement to AW-CQL: same weights give the same bracket
    aw_agent = AWCQLAgent.__new__(AWCQLAgent)
    aw_agent._aw_batch_weight = weight
    aw_loss, _ = aw_agent._transform_cql_per_sample_losses(lse - q_data, lse - q_data)
    torch.testing.assert_close(loss1, aw_loss)


def test_odpr_cql_registered_and_paired_with_aw_cql() -> None:
    target = "holosoma.agents.odpr_cql.odpr_cql_agent.ODPRCQLAgent"
    assert ALGO_DEFAULTS["odpr_cql"]._target_ == target
    odpr = DEFAULTS["g1_29dof_wbt_odpr_cql"]
    aw = DEFAULTS["g1_29dof_wbt_aw_cql"]
    assert odpr.algo._target_ == target
    assert odpr.algo.config.odpr_weights_path == ""
    odpr_cfg = {k: v for k, v in odpr.algo.config.__dict__.items() if k != "odpr_weights_path"}
    aw_cfg = {k: v for k, v in aw.algo.config.__dict__.items() if k != "aw_weights_path"}
    assert odpr_cfg == aw_cfg
    assert odpr.command == aw.command and odpr.reward == aw.reward and odpr.termination == aw.termination
