from __future__ import annotations

import torch

from holosoma.agents.aw_cql.aw_cql_agent import AWCQLAgent
from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.cql.cql_agent import CQLAgent
from holosoma.agents.dw_cql.dw_cql_agent import DWCQLAgent
from holosoma.agents.dw_cql.dw_cql_agent import (
    transition_weighted_mean,
    unweighted_entropy_alpha_loss,
    weighted_twin_bellman_loss,
)
from holosoma.config_values.experiment import DEFAULTS


def test_dw_cql_is_independent_from_cql_and_aw_agents() -> None:
    assert DWCQLAgent.__bases__ == (BaseAlgo,)
    assert not issubclass(DWCQLAgent, CQLAgent)
    assert not issubclass(DWCQLAgent, AWCQLAgent)


def test_dw_cql_weights_complete_conservative_bracket() -> None:
    weight = torch.tensor([0.5, 2.0])
    gap1 = torch.tensor([3.0, -1.0])
    gap2 = torch.tensor([2.0, 4.0])
    transformed1 = transition_weighted_mean(gap1, weight)
    transformed2 = transition_weighted_mean(gap2, weight)

    torch.testing.assert_close(transformed1, (weight * gap1).mean())
    torch.testing.assert_close(transformed2, (weight * gap2).mean())


def test_dw_cql_weights_td_residual_without_changing_target() -> None:
    weight = torch.tensor([0.5, 2.0])
    q1 = torch.tensor([1.0, 4.0])
    q2 = torch.tensor([3.0, 0.0])
    q_target = torch.tensor([2.0, 2.0])
    loss = weighted_twin_bellman_loss(q1, q2, q_target, weight, "mse", 1.0)
    expected = (weight * ((q1 - q_target).square() + (q2 - q_target).square())).mean()

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(q_target, torch.tensor([2.0, 2.0]))


def test_dw_cql_weights_complete_actor_objective() -> None:
    weight = torch.tensor([0.5, 2.0])
    actor_objective = torch.tensor([-3.0, 4.0])
    transformed = transition_weighted_mean(actor_objective, weight)

    torch.testing.assert_close(transformed, (weight * actor_objective).mean())


def test_dw_cql_entropy_alpha_update_remains_unweighted() -> None:
    log_alpha = torch.tensor([0.0], requires_grad=True)
    log_probs = torch.tensor([-3.0, -1.0])
    target_entropy = -2.0
    sidecar_weight = torch.tensor([0.1, 4.0])

    actual = unweighted_entropy_alpha_loss(log_alpha, log_probs, target_entropy)
    expected = (-log_alpha.exp() * (log_probs + target_entropy)).mean()
    incorrectly_weighted = (
        -sidecar_weight * log_alpha.exp() * (log_probs + target_entropy)
    ).mean()

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, incorrectly_weighted)


def test_dw_cql_motion_tracking_is_paired_with_aw_cql() -> None:
    aw = DEFAULTS["g1_29dof_wbt_aw_cql"]
    dw = DEFAULTS["g1_29dof_wbt_dw_cql"]

    assert dw.algo._target_ == "holosoma.agents.dw_cql.dw_cql_agent.DWCQLAgent"
    assert dw.algo.config == type(dw.algo.config)(**vars(aw.algo.config))
    assert dw.command == aw.command
    assert dw.termination == aw.termination
    assert dw.reward == aw.reward
    assert dw.algo.config.use_autotune is True


def test_dw_cql_getup_is_paired_with_aw_cql() -> None:
    aw = DEFAULTS["g1_29dof_wbt_fall_and_getup_aw_cql"]
    dw = DEFAULTS["g1_29dof_wbt_fall_and_getup_dw_cql"]

    assert dw.algo._target_ == "holosoma.agents.dw_cql.dw_cql_agent.DWCQLAgent"
    assert dw.algo.config.offline_dataset_path == aw.algo.config.offline_dataset_path
    assert dw.algo.config.aw_weights_path == aw.algo.config.aw_weights_path
    assert dw.algo.config.batch_size == aw.algo.config.batch_size
    assert dw.algo.config.num_updates == aw.algo.config.num_updates
    assert dw.algo.config.actor_learning_rate == aw.algo.config.actor_learning_rate
    assert dw.algo.config.critic_learning_rate == aw.algo.config.critic_learning_rate
    assert dw.command == aw.command
    assert dw.termination == aw.termination
    assert dw.reward == aw.reward
    assert dw.algo.config.use_autotune is True
