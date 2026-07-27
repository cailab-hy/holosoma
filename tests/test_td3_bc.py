from __future__ import annotations

import torch

from holosoma.agents.td3.td3_agent import TD3BCAgent
from holosoma.config_values.experiment import DEFAULTS


def test_td3_bc_normalizes_bc_actions_per_dimension():
    agent = TD3BCAgent.__new__(TD3BCAgent)
    agent.actor = type(
        "ActorStub",
        (),
        {
            "action_scale": torch.tensor([2.0, 4.0]),
            "action_bias": torch.tensor([0.0, 0.0]),
        },
    )()

    env_actions = torch.tensor([[1.0, -1.0]])

    torch.testing.assert_close(agent._to_normalized_actions(env_actions), torch.tensor([[0.5, -0.25]]))


def test_object_offline_baselines_share_cql_training_budget():
    cql = DEFAULTS["g1_29dof_wbt_cql_w_object"]
    iql = DEFAULTS["g1_29dof_wbt_iql_w_object"]
    td3_bc = DEFAULTS["g1_29dof_wbt_td3_bc_w_object"]

    for experiment in (iql, td3_bc):
        assert experiment.training.num_envs == cql.training.num_envs
        assert experiment.algo.config.num_learning_iterations == cql.algo.config.num_learning_iterations
        assert experiment.algo.config.num_updates == cql.algo.config.num_updates
        assert experiment.algo.config.batch_size == cql.algo.config.batch_size
        assert experiment.algo.config.critic_learning_rate == cql.algo.config.critic_learning_rate
        assert experiment.algo.config.reward_scale == cql.algo.config.reward_scale
        assert experiment.algo.config.offline_dataset_path == cql.algo.config.offline_dataset_path

    assert iql.algo.config.discount == cql.algo.config.gamma
    assert iql.algo.config.bootstrap_truncations == cql.algo.config.bootstrap_truncations
    assert iql.algo.config.actor_learning_rate == 3e-4
    assert iql.algo.config.value_learning_rate == 3e-4
    assert iql.algo.config.tau == 0.005
    assert td3_bc.algo.config.discount == cql.algo.config.gamma
    assert td3_bc.algo.config.actor_learning_rate == 3e-4
    assert td3_bc.algo.config.tau == 0.005
    assert td3_bc.algo.config.policy_delay == 2
