from __future__ import annotations

import torch

from holosoma.agents.td3.td3_agent import TD3BCAgent
from holosoma.agents.td3.td3_utils import EmpiricalNormalization
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


def test_td3_bc_actor_objective_matches_official_q1_formula():
    q1 = torch.tensor([2.0, -4.0])
    policy_actions = torch.tensor([[0.5, -0.5], [0.0, 0.25]])
    dataset_actions = torch.zeros_like(policy_actions)

    total_loss, q_loss, bc_loss, lambda_coef = TD3BCAgent._td3bc_actor_objective(
        q1,
        policy_actions,
        dataset_actions,
        alpha=2.5,
    )

    expected_lambda = torch.tensor(2.5 / 3.0)
    expected_bc = policy_actions.square().mean()
    torch.testing.assert_close(lambda_coef, expected_lambda)
    torch.testing.assert_close(q_loss, -expected_lambda * q1.mean())
    torch.testing.assert_close(bc_loss, expected_bc)
    torch.testing.assert_close(total_loss, q_loss + bc_loss)


def test_td3_bc_normalizer_fits_complete_fixed_dataset():
    normalizer = EmpiricalNormalization(shape=2, device="cpu")
    values = torch.tensor([[1.0, 3.0], [3.0, 7.0], [5.0, 11.0]])

    normalizer.fit(values, chunk_size=2)
    normalizer.eval()

    torch.testing.assert_close(normalizer.mean, values.mean(dim=0))
    torch.testing.assert_close(normalizer.std, values.std(dim=0, unbiased=False))
    assert normalizer.count.item() == values.shape[0]
    count_before = normalizer.count.clone()
    normalizer(values * 100.0)
    torch.testing.assert_close(normalizer.count, count_before)


def _assert_td3_bc_matches_cql_budget(td3_bc, cql):
    assert td3_bc.training.num_envs == cql.training.num_envs
    assert td3_bc.algo.config.num_learning_iterations == cql.algo.config.num_learning_iterations
    assert td3_bc.algo.config.num_updates == cql.algo.config.num_updates
    assert td3_bc.algo.config.batch_size == cql.algo.config.batch_size
    assert td3_bc.algo.config.discount == cql.algo.config.gamma
    assert td3_bc.algo.config.reward_scale == cql.algo.config.reward_scale
    assert td3_bc.algo.config.bootstrap_truncations == cql.algo.config.bootstrap_truncations
    assert td3_bc.algo.config.offline_dataset_path == cql.algo.config.offline_dataset_path
    assert td3_bc.algo.config.actor_learning_rate == 3e-4
    assert td3_bc.algo.config.critic_learning_rate == 3e-4
    assert td3_bc.algo.config.tau == 0.005
    assert td3_bc.algo.config.policy_delay == 2


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
    _assert_td3_bc_matches_cql_budget(td3_bc, cql)


def test_wbt_td3_bc_is_registered_and_matches_cql_budget():
    _assert_td3_bc_matches_cql_budget(
        DEFAULTS["g1_29dof_wbt_td3_bc"],
        DEFAULTS["g1_29dof_wbt_cql"],
    )


def test_fall_and_getup_td3_bc_is_registered_and_matches_cql_budget():
    _assert_td3_bc_matches_cql_budget(
        DEFAULTS["g1_29dof_wbt_fall_and_getup_td3_bc"],
        DEFAULTS["g1_29dof_wbt_fall_and_getup_cql"],
    )
