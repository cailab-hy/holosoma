from __future__ import annotations

import pytest
import torch
from torch import nn

from holosoma.agents.td3.td3 import Actor
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


def test_td3_bc_action_conversion_round_trip():
    agent = TD3BCAgent.__new__(TD3BCAgent)
    agent.actor = type(
        "ActorStub",
        (),
        {
            "action_scale": torch.tensor([2.0, 4.0]),
            "action_bias": torch.tensor([0.25, -0.5]),
        },
    )()
    normalized_actions = torch.tensor([[0.5, -0.25]])

    env_actions = agent._to_env_actions(normalized_actions)

    torch.testing.assert_close(env_actions, torch.tensor([[1.25, -1.5]]))
    torch.testing.assert_close(agent._to_normalized_actions(env_actions), normalized_actions)


def test_td3_bc_rejects_dataset_actions_outside_configured_bounds():
    agent = TD3BCAgent.__new__(TD3BCAgent)
    agent.actor = type(
        "ActorStub",
        (),
        {
            "action_scale": torch.tensor([2.0, 4.0]),
            "action_bias": torch.tensor([0.0, 0.0]),
        },
    )()

    with pytest.raises(ValueError, match="exceed the configured environment action bounds"):
        agent._normalize_dataset_actions(torch.tensor([[2.1, 0.0]]))


def test_td3_actor_output_stays_normalized_despite_env_scale_buffer():
    actor = Actor(
        obs_indices={"actor_obs": {"start": 0, "end": 2, "size": 2}},
        obs_keys=["actor_obs"],
        n_act=2,
        num_envs=1,
        hidden_dim=16,
        use_tanh=True,
        use_layer_norm=False,
        device="cpu",
        action_scale=torch.tensor([3.0, 4.0]),
        action_bias=torch.tensor([0.0, 0.0]),
    )
    actor.fc_mu.bias.data.fill_(torch.atanh(torch.tensor(0.5)))

    actions, _ = actor(torch.zeros(1, 2))

    torch.testing.assert_close(actions, torch.full((1, 2), 0.5))


def test_td3_target_policy_smoothing_is_bounded_in_normalized_space():
    agent = TD3BCAgent.__new__(TD3BCAgent)
    agent.config = type(
        "ConfigStub",
        (),
        {"target_policy_noise": 10.0, "target_noise_clip": 0.2},
    )()

    smoothed = agent._apply_target_policy_smoothing(torch.full((128, 3), 0.95))

    assert float(smoothed.abs().max().item()) <= 1.0
    assert float((smoothed - 0.95).abs().max().item()) <= 0.200001


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


def test_td3_bc_zero_alpha_reduces_actor_objective_to_bc():
    q1 = torch.tensor([100.0, -100.0])
    policy_actions = torch.tensor([[0.5, -0.5], [0.0, 0.25]])
    dataset_actions = torch.zeros_like(policy_actions)

    total_loss, q_loss, bc_loss, lambda_coef = TD3BCAgent._td3bc_actor_objective(
        q1,
        policy_actions,
        dataset_actions,
        alpha=0.0,
    )

    torch.testing.assert_close(lambda_coef, torch.tensor(0.0))
    torch.testing.assert_close(q_loss, torch.tensor(0.0))
    torch.testing.assert_close(total_loss, bc_loss)


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


class _VectorEvalActor(nn.Module):
    def forward(self, observations):
        return torch.zeros(observations.shape[0], 2, device=observations.device), None


class _VectorEvalEnv:
    num_envs = 3

    def __init__(self):
        self.step_count = 0
        self.is_evaluating = False

    def set_is_evaluating(self):
        self.is_evaluating = True

    def set_is_training(self):
        self.is_evaluating = False

    def reset(self):
        self.step_count = 0
        return torch.zeros(self.num_envs, 4)

    def step(self, actions):
        assert actions.shape == (self.num_envs, 2)
        torch.testing.assert_close(actions, torch.zeros_like(actions))
        self.step_count += 1
        observations = torch.zeros(self.num_envs, 4)
        rewards = torch.ones(self.num_envs)

        if self.step_count == 1:
            dones = torch.tensor([True, False, False])
            reasons = {
                "bad_tracking": torch.tensor([True, False, False]),
                "bad_tracking_body_pos": torch.tensor([True, False, False]),
                "motion_ends": torch.zeros(self.num_envs, dtype=torch.bool),
                "timeout": torch.zeros(self.num_envs, dtype=torch.bool),
            }
        else:
            dones = torch.tensor([False, True, True])
            reasons = {
                "bad_tracking": torch.zeros(self.num_envs, dtype=torch.bool),
                "bad_tracking_body_pos": torch.zeros(self.num_envs, dtype=torch.bool),
                "motion_ends": torch.tensor([False, True, False]),
                "timeout": torch.tensor([False, False, True]),
            }

        return observations, rewards, dones, {
            "time_outs": reasons["timeout"],
            "termination_reasons": reasons,
        }


def test_td3_bc_vectorized_eval_returns_one_episode_per_env():
    agent = TD3BCAgent.__new__(TD3BCAgent)
    agent.env = _VectorEvalEnv()
    agent.actor = _VectorEvalActor()
    agent.actor.action_scale = torch.tensor([2.0, 4.0])
    agent.actor.action_bias = torch.tensor([0.0, 0.0])
    agent.obs_normalization = False
    agent.device = "cpu"

    results = agent.evaluate_vectorized_episodes(max_eval_steps=3)

    assert len(results) == agent.env.num_envs
    assert [result["stop_reason"] for result in results] == ["bad_tracking", "motion_ends", "timeout"]
    assert [result["episode_length"] for result in results] == [1, 2, 2]
    assert [result["episode_return"] for result in results] == [1.0, 2.0, 2.0]
    assert results[0]["bad_tracking_details"] == ["body_pos"]
    assert agent.env.is_evaluating is False
