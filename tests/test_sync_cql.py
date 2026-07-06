from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.agents.bf_cql.sync_cql_agent import (
    SyncCQLAgent,
    build_group_to_action_mask,
    counterfactual_actions_from_group_masks,
    synergy_residual,
)


def _minimal_agent(*, mode: str = "topk", k: int = 2, delta: float = 0.5) -> SyncCQLAgent:
    agent = object.__new__(SyncCQLAgent)
    agent.device = "cpu"
    agent.bf_cql_group_indices = [(0,), (1,), (2,), (3,)]
    agent.bf_cql_group_names = ["g0", "g1", "g2", "g3"]
    agent.config = SimpleNamespace(
        sync_cql=SimpleNamespace(
            K=k,
            delta_threshold=delta,
            selection_mode=mode,
            drift_mode="rmse",
            eps_gain=0.0,
            margin_m=0.0,
            alpha2=0.0,
            alpha2_lagrange=False,
            tau_syn=5.0,
            lambda_cf=0.0,
            drift_ema=0.0,
            drift_std_momentum=0.999,
            freeze_drift_stats=False,
        )
    )
    agent.sync_group_to_action_mask = build_group_to_action_mask(agent.bf_cql_group_indices, 4, device="cpu")
    return agent


def test_selection_none_zero_sync_penalty_is_backward_compatible_at_loss_level():
    agent = _minimal_agent(mode="none", k=2)
    observations = torch.randn(5, 3)
    critic_observations = torch.randn(5, 4)
    dataset_actions = torch.randn(5, 4)
    q1 = torch.randn(5)
    q2 = torch.randn(5)

    sync_loss, sync_penalty, *_rest, selected_mask, _subset_hash = agent._compute_sync_penalty(
        observations,
        critic_observations,
        dataset_actions,
        q1,
        q2,
    )

    assert torch.allclose(sync_loss, torch.tensor(0.0))
    assert torch.allclose(sync_penalty, torch.tensor(0.0))
    assert selected_mask.shape == (5, 4)
    assert not selected_mask.any()


def test_additive_q_synergy_residual_vanishes():
    group_indices = [(0, 1), (2,), (3,)]
    group_to_action = build_group_to_action_mask(group_indices, 4, device="cpu")
    dataset_actions = torch.randn(7, 4)
    actor_actions = torch.randn(7, 4)
    block_masks = torch.tensor(
        [
            [[True, False, True]],
            [[False, True, True]],
            [[True, True, False]],
            [[True, False, False]],
            [[False, True, False]],
            [[False, False, True]],
            [[True, True, True]],
        ],
        dtype=torch.bool,
    )
    singleton_masks = torch.zeros(7, 3, 3, dtype=torch.bool)
    for batch_idx in range(7):
        selected = torch.nonzero(block_masks[batch_idx, 0], as_tuple=False).flatten()
        for singleton_idx, group_idx in enumerate(selected):
            singleton_masks[batch_idx, singleton_idx, group_idx] = True
    singleton_valid = singleton_masks.any(dim=2)

    weights = torch.tensor([0.3, -0.7, 1.2, 0.5])
    q_data = dataset_actions @ weights
    q_block = counterfactual_actions_from_group_masks(
        dataset_actions,
        actor_actions,
        block_masks,
        group_to_action,
    ).squeeze(1) @ weights
    q_singletons = (
        counterfactual_actions_from_group_masks(dataset_actions, actor_actions, singleton_masks, group_to_action)
        @ weights
    )

    delta, _, _ = synergy_residual(q_data, q_block, q_singletons, singleton_valid)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)


def test_interaction_q_synergy_residual_has_block_gradients():
    group_indices = [(0,), (1,)]
    group_to_action = build_group_to_action_mask(group_indices, 2, device="cpu")
    dataset_actions = torch.zeros(4, 2)
    actor_actions = torch.ones(4, 2, requires_grad=True)
    block_masks = torch.ones(4, 1, 2, dtype=torch.bool)
    singleton_masks = torch.tensor([[[True, False], [False, True]]] * 4, dtype=torch.bool)
    singleton_valid = singleton_masks.any(dim=2)

    block_actions = counterfactual_actions_from_group_masks(
        dataset_actions,
        actor_actions,
        block_masks,
        group_to_action,
    ).squeeze(1)
    singleton_actions = counterfactual_actions_from_group_masks(
        dataset_actions,
        actor_actions,
        singleton_masks,
        group_to_action,
    )
    q_data = dataset_actions[:, 0] * dataset_actions[:, 1]
    q_block = block_actions[:, 0] * block_actions[:, 1]
    q_singletons = singleton_actions[..., 0] * singleton_actions[..., 1]

    delta, _, _ = synergy_residual(q_data.detach(), q_block, q_singletons, singleton_valid)
    penalty = torch.relu(delta).mean()
    penalty.backward()

    assert torch.all(delta > 0.0)
    assert actor_actions.grad is not None
    assert torch.all(actor_actions.grad > 0.0)


def test_selection_shape_empty_mask_and_no_gradient_through_selection():
    agent = _minimal_agent(mode="topk", k=2, delta=10.0)
    batch_size = 6
    group_drift = torch.ones(batch_size, 4, requires_grad=True)

    selected_mask, selected_indices, active_mask = agent._select_sync_groups(
        observations=torch.randn(batch_size, 3),
        critic_observations=torch.randn(batch_size, 4),
        dataset_actions=torch.randn(batch_size, 4),
        actor_actions=torch.randn(batch_size, 4),
        q_data_min=torch.randn(batch_size),
        group_drift=group_drift,
    )

    assert selected_mask.shape == (batch_size, 4)
    assert selected_indices.shape == (batch_size, 2)
    assert active_mask.shape == (batch_size,)
    assert not selected_mask.any()
    assert not active_mask.any()
    assert selected_mask.requires_grad is False
    assert selected_indices.requires_grad is False


def test_topk_selection_does_not_call_critic_and_respects_budget_screening():
    agent = _minimal_agent(mode="topk", k=2, delta=0.5)

    class RaisingCritic:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("topk selection must not evaluate Q")

    agent.qnet = RaisingCritic()
    group_drift = torch.tensor(
        [
            [0.1, 0.9, 0.8, 0.2],
            [0.7, 0.6, 0.1, 0.9],
        ]
    )
    selected_mask, selected_indices, active_mask = agent._select_sync_groups(
        observations=torch.randn(2, 3),
        critic_observations=torch.randn(2, 4),
        dataset_actions=torch.randn(2, 4),
        actor_actions=torch.randn(2, 4),
        q_data_min=torch.randn(2),
        group_drift=group_drift,
    )

    assert active_mask.tolist() == [True, True]
    assert selected_mask.sum(dim=1).tolist() == [2, 2]
    assert selected_indices.shape == (2, 2)
    assert 2 + 1 <= 2 + agent.config.sync_cql.K


def test_density_drift_mode_placeholder_is_explicit():
    agent = _minimal_agent(mode="topk")
    agent.config.sync_cql.drift_mode = "density"
    agent.sync_action_std = torch.ones(4)

    with pytest.raises(NotImplementedError, match="CVAE"):
        agent._compute_group_drift(torch.zeros(2, 4), torch.ones(2, 4))
