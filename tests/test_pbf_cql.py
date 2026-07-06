import pytest

torch = pytest.importorskip("torch")

from holosoma.agents.pbf_cql.pbf_cql_agent import (  # noqa: E402
    build_all_pair_group_masks,
    build_group_to_action_mask,
    build_singleton_group_masks,
    counterfactual_actions_from_group_masks,
)


def test_pbf_counterfactual_shapes_g9():
    batch_size = 4
    action_dim = 29
    num_groups = 9
    group_indices = [tuple([i]) for i in range(8)] + [tuple(range(8, action_dim))]
    dataset_actions = torch.zeros(batch_size, action_dim)
    actor_actions = torch.ones(batch_size, action_dim)

    group_to_action = build_group_to_action_mask(group_indices, action_dim, device="cpu")
    singleton_masks = build_singleton_group_masks(num_groups, device="cpu")
    pair_masks, pair_i, pair_j = build_all_pair_group_masks(num_groups, device="cpu")

    singleton_actions = counterfactual_actions_from_group_masks(
        dataset_actions,
        actor_actions,
        singleton_masks,
        group_to_action,
    )
    pair_actions = counterfactual_actions_from_group_masks(
        dataset_actions,
        actor_actions,
        pair_masks,
        group_to_action,
    )

    assert singleton_actions.shape == (batch_size, 9, 29)
    assert pair_actions.shape == (batch_size, 36, 29)
    assert pair_i.shape == (36,)
    assert pair_j.shape == (36,)
    assert pair_masks.shape == (36, 9)


def test_pbf_pair_residual_gradient_only_flows_to_pair_q():
    batch_size = 4
    num_groups = 9
    pair_masks, pair_i, pair_j = build_all_pair_group_masks(num_groups, device="cpu")
    num_pairs = int(pair_masks.shape[0])

    q_data = torch.randn(batch_size, requires_grad=True)
    q_single = torch.randn(batch_size, num_groups, requires_grad=True)
    q_pair = torch.randn(batch_size, num_pairs, requires_grad=True)

    q_data_anchor = q_data.detach()
    v_single = (q_single.detach() - q_data_anchor[:, None]).detach()
    v_pair = q_pair - q_data_anchor[:, None]
    single_sum = v_single[:, pair_i] + v_single[:, pair_j]
    delta_pair = v_pair - single_sum.detach()
    loss = torch.relu(delta_pair).mean()
    loss.backward()

    assert q_pair.grad is not None
    assert q_pair.grad.abs().sum() > 0
    assert q_single.grad is None
    assert q_data.grad is None
