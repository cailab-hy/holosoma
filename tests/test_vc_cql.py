from __future__ import annotations

import torch

from holosoma.agents.cql.cql_agent import CQLAgent
from holosoma.agents.vc_cql.vc_cql import (
    VCActor,
    cap_covariance_condition,
    cap_covariance_trace_ratio,
    covariance_kl,
    linearized_squashed_covariance,
    low_rank_covariance,
    low_rank_gaussian_log_prob,
    margin_gate_conservative_gap,
    weighted_contour_covariance,
)
from holosoma.config_values.algo import vc_cql


def _actor(action_dim: int = 6, rank: int = 2) -> VCActor:
    return VCActor(
        obs_indices={"actor_obs": {"start": 0, "end": 5, "size": 5}},
        obs_keys=["actor_obs"],
        n_act=action_dim,
        num_envs=4,
        hidden_dim=32,
        log_std_max=0.0,
        log_std_min=-5.0,
        use_tanh=True,
        use_layer_norm=False,
        device="cpu",
        action_scale=torch.linspace(0.5, 1.5, action_dim),
        action_bias=torch.zeros(action_dim),
        covariance_rank=rank,
        factor_max=1.0,
    )


def test_low_rank_log_prob_matches_full_gaussian() -> None:
    torch.manual_seed(3)
    batch_size, action_dim, rank = 4, 5, 2
    mean = torch.randn(batch_size, action_dim)
    log_std = torch.randn(batch_size, action_dim).clamp(-2.0, -0.2)
    factor = 0.2 * torch.randn(batch_size, action_dim, rank)
    value = torch.randn(batch_size, action_dim)

    actual = low_rank_gaussian_log_prob(value, mean, log_std, factor)
    covariance = low_rank_covariance(log_std, factor)
    expected = torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance).log_prob(value)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_actor_samples_and_covariance_are_well_formed() -> None:
    torch.manual_seed(5)
    actor = _actor()
    observations = torch.randn(7, 5)
    actions, log_prob = actor.get_actions_and_log_probs(observations)
    _, log_std, factor = actor.distribution_parameters(observations)
    covariance = low_rank_covariance(log_std, factor)

    assert actions.shape == (7, 6)
    assert log_prob.shape == (7,)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_prob).all()
    assert torch.linalg.eigvalsh(covariance).min() > 0.0


def test_actor_log_prob_is_amp_bfloat16_safe() -> None:
    torch.manual_seed(6)
    actor = _actor()
    observations = torch.randn(7, 5)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actions, log_prob = actor.get_actions_and_log_probs(observations)

    assert actions.dtype in (torch.float32, torch.bfloat16)
    assert log_prob.dtype in (torch.float32, torch.bfloat16)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_prob).all()


def test_weighted_contour_covariance_recovers_wider_direction() -> None:
    torch.manual_seed(7)
    deltas = 0.5 * torch.randn(8, 4096, 2)
    q_drops = deltas[..., 0].square() + 25.0 * deltas[..., 1].square()
    covariance, weights = weighted_contour_covariance(
        deltas,
        q_drops,
        temperature=0.25,
        epsilon=1e-5,
        shrinkage=0.0,
    )

    assert weights.shape == (8, 4096)
    assert covariance[:, 0, 0].mean() > 3.0 * covariance[:, 1, 1].mean()


def test_margin_gate_stops_conservative_gradient_below_margin() -> None:
    gap = torch.tensor([-2.0, 0.0, 0.5, 3.0], requires_grad=True)
    gated = margin_gate_conservative_gap(gap, target_margin=0.5)

    torch.testing.assert_close(gated, torch.tensor([0.0, 0.0, 0.0, 2.5]))
    gated.sum().backward()
    torch.testing.assert_close(gap.grad, torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_vanilla_cql_conservative_gap_transform_remains_identity() -> None:
    q1_gap = torch.tensor([-2.0, 1.0])
    q2_gap = torch.tensor([-1.0, 3.0])
    transformed_q1, transformed_q2 = CQLAgent._transform_cql_per_sample_losses(
        object.__new__(CQLAgent),
        q1_gap,
        q2_gap,
    )

    assert transformed_q1 is q1_gap
    assert transformed_q2 is q2_gap


def test_contour_condition_number_is_capped() -> None:
    covariance = torch.diag_embed(torch.tensor([[1e-6, 1e-3, 1.0], [1e-8, 0.1, 2.0]]))
    capped = cap_covariance_condition(covariance, max_condition=100.0, epsilon=1e-8)
    eigenvalues = torch.linalg.eigvalsh(capped)
    condition = eigenvalues[:, -1] / eigenvalues[:, 0]

    assert torch.all(condition <= 100.01)


def test_contour_trace_is_capped_relative_to_policy() -> None:
    contour = 10.0 * torch.eye(4).expand(3, 4, 4).clone()
    policy = 0.5 * torch.eye(4).expand(3, 4, 4).clone()
    capped, scale = cap_covariance_trace_ratio(contour, policy, max_ratio=2.0, epsilon=1e-8)
    contour_trace = torch.diagonal(capped, dim1=-2, dim2=-1).sum(dim=-1)
    policy_trace = torch.diagonal(policy, dim1=-2, dim2=-1).sum(dim=-1)

    assert torch.all(contour_trace <= 2.0 * policy_trace + 1e-6)
    assert torch.all(scale < 1.0)


def test_vc_cql_kill_test_defaults_use_fixed_alpha_and_stabilizers() -> None:
    config = vc_cql.config

    assert config.use_autotune is False
    assert config.alpha_init == 0.3
    assert config.vc_margin_gating is True
    assert config.vc_target_margin == 0.0
    assert config.vc_cov_shrinkage == 0.5
    assert config.vc_cov_max_condition == 100.0
    assert config.vc_max_trace_ratio == 2.0


def test_covariance_kl_is_zero_at_match_and_positive_at_mismatch() -> None:
    target = torch.diag_embed(torch.tensor([[0.2, 0.5, 1.0], [0.4, 0.7, 1.2]]))
    matched = covariance_kl(target, target, epsilon=1e-6)
    mismatched = covariance_kl(2.0 * target, target, epsilon=1e-6)

    torch.testing.assert_close(matched, torch.zeros_like(matched), atol=1e-5, rtol=0.0)
    assert torch.all(mismatched > 0.1)


def test_squashed_covariance_uses_action_scale_and_tanh_jacobian() -> None:
    raw_covariance = torch.eye(2).unsqueeze(0)
    raw_mean = torch.tensor([[0.0, 2.0]])
    action_scale = torch.tensor([2.0, 3.0])
    covariance = linearized_squashed_covariance(raw_covariance, raw_mean, action_scale)

    assert covariance[0, 0, 0] == 4.0
    assert covariance[0, 1, 1] < 0.1


def test_contour_loss_updates_only_covariance_heads_when_features_detached() -> None:
    torch.manual_seed(11)
    actor = _actor()
    observations = torch.randn(16, 5)
    _, log_std, factor = actor.distribution_parameters(
        observations,
        detach_features_for_covariance=True,
    )
    policy_covariance = low_rank_covariance(log_std, factor)
    target_covariance = 0.2 * torch.eye(6).expand(16, 6, 6)
    covariance_kl(policy_covariance, target_covariance, epsilon=1e-5).mean().backward()

    assert actor.fc_logstd.weight.grad is not None
    assert actor.fc_logstd.weight.grad.norm() > 0.0
    assert actor.fc_cov_factor.weight.grad is not None
    assert actor.fc_cov_factor.weight.grad.norm() > 0.0
    assert actor.fc_mu[0].weight.grad is None
    assert all(parameter.grad is None for parameter in actor.net.parameters())
