from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from holosoma.agents.acl_ql.acl_weight_network import (
    ACLWeightNetwork,
    acl_action_distance,
    acl_quality_for_ood,
    acl_quality_raw,
    clamp_acl_quality,
    compute_acl_monotonicity_loss,
    compute_acl_positivity_loss,
    compute_acl_surrogate_loss,
)
from holosoma.config_values.algo import DEFAULTS as ALGO_DEFAULTS
from holosoma.config_values.experiment import DEFAULTS as EXP_DEFAULTS
from scripts import acl_precompute_quality
from scripts.acl_precompute_quality import discounted_returns, normalize01


def test_acl_quality_eq13_uses_normalized_return_and_reward() -> None:
    rewards = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    dones = np.asarray([False, False, True])
    truncations = np.asarray([False, False, False])
    returns = discounted_returns(rewards, dones, truncations, gamma=1.0)
    quality = 0.5 * normalize01(returns) + 0.5 * normalize01(rewards)
    np.testing.assert_allclose(returns, [6.0, 5.0, 3.0])
    np.testing.assert_allclose(quality, [0.5, 7.0 / 12.0, 0.5])


def test_acl_ood_quality_eq14_is_bounded() -> None:
    m_in = torch.tensor([0.2, 0.8])
    distance = torch.tensor([0.0, 1.0])
    out = acl_quality_for_ood(m_in, distance)
    torch.testing.assert_close(out, torch.tensor([0.6, 0.65]))
    assert torch.all(out > 0.0)
    assert torch.all(out < 1.0)


def test_acl_distance_modes_original_and_rms() -> None:
    delta = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    original = acl_action_distance(delta, "original_l2")
    rms = acl_action_distance(delta, "rms")
    torch.testing.assert_close(original, torch.tensor([5.0, 5.0 ** 0.5]))
    torch.testing.assert_close(rms, original / (delta.shape[-1] ** 0.5))
    torch.testing.assert_close(acl_action_distance(torch.zeros_like(delta), "original_l2"), torch.zeros(2))


def test_acl_quality_raw_and_clamp_fractions_are_explicit() -> None:
    raw = acl_quality_raw(torch.tensor([0.0, 0.5, 1.0]), torch.tensor([3.0, 0.0, -1.0]))
    torch.testing.assert_close(raw, torch.tensor([-0.25, 0.75, 1.25]))
    clamped = clamp_acl_quality(raw)
    assert torch.isclose((raw < 1e-6).float().mean(), torch.tensor(1.0 / 3.0))
    assert torch.isclose((raw > 1.0 - 1e-6).float().mean(), torch.tensor(1.0 / 3.0))
    assert torch.all(clamped > 0.0)
    assert torch.all(clamped < 1.0)


def test_acl_sidecar_respects_episode_boundaries_and_hash(tmp_path: Path) -> None:
    h5_path = tmp_path / "dataset.h5"
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.attrs["num_samples"] = 4
        h5_file["rewards"] = np.asarray([1.0, 2.0, 10.0, 20.0], dtype=np.float32)
        h5_file["dones"] = np.asarray([0, 1, 0, 1], dtype=np.uint8)
        h5_file["truncations"] = np.zeros(4, dtype=np.uint8)
    assert acl_precompute_quality.main([str(h5_path), "--gamma", "1.0"]) == 0
    out = Path(f"{h5_path}.acl_quality.npz")
    with np.load(out) as sidecar:
        np.testing.assert_allclose(sidecar["returns"], [3.0, 2.0, 30.0, 20.0])
        assert int(sidecar["n"]) == 4
        assert "rhash" in sidecar
    assert acl_precompute_quality.main([str(h5_path), "--verify"]) == 0
    with h5py.File(h5_path, "a") as h5_file:
        h5_file["rewards"][0] = 99.0
    assert acl_precompute_quality.main([str(h5_path), "--verify"]) == 1


def test_acl_losses_send_gradients_only_to_weight_network_inputs() -> None:
    net = ACLWeightNetwork(obs_dim=3, action_dim=2, hidden_dim=8, num_layers=2)
    obs = torch.randn(4, 3)
    actions_mu = torch.randn(4, 2)
    actions_beta = torch.randn(4, 2)
    w_mu, _ = net(obs, actions_mu)
    _, w_beta = net(obs, actions_beta)
    m_mu = torch.linspace(0.2, 0.8, 4)
    m_beta = torch.linspace(0.3, 0.9, 4)
    mono = compute_acl_monotonicity_loss(w_mu, w_beta, m_mu, m_beta)
    ord_loss, cql_loss = compute_acl_surrogate_loss(
        w_mu,
        w_beta,
        log_mu=torch.full((4,), -2.0),
        log_beta=torch.full((4,), -1.5),
        d_ord=torch.ones(4),
        d_cql=torch.ones(4),
        alpha=5.0,
    )
    loss = mono + ord_loss + cql_loss + compute_acl_positivity_loss(w_mu, w_beta)
    loss.backward()
    assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in net.parameters())


def test_acl_configs_are_registered() -> None:
    assert ALGO_DEFAULTS["acl_ql"]._target_ == "holosoma.agents.acl_ql.acl_ql_agent.ACLQLAgent"
    assert EXP_DEFAULTS["g1_29dof_wbt_acl_ql"].algo._target_ == "holosoma.agents.acl_ql.acl_ql_agent.ACLQLAgent"
    assert EXP_DEFAULTS["g1_29dof_wbt_acl_ql"].algo.config.acl_distance_mode == "original_l2"
    assert EXP_DEFAULTS["g1_29dof_wbt_acl_ql_rms"].algo.config.acl_distance_mode == "rms"
    assert EXP_DEFAULTS["g1_29dof_wbt_cql"].algo._target_ == "holosoma.agents.cql.cql_agent.CQLAgent"
    assert EXP_DEFAULTS["g1_29dof_wbt_aw_cql"].algo._target_ == "holosoma.agents.aw_cql.aw_cql_agent.AWCQLAgent"
    assert EXP_DEFAULTS["g1_29dof_wbt_asym_cql"].algo._target_ == "holosoma.agents.asym_cql.asym_cql_agent.AsymCQLAgent"
    acl_cfg = EXP_DEFAULTS["g1_29dof_wbt_acl_ql"].algo.config
    rms_cfg = EXP_DEFAULTS["g1_29dof_wbt_acl_ql_rms"].algo.config
    assert {**acl_cfg.__dict__, "acl_distance_mode": "rms"} == rms_cfg.__dict__
