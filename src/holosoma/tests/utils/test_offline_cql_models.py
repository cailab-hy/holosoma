"""Tests for offline_cql model components.

Covers:
* ScalarQNetwork forward shape
* TwinQCritic forward / min_q / q_values_for_actions shapes
* TwinQCritic.create_target (weight equality, grad freeze, independence)
* polyak_update convergence and idempotence
* Actor reuse (imported unchanged from fast_sac)
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from holosoma.agents.offline_cql.offline_cql import (
    Actor,
    ScalarQNetwork,
    TwinQCritic,
    polyak_update,
)

# ── Fixtures ──────────────────────────────────────────────────────────

B = 32  # batch size
OBS_DIM = 48  # actor obs dim
CRITIC_OBS_DIM = 72  # critic obs dim (actor_obs + critic_obs)
ACT_DIM = 29
HIDDEN = 256
NUM_Q = 2
DEVICE = "cpu"

# Observation indices that simulate a realistic two-key setup:
#   actor_obs: [0, OBS_DIM)   — used by actor
#   critic_obs: [OBS_DIM, CRITIC_OBS_DIM)  — extra features for critic
OBS_INDICES = {
    "actor_obs": {"start": 0, "end": OBS_DIM, "size": OBS_DIM},
    "critic_obs": {"start": OBS_DIM, "end": CRITIC_OBS_DIM, "size": CRITIC_OBS_DIM - OBS_DIM},
}

ACTOR_OBS_KEYS = ["actor_obs"]
CRITIC_OBS_KEYS = ["actor_obs", "critic_obs"]


@pytest.fixture()
def scalar_qnet() -> ScalarQNetwork:
    return ScalarQNetwork(
        n_obs=CRITIC_OBS_DIM, n_act=ACT_DIM, hidden_dim=HIDDEN, device=DEVICE
    )


@pytest.fixture()
def twin_q() -> TwinQCritic:
    return TwinQCritic(
        obs_indices=OBS_INDICES,
        obs_keys=CRITIC_OBS_KEYS,
        n_act=ACT_DIM,
        hidden_dim=HIDDEN,
        num_q_networks=NUM_Q,
        device=DEVICE,
    )


# ── ScalarQNetwork tests ─────────────────────────────────────────────


class TestScalarQNetwork:
    def test_forward_shape(self, scalar_qnet: ScalarQNetwork) -> None:
        obs = torch.randn(B, CRITIC_OBS_DIM)
        act = torch.randn(B, ACT_DIM)
        q = scalar_qnet(obs, act)
        assert q.shape == (B, 1)

    def test_forward_dtype(self, scalar_qnet: ScalarQNetwork) -> None:
        obs = torch.randn(B, CRITIC_OBS_DIM)
        act = torch.randn(B, ACT_DIM)
        q = scalar_qnet(obs, act)
        assert q.dtype == torch.float32

    def test_gradients_flow(self, scalar_qnet: ScalarQNetwork) -> None:
        obs = torch.randn(B, CRITIC_OBS_DIM)
        act = torch.randn(B, ACT_DIM)
        q = scalar_qnet(obs, act)
        loss = q.mean()
        loss.backward()
        # At least one parameter should have a gradient
        grads = [p.grad for p in scalar_qnet.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_no_layer_norm(self) -> None:
        """With use_layer_norm=False, network should still produce [B, 1]."""
        qnet = ScalarQNetwork(
            n_obs=CRITIC_OBS_DIM,
            n_act=ACT_DIM,
            hidden_dim=HIDDEN,
            use_layer_norm=False,
            device=DEVICE,
        )
        obs = torch.randn(B, CRITIC_OBS_DIM)
        act = torch.randn(B, ACT_DIM)
        assert qnet(obs, act).shape == (B, 1)


# ── TwinQCritic tests ────────────────────────────────────────────────


class TestTwinQCritic:
    def test_forward_shape(self, twin_q: TwinQCritic) -> None:
        obs = torch.randn(B, CRITIC_OBS_DIM)
        act = torch.randn(B, ACT_DIM)
        q = twin_q(obs, act)
        assert q.shape == (NUM_Q, B, 1)

    def test_min_q_shape(self, twin_q: TwinQCritic) -> None:
        obs = torch.randn(B, CRITIC_OBS_DIM)
        act = torch.randn(B, ACT_DIM)
        q_min = twin_q.min_q(obs, act)
        assert q_min.shape == (B, 1)

    def test_min_q_is_elementwise_min(self, twin_q: TwinQCritic) -> None:
        """min_q should equal torch.min(forward, dim=0)."""
        obs = torch.randn(B, CRITIC_OBS_DIM)
        act = torch.randn(B, ACT_DIM)
        q_all = twin_q(obs, act)  # [NUM_Q, B, 1]
        q_min = twin_q.min_q(obs, act)
        expected = q_all.min(dim=0).values
        torch.testing.assert_close(q_min, expected)

    def test_q_values_for_actions_shape(self, twin_q: TwinQCritic) -> None:
        N = 10  # actions per sample
        obs_full = torch.randn(B, CRITIC_OBS_DIM)
        obs_processed = twin_q.process_obs(obs_full)
        actions = torch.randn(B, N, ACT_DIM)
        q_vals = twin_q.q_values_for_actions(obs_processed, actions)
        assert q_vals.shape == (NUM_Q, B, N, 1)

    def test_q_values_for_actions_consistency(self, twin_q: TwinQCritic) -> None:
        """q_values_for_actions[:, :, j, :] should match forward(obs, actions[:, j])
        when obs is already processed."""
        N = 5
        obs_full = torch.randn(B, CRITIC_OBS_DIM)
        obs_processed = twin_q.process_obs(obs_full)
        actions = torch.randn(B, N, ACT_DIM)

        q_batch = twin_q.q_values_for_actions(obs_processed, actions)
        for j in range(N):
            # Evaluate the j-th action set through the Q-networks directly
            q_single = torch.stack(
                [qnet(obs_processed, actions[:, j]) for qnet in twin_q.qnets], dim=0
            )
            torch.testing.assert_close(q_batch[:, :, j, :], q_single)

    def test_process_obs_slicing(self, twin_q: TwinQCritic) -> None:
        """process_obs should concatenate actor_obs and critic_obs."""
        obs = torch.randn(B, CRITIC_OBS_DIM)
        processed = twin_q.process_obs(obs)
        # With both keys, should equal the full obs
        torch.testing.assert_close(processed, obs)

    def test_process_obs_actor_only(self) -> None:
        """With only actor_obs key, should slice first OBS_DIM dims."""
        critic = TwinQCritic(
            obs_indices=OBS_INDICES,
            obs_keys=ACTOR_OBS_KEYS,  # only actor_obs
            n_act=ACT_DIM,
            hidden_dim=HIDDEN,
            device=DEVICE,
        )
        obs = torch.randn(B, CRITIC_OBS_DIM)
        processed = critic.process_obs(obs)
        assert processed.shape == (B, OBS_DIM)
        torch.testing.assert_close(processed, obs[:, :OBS_DIM])

    def test_num_q_networks_validation(self) -> None:
        with pytest.raises(ValueError, match="num_q_networks must be at least 1"):
            TwinQCritic(
                obs_indices=OBS_INDICES,
                obs_keys=CRITIC_OBS_KEYS,
                n_act=ACT_DIM,
                hidden_dim=HIDDEN,
                num_q_networks=0,
                device=DEVICE,
            )

    def test_three_q_networks(self) -> None:
        """Ensemble with 3 Q-networks should work."""
        critic = TwinQCritic(
            obs_indices=OBS_INDICES,
            obs_keys=CRITIC_OBS_KEYS,
            n_act=ACT_DIM,
            hidden_dim=HIDDEN,
            num_q_networks=3,
            device=DEVICE,
        )
        obs = torch.randn(B, CRITIC_OBS_DIM)
        act = torch.randn(B, ACT_DIM)
        assert critic(obs, act).shape == (3, B, 1)
        assert critic.min_q(obs, act).shape == (B, 1)


# ── create_target tests ──────────────────────────────────────────────


class TestCreateTarget:
    def test_weights_identical_after_creation(self, twin_q: TwinQCritic) -> None:
        target = TwinQCritic.create_target(twin_q)
        for p_src, p_tgt in zip(twin_q.parameters(), target.parameters()):
            torch.testing.assert_close(p_src.data, p_tgt.data)

    def test_target_has_no_grad(self, twin_q: TwinQCritic) -> None:
        target = TwinQCritic.create_target(twin_q)
        for p in target.parameters():
            assert not p.requires_grad

    def test_source_still_has_grad(self, twin_q: TwinQCritic) -> None:
        _ = TwinQCritic.create_target(twin_q)
        for p in twin_q.parameters():
            assert p.requires_grad

    def test_target_is_independent_copy(self, twin_q: TwinQCritic) -> None:
        """Modifying source should not affect target."""
        target = TwinQCritic.create_target(twin_q)
        # Zero out source weights
        with torch.no_grad():
            for p in twin_q.parameters():
                p.zero_()
        # Target should be unchanged (non-zero somewhere)
        any_nonzero = any(p.data.abs().sum() > 0 for p in target.parameters())
        assert any_nonzero, "Target should be an independent copy"

    def test_target_produces_same_output(self, twin_q: TwinQCritic) -> None:
        target = TwinQCritic.create_target(twin_q)
        obs = torch.randn(B, CRITIC_OBS_DIM)
        act = torch.randn(B, ACT_DIM)
        q_src = twin_q(obs, act)
        q_tgt = target(obs, act)
        torch.testing.assert_close(q_src, q_tgt)

    def test_target_has_same_attributes(self, twin_q: TwinQCritic) -> None:
        target = TwinQCritic.create_target(twin_q)
        assert target.num_q_networks == twin_q.num_q_networks
        assert target.n_act == twin_q.n_act
        assert target.hidden_dim == twin_q.hidden_dim
        assert target.obs_keys == twin_q.obs_keys


# ── polyak_update tests ──────────────────────────────────────────────


class TestPolyakUpdate:
    def test_tau_one_copies_source(self, twin_q: TwinQCritic) -> None:
        """tau=1.0 should make target identical to source."""
        target = TwinQCritic.create_target(twin_q)
        # Randomise source weights to differ from target
        with torch.no_grad():
            for p in twin_q.parameters():
                p.normal_()
        polyak_update(twin_q, target, tau=1.0)
        for p_src, p_tgt in zip(twin_q.parameters(), target.parameters()):
            torch.testing.assert_close(p_src.data, p_tgt.data)

    def test_tau_zero_keeps_target(self, twin_q: TwinQCritic) -> None:
        """tau=0.0 should leave target unchanged."""
        target = TwinQCritic.create_target(twin_q)
        # Save target weights
        original_weights = [p.data.clone() for p in target.parameters()]
        # Randomise source
        with torch.no_grad():
            for p in twin_q.parameters():
                p.normal_()
        polyak_update(twin_q, target, tau=0.0)
        for orig, p_tgt in zip(original_weights, target.parameters()):
            torch.testing.assert_close(orig, p_tgt.data)

    def test_interpolation(self, twin_q: TwinQCritic) -> None:
        """tau=0.5 should produce the midpoint."""
        target = TwinQCritic.create_target(twin_q)
        target_original = [p.data.clone() for p in target.parameters()]
        # Randomise source
        with torch.no_grad():
            for p in twin_q.parameters():
                p.normal_()
        source_weights = [p.data.clone() for p in twin_q.parameters()]

        polyak_update(twin_q, target, tau=0.5)
        for src, tgt_orig, p_tgt in zip(
            source_weights, target_original, target.parameters()
        ):
            expected = 0.5 * src + 0.5 * tgt_orig
            torch.testing.assert_close(p_tgt.data, expected)

    def test_convergence(self, twin_q: TwinQCritic) -> None:
        """Repeated small-tau updates should converge target to source."""
        target = TwinQCritic.create_target(twin_q)
        with torch.no_grad():
            for p in twin_q.parameters():
                p.normal_()
        for _ in range(2000):
            polyak_update(twin_q, target, tau=0.01)
        for p_src, p_tgt in zip(twin_q.parameters(), target.parameters()):
            torch.testing.assert_close(p_src.data, p_tgt.data, atol=1e-3, rtol=1e-3)

    def test_does_not_modify_source(self, twin_q: TwinQCritic) -> None:
        target = TwinQCritic.create_target(twin_q)
        source_before = [p.data.clone() for p in twin_q.parameters()]
        polyak_update(twin_q, target, tau=0.5)
        for before, p in zip(source_before, twin_q.parameters()):
            torch.testing.assert_close(before, p.data)


# ── Actor reuse verification ─────────────────────────────────────────


class TestActorReuse:
    """Verify that Actor imported from fast_sac works correctly."""

    def test_actor_is_from_fast_sac(self) -> None:
        """The Actor class should be the exact same class from fast_sac."""
        from holosoma.agents.fast_sac.fast_sac import Actor as FastSACActor

        assert Actor is FastSACActor

    def test_actor_forward_shape(self) -> None:
        actor = Actor(
            obs_indices=OBS_INDICES,
            obs_keys=ACTOR_OBS_KEYS,
            n_act=ACT_DIM,
            num_envs=1,
            hidden_dim=HIDDEN,
            log_std_max=2.0,
            log_std_min=-5.0,
            device=DEVICE,
        )
        obs = torch.randn(B, CRITIC_OBS_DIM)
        action, mean, log_std = actor(obs)
        assert action.shape == (B, ACT_DIM)
        assert mean.shape == (B, ACT_DIM)
        assert log_std.shape == (B, ACT_DIM)

    def test_actor_get_actions_and_log_probs_shape(self) -> None:
        actor = Actor(
            obs_indices=OBS_INDICES,
            obs_keys=ACTOR_OBS_KEYS,
            n_act=ACT_DIM,
            num_envs=1,
            hidden_dim=HIDDEN,
            log_std_max=2.0,
            log_std_min=-5.0,
            device=DEVICE,
        )
        obs = torch.randn(B, CRITIC_OBS_DIM)
        action, log_prob = actor.get_actions_and_log_probs(obs)
        assert action.shape == (B, ACT_DIM)
        assert log_prob.shape == (B,), f"log_prob should be summed over actions, got {log_prob.shape}"

    def test_actor_action_scale_preserved(self) -> None:
        """Custom action_scale should be registered and used."""
        scale = torch.rand(ACT_DIM) * 2 + 0.1
        bias = torch.rand(ACT_DIM) * 0.5
        actor = Actor(
            obs_indices=OBS_INDICES,
            obs_keys=ACTOR_OBS_KEYS,
            n_act=ACT_DIM,
            num_envs=1,
            hidden_dim=HIDDEN,
            log_std_max=2.0,
            log_std_min=-5.0,
            device=DEVICE,
            action_scale=scale,
            action_bias=bias,
        )
        torch.testing.assert_close(actor.action_scale, scale)
        torch.testing.assert_close(actor.action_bias, bias)
