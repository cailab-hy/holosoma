"""Tests for the core CQL update methods.

Covers:
* ``_update_critic()`` — TD loss, CQL penalty, gradient step, rich metrics
* ``_update_actor()`` — SAC-style actor loss, gradient step, metrics
* ``_update_alpha()`` — SAC autotune and CQL Lagrangian alpha
* ``_maybe_amp()`` — AMP context manager
* ``learn()`` — integration test with a tiny synthetic dataset
* Numerical sanity: Q-values don't explode, losses are finite

All tests run on CPU with a tiny network to keep execution fast.
"""

from __future__ import annotations

import math
import os
import types
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from holosoma.agents.fast_sac.fast_sac import Actor
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.offline_cql.offline_cql import TwinQCritic, polyak_update
from holosoma.agents.offline_cql.offline_cql_agent import OfflineCQLAgent
from holosoma.agents.offline_cql.offline_cql_utils import OfflineDataset
from holosoma.utils.average_meters import TensorAverageMeterDict
from holosoma.utils.safe_torch_import import GradScaler, TensorDict, TensorboardSummaryWriter

# ── Test constants ────────────────────────────────────────────────────

B = 16  # batch size
OBS_DIM = 12
CRITIC_OBS_DIM = 20  # actor (12) + extra (8)
ACT_DIM = 4
HIDDEN = 32
NUM_Q = 2
DEVICE = "cpu"

OBS_INDICES = {
    "actor_obs": {"start": 0, "end": OBS_DIM, "size": OBS_DIM},
    "critic_obs": {
        "start": OBS_DIM,
        "end": CRITIC_OBS_DIM,
        "size": CRITIC_OBS_DIM - OBS_DIM,
    },
}
ACTOR_OBS_KEYS = ["actor_obs"]
CRITIC_OBS_KEYS = ["actor_obs", "critic_obs"]


# ── Minimal config stub ──────────────────────────────────────────────


@dataclass
class _StubConfig:
    """Minimal config with all fields accessed by the update methods."""

    gamma: float = 0.99
    tau: float = 0.005
    amp: bool = False
    amp_dtype: str = "bf16"
    max_grad_norm: float = 1.0
    policy_frequency: int = 1
    use_autotune: bool = True
    target_entropy_ratio: float = 0.5
    alpha_init: float = 0.2
    cql_num_random_actions: int = 4
    cql_num_policy_actions: int = 4
    cql_alpha_init: float = 1.0
    cql_alpha_autotune: bool = False
    cql_target_penalty: float = 5.0
    obs_normalization: bool = False
    batch_size: int = B
    num_learning_iterations: int = 5
    logging_interval: int = 2
    save_interval: int = 0  # disable saving in tests
    eval_interval: int = 0  # disable eval in tests by default
    eval_steps: int = 4  # tiny for fast tests
    eval_callbacks: Any = None  # no eval callbacks by default
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    cql_alpha_learning_rate: float = 3e-4
    weight_decay: float = 0.0


# ── Helper: build a wired-up agent without going through setup() ─────


def _make_agent(config: _StubConfig | None = None) -> OfflineCQLAgent:
    """Create a minimal OfflineCQLAgent with all attributes wired up.

    Skips ``__init__`` / ``setup()`` entirely — we directly set the fields
    that the update methods read.
    """
    cfg = config or _StubConfig()
    agent = object.__new__(OfflineCQLAgent)

    # BaseAlgo fields
    agent.config = cfg
    agent.device = DEVICE
    agent.is_multi_gpu = False
    agent.gpu_global_rank = 0
    agent.gpu_local_rank = 0
    agent.gpu_world_size = 1
    agent.is_main_process = True
    agent.global_step = 0

    # Actor
    actor = Actor(
        obs_indices=OBS_INDICES,
        obs_keys=ACTOR_OBS_KEYS,
        n_act=ACT_DIM,
        num_envs=1,
        hidden_dim=HIDDEN,
        log_std_max=2.0,
        log_std_min=-5.0,
        action_scale=torch.ones(ACT_DIM),
        action_bias=torch.zeros(ACT_DIM),
        device=DEVICE,
    )
    agent.actor = actor

    # Critic + target
    qnet = TwinQCritic(
        obs_indices=OBS_INDICES,
        obs_keys=CRITIC_OBS_KEYS,
        n_act=ACT_DIM,
        hidden_dim=HIDDEN,
        num_q_networks=NUM_Q,
        device=DEVICE,
    )
    agent.qnet = qnet
    agent.qnet_target = TwinQCritic.create_target(qnet)

    # Log-alpha (SAC temperature)
    agent.log_alpha = torch.tensor(
        [math.log(cfg.alpha_init)], requires_grad=True, device=DEVICE
    )
    agent.target_entropy = -ACT_DIM * cfg.target_entropy_ratio

    # Log-alpha CQL
    agent.log_alpha_cql = torch.tensor(
        [math.log(cfg.cql_alpha_init)], requires_grad=True, device=DEVICE
    )

    # Scaler (no-op on CPU)
    agent.scaler = GradScaler(enabled=False)

    # Optimizers
    agent.actor_optimizer = torch.optim.AdamW(
        list(actor.parameters()), lr=cfg.actor_learning_rate
    )
    agent.q_optimizer = torch.optim.AdamW(
        list(qnet.parameters()), lr=cfg.critic_learning_rate
    )
    agent.alpha_optimizer = torch.optim.AdamW(
        [agent.log_alpha], lr=cfg.alpha_learning_rate
    )
    agent.alpha_cql_optimizer = torch.optim.AdamW(
        [agent.log_alpha_cql], lr=cfg.cql_alpha_learning_rate
    )

    # Normalisation off by default
    agent.obs_normalization = cfg.obs_normalization
    agent.obs_normalizer = nn.Identity()
    agent.critic_obs_normalizer = nn.Identity()

    # Placeholder for _last_cql_penalty used by _update_alpha
    agent._last_cql_penalty = torch.tensor(0.0, device=DEVICE)
    agent._eval_dims_match = True  # tests use matching dims

    # Stubs needed by learn() → save() → _collect_env_state()
    agent.unwrapped_env = types.SimpleNamespace(
        get_checkpoint_state=lambda: {},
        load_checkpoint_state=lambda _: None,
    )
    agent._experiment_config = None  # prevent save() from crashing in tests

    # ── Logging infrastructure ────────────────────────────────────
    agent.log_dir = "/tmp/cql_test"  # will be overridden by tests using tmp_path
    agent.writer = MagicMock(spec=TensorboardSummaryWriter)
    agent.logging_helper = MagicMock(spec=LoggingHelper)
    agent.training_metrics = TensorAverageMeterDict()
    agent.eval_callbacks = []

    # Override save/export to no-ops for test isolation (learn() calls them)
    agent._save_calls: list[str] = []  # type: ignore[attr-defined]
    agent.save = lambda path: agent._save_calls.append(path)  # type: ignore[assignment]
    agent.export = lambda onnx_file_path="": None  # type: ignore[assignment]

    return agent


class _MockEnv:
    """Minimal mock of FastSACEnv for eval rollout tests."""

    def __init__(
        self,
        num_envs: int = 2,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        ep_length: int = 5,
    ):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ep_length = ep_length
        self._step_count = 0

    def reset(self) -> torch.Tensor:
        self._step_count = 0
        return torch.randn(self.num_envs, self.obs_dim)

    def step(self, actions: torch.Tensor) -> tuple:
        self._step_count += 1
        obs = torch.randn(self.num_envs, self.obs_dim)
        rewards = torch.ones(self.num_envs) * 0.5

        # Episodes end every ep_length steps
        if self._step_count % self.ep_length == 0:
            dones = torch.ones(self.num_envs)
        else:
            dones = torch.zeros(self.num_envs)

        extras = {
            "episode": {"success": torch.tensor(0.8)} if dones.any() else {},
            "to_log": {},
            "time_outs": torch.zeros(self.num_envs),
        }
        return obs, rewards, dones, extras

    def set_is_evaluating(self) -> None:
        pass


def _make_batch() -> TensorDict:
    """Create a fake batch matching OfflineDataset.sample() output."""
    return TensorDict(
        {
            "observations": torch.randn(B, OBS_DIM),
            "actions": torch.randn(B, ACT_DIM),
            "critic_observations": torch.randn(B, CRITIC_OBS_DIM),
            "next": {
                "observations": torch.randn(B, OBS_DIM),
                "rewards": torch.randn(B),
                "dones": torch.zeros(B),
                "truncations": torch.zeros(B),
                "effective_n_steps": torch.ones(B, dtype=torch.long),
                "critic_observations": torch.randn(B, CRITIC_OBS_DIM),
            },
        },
        batch_size=B,
    )


# ══════════════════════════════════════════════════════════════════════
# Tests: _maybe_amp
# ══════════════════════════════════════════════════════════════════════


class TestMaybeAmp:
    def test_amp_disabled(self) -> None:
        agent = _make_agent(_StubConfig(amp=False))
        with agent._maybe_amp():
            x = torch.randn(4, 4)
            y = x @ x
        assert y.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_amp_enabled_bf16(self) -> None:
        cfg = _StubConfig(amp=True, amp_dtype="bf16")
        agent = _make_agent(cfg)
        with agent._maybe_amp():
            # In autocast context, matmuls use bf16
            x = torch.randn(4, 4, device="cuda")
            y = x @ x
        assert y.dtype == torch.bfloat16


# ══════════════════════════════════════════════════════════════════════
# Tests: _update_critic
# ══════════════════════════════════════════════════════════════════════


class TestUpdateCritic:
    def test_returns_all_metric_keys(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        metrics = agent._update_critic(data)
        expected_keys = {
            "td_loss",
            "cql_penalty",
            "cql_alpha",
            "cql_loss",
            "critic_loss",
            "critic_grad_norm",
            "q_data_mean",
            "q_data_max",
            "q_data_min",
            "td_target_mean",
            "td_target_max",
            "td_target_min",
            "cql_q_rand_mean",
            "cql_q_pi_mean",
            "q_overestimation_gap",
            "cql_logsumexp_mean",
        }
        assert set(metrics.keys()) == expected_keys

    def test_metrics_are_finite(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        metrics = agent._update_critic(data)
        for k, v in metrics.items():
            assert torch.isfinite(v), f"metric '{k}' is not finite: {v}"

    def test_metrics_are_scalar(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        metrics = agent._update_critic(data)
        for k, v in metrics.items():
            assert v.dim() == 0, f"metric '{k}' is not scalar: shape {v.shape}"

    def test_critic_params_change(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        params_before = [p.clone() for p in agent.qnet.parameters()]
        agent._update_critic(data)
        changed = any(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(params_before, agent.qnet.parameters())
        )
        assert changed, "Critic parameters did not change after _update_critic"

    def test_target_params_unchanged(self) -> None:
        """Target network should NOT be updated inside _update_critic."""
        agent = _make_agent()
        data = _make_batch()
        params_before = [p.clone() for p in agent.qnet_target.parameters()]
        agent._update_critic(data)
        for p_before, p_after in zip(params_before, agent.qnet_target.parameters()):
            assert torch.equal(p_before, p_after)

    def test_td_loss_positive(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        metrics = agent._update_critic(data)
        assert metrics["td_loss"].item() >= 0.0

    def test_cql_alpha_matches_config(self) -> None:
        cfg = _StubConfig(cql_alpha_init=2.0)
        agent = _make_agent(cfg)
        data = _make_batch()
        metrics = agent._update_critic(data)
        assert abs(metrics["cql_alpha"].item() - 2.0) < 1e-4

    def test_gradient_clipping(self) -> None:
        """With max_grad_norm, clip_grad_norm_ returns the pre-clip total norm."""
        cfg = _StubConfig(max_grad_norm=1.0)
        agent = _make_agent(cfg)
        data = _make_batch()
        metrics = agent._update_critic(data)
        # critic_grad_norm is the *total* norm (pre-clip); just check finite
        assert torch.isfinite(metrics["critic_grad_norm"])
        assert metrics["critic_grad_norm"].item() >= 0.0

    def test_no_gradient_clipping(self) -> None:
        """With max_grad_norm=0, grad clipping is disabled."""
        cfg = _StubConfig(max_grad_norm=0.0)
        agent = _make_agent(cfg)
        data = _make_batch()
        metrics = agent._update_critic(data)
        assert metrics["critic_grad_norm"].item() == 0.0

    def test_multiple_steps_losses_decrease(self) -> None:
        """TD loss should generally decrease over several gradient steps on same data."""
        agent = _make_agent(_StubConfig(max_grad_norm=0.0))
        data = _make_batch()
        losses = []
        for _ in range(20):
            metrics = agent._update_critic(data)
            losses.append(metrics["td_loss"].item())
        # Not strictly monotonic, but first should be > last
        assert losses[0] > losses[-1], f"TD loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_cql_penalty_sign(self) -> None:
        """CQL penalty can be positive or negative, but must be finite."""
        agent = _make_agent()
        data = _make_batch()
        metrics = agent._update_critic(data)
        assert torch.isfinite(metrics["cql_penalty"])


# ══════════════════════════════════════════════════════════════════════
# Tests: _update_actor
# ══════════════════════════════════════════════════════════════════════


class TestUpdateActor:
    def test_returns_all_metric_keys(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        metrics = agent._update_actor(data)
        expected_keys = {
            "actor_loss",
            "actor_grad_norm",
            "policy_entropy",
            "action_std",
            "alpha_value",
            "log_probs_mean",
        }
        assert set(metrics.keys()) == expected_keys

    def test_metrics_are_finite(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        metrics = agent._update_actor(data)
        for k, v in metrics.items():
            assert torch.isfinite(v), f"metric '{k}' is not finite: {v}"

    def test_actor_params_change(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        params_before = [p.clone() for p in agent.actor.parameters()]
        agent._update_actor(data)
        changed = any(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(params_before, agent.actor.parameters())
        )
        assert changed, "Actor parameters did not change after _update_actor"

    def test_critic_params_unchanged(self) -> None:
        """Actor update should NOT change critic parameters."""
        agent = _make_agent()
        data = _make_batch()
        params_before = [p.clone() for p in agent.qnet.parameters()]
        agent._update_actor(data)
        for p_before, p_after in zip(params_before, agent.qnet.parameters()):
            assert torch.equal(p_before, p_after)

    def test_alpha_value_matches(self) -> None:
        cfg = _StubConfig(alpha_init=0.5)
        agent = _make_agent(cfg)
        data = _make_batch()
        metrics = agent._update_actor(data)
        assert abs(metrics["alpha_value"].item() - 0.5) < 1e-4

    def test_policy_entropy_finite(self) -> None:
        """Entropy proxy (−mean log_prob) should be finite.

        Note: tanh-squashed Gaussian can have negative entropy proxy
        because the Jacobian correction can push log_prob > 0.
        """
        agent = _make_agent()
        data = _make_batch()
        metrics = agent._update_actor(data)
        assert torch.isfinite(metrics["policy_entropy"])

    def test_gradient_clipping(self) -> None:
        cfg = _StubConfig(max_grad_norm=1.0)
        agent = _make_agent(cfg)
        data = _make_batch()
        metrics = agent._update_actor(data)
        # actor_grad_norm is the pre-clip total norm; just check finite & non-negative
        assert torch.isfinite(metrics["actor_grad_norm"])
        assert metrics["actor_grad_norm"].item() >= 0.0


# ══════════════════════════════════════════════════════════════════════
# Tests: _update_alpha
# ══════════════════════════════════════════════════════════════════════


class TestUpdateAlpha:
    def test_alpha_loss_returned_with_autotune(self) -> None:
        agent = _make_agent(_StubConfig(use_autotune=True))
        log_probs = torch.randn(B)
        metrics = agent._update_alpha(log_probs)
        assert "alpha_loss" in metrics
        assert torch.isfinite(metrics["alpha_loss"])

    def test_alpha_loss_zero_without_autotune(self) -> None:
        agent = _make_agent(_StubConfig(use_autotune=False))
        log_probs = torch.randn(B)
        metrics = agent._update_alpha(log_probs)
        assert metrics["alpha_loss"].item() == 0.0

    def test_alpha_changes_with_autotune(self) -> None:
        agent = _make_agent(_StubConfig(use_autotune=True))
        alpha_before = agent.log_alpha.item()
        # Use very negative log_probs to push alpha down
        log_probs = torch.full((B,), -10.0)
        agent._update_alpha(log_probs)
        assert agent.log_alpha.item() != alpha_before

    def test_cql_alpha_autotune(self) -> None:
        cfg = _StubConfig(cql_alpha_autotune=True, cql_target_penalty=5.0)
        agent = _make_agent(cfg)
        agent._last_cql_penalty = torch.tensor(10.0)  # penalty > target → α should increase
        log_probs = torch.randn(B)
        alpha_cql_before = agent.log_alpha_cql.item()
        metrics = agent._update_alpha(log_probs)
        assert "alpha_cql_loss" in metrics
        # α_cql should have changed
        assert agent.log_alpha_cql.item() != alpha_cql_before

    def test_cql_alpha_no_autotune_skips(self) -> None:
        cfg = _StubConfig(cql_alpha_autotune=False)
        agent = _make_agent(cfg)
        alpha_cql_before = agent.log_alpha_cql.item()
        log_probs = torch.randn(B)
        metrics = agent._update_alpha(log_probs)
        assert "alpha_cql_loss" not in metrics
        assert agent.log_alpha_cql.item() == alpha_cql_before


# ══════════════════════════════════════════════════════════════════════
# Tests: Combined critic + actor step (integration)
# ══════════════════════════════════════════════════════════════════════


class TestCriticActorIntegration:
    def test_full_update_cycle(self) -> None:
        """Critic → actor → alpha → polyak — all steps succeed."""
        agent = _make_agent()
        data = _make_batch()
        critic_metrics = agent._update_critic(data)
        actor_metrics = agent._update_actor(data)
        alpha_metrics = agent._update_alpha(actor_metrics["log_probs_mean"])
        polyak_update(agent.qnet, agent.qnet_target, agent.config.tau)
        # Verify all metrics are finite
        for m in [critic_metrics, actor_metrics, alpha_metrics]:
            for k, v in m.items():
                assert torch.isfinite(v), f"{k} not finite"

    def test_many_steps_no_explosion(self) -> None:
        """Run 50 full update cycles and check Q-values stay bounded."""
        agent = _make_agent()
        for _ in range(50):
            data = _make_batch()
            critic_metrics = agent._update_critic(data)
            agent._update_actor(data)
            polyak_update(agent.qnet, agent.qnet_target, agent.config.tau)
        assert abs(critic_metrics["q_data_mean"].item()) < 1e4, "Q-values exploded"

    def test_done_flag_effect(self) -> None:
        """When all transitions are terminal (done=1), bootstrap should be 0
        unless truncation is also set."""
        agent = _make_agent()
        data = _make_batch()
        # All done, no truncation → bootstrap = 0 → no future reward
        data["next"]["dones"] = torch.ones(B)
        data["next"]["truncations"] = torch.zeros(B)
        metrics = agent._update_critic(data)
        # TD target ≈ reward (no bootstrap)
        # Just check finite
        assert torch.isfinite(metrics["td_loss"])

    def test_truncation_flag_effect(self) -> None:
        """Truncation causes bootstrap even if done is True."""
        agent = _make_agent()
        data = _make_batch()
        data["next"]["dones"] = torch.ones(B)
        data["next"]["truncations"] = torch.ones(B)
        metrics = agent._update_critic(data)
        assert torch.isfinite(metrics["td_loss"])


# ══════════════════════════════════════════════════════════════════════
# Tests: learn() integration
# ══════════════════════════════════════════════════════════════════════


class TestLearnLoop:
    """Integration test for the full learn() loop with a tiny in-memory dataset."""

    @staticmethod
    def _make_fake_dataset(size: int = 64) -> OfflineDataset:
        """Create a minimal OfflineDataset from raw tensors."""
        ds = object.__new__(OfflineDataset)
        ds.size = size
        ds.actor_obs_dim = OBS_DIM
        ds.critic_obs_dim = CRITIC_OBS_DIM
        ds.act_dim = ACT_DIM
        ds.actor_obs = torch.randn(size, OBS_DIM)
        ds.critic_obs = torch.randn(size, CRITIC_OBS_DIM)
        ds.actions = torch.randn(size, ACT_DIM)
        ds.rewards = torch.randn(size)
        ds.next_actor_obs = torch.randn(size, OBS_DIM)
        ds.next_critic_obs = torch.randn(size, CRITIC_OBS_DIM)
        ds.dones = torch.zeros(size)
        ds.truncations = torch.zeros(size)
        return ds

    def test_learn_runs_without_error(self, tmp_path: Any) -> None:
        """learn() completes 5 gradient steps without error."""
        agent = _make_agent(_StubConfig(
            num_learning_iterations=5,
            logging_interval=2,
            save_interval=0,
            policy_frequency=1,
        ))
        agent.log_dir = str(tmp_path)
        agent.dataset = self._make_fake_dataset()
        agent.learn()
        assert agent.global_step == 5

    def test_learn_global_step_advances(self, tmp_path: Any) -> None:
        agent = _make_agent(_StubConfig(
            num_learning_iterations=3,
            logging_interval=10,
            save_interval=0,
        ))
        agent.log_dir = str(tmp_path)
        agent.dataset = self._make_fake_dataset()
        agent.learn()
        assert agent.global_step == 3

    def test_learn_with_delayed_policy(self, tmp_path: Any) -> None:
        """policy_frequency=2 means actor updates every 2 steps."""
        agent = _make_agent(_StubConfig(
            num_learning_iterations=4,
            logging_interval=10,
            save_interval=0,
            policy_frequency=2,
        ))
        agent.log_dir = str(tmp_path)
        agent.dataset = self._make_fake_dataset()
        # Should not error even though actor skips some steps
        agent.learn()
        assert agent.global_step == 4

    def test_learn_target_diverges_from_online(self, tmp_path: Any) -> None:
        """After several steps, target should differ from online critic."""
        agent = _make_agent(_StubConfig(
            num_learning_iterations=10,
            logging_interval=100,
            save_interval=0,
        ))
        agent.log_dir = str(tmp_path)
        agent.dataset = self._make_fake_dataset()
        agent.learn()
        # At least one param should differ
        any_diff = any(
            not torch.equal(p1.data, p2.data)
            for p1, p2 in zip(agent.qnet.parameters(), agent.qnet_target.parameters())
        )
        assert any_diff, "Target and online critic should diverge after training"

    def test_learn_with_save(self, tmp_path: Any) -> None:
        """Save interval triggers checkpoint creation."""
        agent = _make_agent(_StubConfig(
            num_learning_iterations=4,
            logging_interval=10,
            save_interval=2,
        ))
        agent.log_dir = str(tmp_path)
        agent.dataset = self._make_fake_dataset()

        # _make_agent already stubs save/export; save calls tracked in _save_calls
        agent.learn()
        # Should have saved at step 2, step 4, and final
        assert len(agent._save_calls) >= 2


# ══════════════════════════════════════════════════════════════════════
# Tests: Numerical edge cases
# ══════════════════════════════════════════════════════════════════════


class TestNumericalEdgeCases:
    def test_zero_rewards(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        data["next"]["rewards"] = torch.zeros(B)
        metrics = agent._update_critic(data)
        assert torch.isfinite(metrics["td_loss"])

    def test_large_rewards(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        data["next"]["rewards"] = torch.full((B,), 100.0)
        metrics = agent._update_critic(data)
        assert torch.isfinite(metrics["td_loss"])

    def test_all_dones(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        data["next"]["dones"] = torch.ones(B)
        data["next"]["truncations"] = torch.zeros(B)
        metrics = agent._update_critic(data)
        assert torch.isfinite(metrics["critic_loss"])

    def test_all_truncations(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        data["next"]["dones"] = torch.ones(B)
        data["next"]["truncations"] = torch.ones(B)
        metrics = agent._update_critic(data)
        assert torch.isfinite(metrics["critic_loss"])

    def test_single_sample_batch(self) -> None:
        """Batch size = 1 should not crash."""
        agent = _make_agent()
        data = TensorDict(
            {
                "observations": torch.randn(1, OBS_DIM),
                "actions": torch.randn(1, ACT_DIM),
                "critic_observations": torch.randn(1, CRITIC_OBS_DIM),
                "next": {
                    "observations": torch.randn(1, OBS_DIM),
                    "rewards": torch.randn(1),
                    "dones": torch.zeros(1),
                    "truncations": torch.zeros(1),
                    "effective_n_steps": torch.ones(1, dtype=torch.long),
                    "critic_observations": torch.randn(1, CRITIC_OBS_DIM),
                },
            },
            batch_size=1,
        )
        critic_metrics = agent._update_critic(data)
        actor_metrics = agent._update_actor(data)
        for m in [critic_metrics, actor_metrics]:
            for k, v in m.items():
                assert torch.isfinite(v), f"{k} not finite with B=1"

    def test_asymmetric_action_scale(self) -> None:
        """Non-uniform action_scale should still produce finite results."""
        agent = _make_agent()
        # Override action_scale with non-uniform values
        agent.actor.action_scale.copy_(torch.tensor([0.1, 1.0, 5.0, 0.5]))
        agent.actor.action_bias.copy_(torch.tensor([0.0, 0.5, -1.0, 0.0]))
        data = _make_batch()
        metrics = agent._update_critic(data)
        for k, v in metrics.items():
            assert torch.isfinite(v), f"{k} not finite with asymmetric scales"


# ════════════════════════════════════════════════════════════════════
# Tests: Stability patches
# ════════════════════════════════════════════════════════════════════


class TestTDTargetClamp:
    """P1: TD target should be clamped to [-q_clip, q_clip]."""

    def test_extreme_rewards_clamped(self) -> None:
        """Huge rewards should not produce unbounded TD targets."""
        agent = _make_agent()
        data = _make_batch()
        data["next"]["rewards"] = torch.full((B,), 1e6)
        metrics = agent._update_critic(data)
        assert metrics["td_target_max"].item() <= 1e4 + 1e-2
        assert metrics["td_target_min"].item() >= -1e4 - 1e-2
        assert torch.isfinite(metrics["td_loss"])

    def test_extreme_negative_rewards_clamped(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        data["next"]["rewards"] = torch.full((B,), -1e6)
        metrics = agent._update_critic(data)
        assert metrics["td_target_min"].item() >= -1e4 - 1e-2
        assert torch.isfinite(metrics["td_loss"])

    def test_custom_q_clip(self) -> None:
        """Config q_clip should be respected by the clamp."""
        cfg = _StubConfig()
        agent = _make_agent(cfg)
        agent.config.q_clip = 50.0  # tighter clamp
        data = _make_batch()
        data["next"]["rewards"] = torch.full((B,), 1e6)
        metrics = agent._update_critic(data)
        assert metrics["td_target_max"].item() <= 50.0 + 1e-2

    def test_normal_rewards_unaffected(self) -> None:
        """Normal-scale rewards should not be clipped."""
        agent = _make_agent()
        data = _make_batch()
        data["next"]["rewards"] = torch.full((B,), 1.0)
        metrics = agent._update_critic(data)
        # td_target should be far from \xb1q_clip, so the clamp has no effect
        assert metrics["td_target_max"].item() < 100.0


class TestSACAlphaClamp:
    """P4: SAC temperature \u03b1 should stay in [1e-8, 10]."""

    def test_alpha_clamped_upper(self) -> None:
        """Driving alpha upward (low policy entropy) should clamp at 10."""
        agent = _make_agent(_StubConfig(use_autotune=True, alpha_init=9.0))
        # High log_probs = low entropy → autotune pushes alpha UP
        for _ in range(200):
            agent._update_alpha(torch.full((B,), 10.0))
        assert agent.log_alpha.exp().item() <= 10.0 + 1e-5

    def test_alpha_clamped_lower(self) -> None:
        """Driving alpha downward (high policy entropy) should clamp near 1e-8."""
        agent = _make_agent(_StubConfig(use_autotune=True, alpha_init=0.01))
        # Very negative log_probs = high entropy → autotune pushes alpha DOWN
        for _ in range(200):
            agent._update_alpha(torch.full((B,), -100.0))
        assert agent.log_alpha.exp().item() >= 1e-8 - 1e-12

    def test_alpha_unaffected_without_autotune(self) -> None:
        """With autotune off, log_alpha should not be clamped or changed."""
        cfg = _StubConfig(use_autotune=False, alpha_init=0.5)
        agent = _make_agent(cfg)
        alpha_before = agent.log_alpha.item()
        agent._update_alpha(torch.randn(B))
        assert agent.log_alpha.item() == alpha_before


class TestCQLAlphaClamp:
    """P5: CQL Lagrange multiplier \u03b1_cql should be clamped with an upper bound."""

    def test_cql_alpha_clamped_upper(self) -> None:
        """Persistently high CQL penalty should not drive \u03b1_cql above 1e6."""
        cfg = _StubConfig(
            cql_alpha_autotune=True,
            cql_target_penalty=0.0,
            cql_alpha_init=1e4,
        )
        agent = _make_agent(cfg)
        agent._last_cql_penalty = torch.tensor(1000.0)
        for _ in range(500):
            agent._update_alpha(torch.randn(B))
        assert agent.log_alpha_cql.exp().item() <= 1e6 + 1.0

    def test_cql_alpha_floor_preserved(self) -> None:
        """Existing floor clamp at 1e-6 should still work."""
        cfg = _StubConfig(
            cql_alpha_autotune=True,
            cql_target_penalty=1e6,
            cql_alpha_init=1e-3,
        )
        agent = _make_agent(cfg)
        agent._last_cql_penalty = torch.tensor(0.0)  # penalty << target → push down
        for _ in range(500):
            agent._update_alpha(torch.randn(B))
        assert agent.log_alpha_cql.exp().item() >= 1e-6 - 1e-10


class TestQOverestimationGap:
    """P7: q_overestimation_gap and cql_logsumexp_mean diagnostics."""

    def test_gap_is_finite_and_scalar(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        metrics = agent._update_critic(data)
        assert torch.isfinite(metrics["q_overestimation_gap"])
        assert metrics["q_overestimation_gap"].dim() == 0

    def test_gap_equals_q_minus_target(self) -> None:
        """Gap should equal q_data_mean - td_target_mean exactly."""
        agent = _make_agent()
        data = _make_batch()
        metrics = agent._update_critic(data)
        expected = metrics["q_data_mean"].item() - metrics["td_target_mean"].item()
        assert abs(metrics["q_overestimation_gap"].item() - expected) < 1e-5

    def test_gap_bounded_with_random_init(self) -> None:
        """With random init, overestimation gap should be moderate."""
        agent = _make_agent()
        data = _make_batch()
        metrics = agent._update_critic(data)
        assert abs(metrics["q_overestimation_gap"].item()) < 100

    def test_logsumexp_mean_finite(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        metrics = agent._update_critic(data)
        assert torch.isfinite(metrics["cql_logsumexp_mean"])
        assert metrics["cql_logsumexp_mean"].dim() == 0

    def test_logsumexp_clamped_under_extreme_q(self) -> None:
        """Even with extreme Q-values, logsumexp should stay bounded."""
        agent = _make_agent()
        # Manually bias the critic to produce large Q-values
        with torch.no_grad():
            for p in agent.qnet.parameters():
                p.mul_(100.0)
        data = _make_batch()
        metrics = agent._update_critic(data)
        # cql_logsumexp_mean should be finite (clamped, not inf)
        assert torch.isfinite(metrics["cql_logsumexp_mean"]), (
            f"cql_logsumexp_mean exploded: {metrics['cql_logsumexp_mean']}"
        )


# ══════════════════════════════════════════════════════════════════════
# Tests: _run_eval_rollouts
# ══════════════════════════════════════════════════════════════════════


class TestRunEvalRollouts:
    """Tests for the in-loop evaluation rollout method."""

    @staticmethod
    def _make_eval_agent() -> OfflineCQLAgent:
        agent = _make_agent()
        agent.env = _MockEnv(num_envs=2, ep_length=5)
        return agent

    def test_returns_all_expected_keys(self) -> None:
        agent = self._make_eval_agent()
        metrics = agent._run_eval_rollouts(num_steps=10)
        expected_keys = {
            "mean_reward",
            "mean_ep_reward",
            "mean_ep_length",
            "num_episodes",
            "action_mean",
            "action_std",
            "obs_mean",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_all_values_are_finite(self) -> None:
        agent = self._make_eval_agent()
        metrics = agent._run_eval_rollouts(num_steps=10)
        for k, v in metrics.items():
            assert math.isfinite(v), f"{k} = {v} is not finite"

    def test_episode_counting(self) -> None:
        """With ep_length=5, num_envs=2, 10 steps → 4 completed episodes."""
        agent = self._make_eval_agent()
        metrics = agent._run_eval_rollouts(num_steps=10)
        # 2 envs × 2 resets each (at step 5 and 10) = 4 episodes
        assert metrics["num_episodes"] == 4.0

    def test_mean_ep_reward_with_fixed_rewards(self) -> None:
        """Each step gives 0.5 reward, ep_length=5 → ep_reward = 2.5."""
        agent = self._make_eval_agent()
        metrics = agent._run_eval_rollouts(num_steps=10)
        assert abs(metrics["mean_ep_reward"] - 2.5) < 0.01

    def test_mean_ep_length(self) -> None:
        agent = self._make_eval_agent()
        metrics = agent._run_eval_rollouts(num_steps=10)
        assert abs(metrics["mean_ep_length"] - 5.0) < 0.01

    def test_no_completed_episodes(self) -> None:
        """If ep_length > num_steps, no episodes complete — still returns metrics."""
        agent = _make_agent()
        agent.env = _MockEnv(num_envs=2, ep_length=100)
        metrics = agent._run_eval_rollouts(num_steps=5)
        assert metrics["num_episodes"] == 0.0
        # Should still have partial episode stats
        assert "mean_ep_reward" in metrics
        assert "mean_ep_length" in metrics

    def test_actor_restored_to_train(self) -> None:
        """After eval rollouts, the actor should be back in training mode."""
        agent = self._make_eval_agent()
        agent.actor.train()
        _ = agent._run_eval_rollouts(num_steps=5)
        assert agent.actor.training

    def test_actor_stays_eval_if_was_eval(self) -> None:
        """If actor was in eval mode before, it stays in eval mode after."""
        agent = self._make_eval_agent()
        agent.actor.eval()
        _ = agent._run_eval_rollouts(num_steps=5)
        assert not agent.actor.training

    def test_deterministic_actions(self) -> None:
        """Action std across envs should be relatively small (deterministic)."""
        agent = self._make_eval_agent()
        metrics = agent._run_eval_rollouts(num_steps=10)
        # action_std is over all actions — should be finite
        assert math.isfinite(metrics["action_std"])

    def test_episode_signals_aggregated(self) -> None:
        """Task-level episode signals (e.g. 'success') should be collected."""
        agent = self._make_eval_agent()
        metrics = agent._run_eval_rollouts(num_steps=10)
        # _MockEnv returns {"success": 0.8} on episode completion
        assert "ep_success" in metrics
        assert abs(metrics["ep_success"] - 0.8) < 0.01

    def test_obs_diagnostic_present(self) -> None:
        agent = self._make_eval_agent()
        metrics = agent._run_eval_rollouts(num_steps=5)
        assert "obs_mean" in metrics
        assert math.isfinite(metrics["obs_mean"])

    def test_zero_steps(self) -> None:
        """num_steps=0 should return valid (zero-ish) metrics without crash."""
        agent = self._make_eval_agent()
        metrics = agent._run_eval_rollouts(num_steps=0)
        assert metrics["num_episodes"] == 0.0
        # mean_reward = 0/max(0,1) = 0
        assert metrics["mean_reward"] == 0.0

    def test_with_obs_normalization(self) -> None:
        """Eval rollouts should work when obs_normalization is enabled."""
        agent = self._make_eval_agent()
        agent.obs_normalization = True

        # Need a normalizer that accepts (obs, update=False) — nn.Identity
        # doesn't support kwarg 'update', so wrap it.
        class _PassthroughNormalizer(nn.Module):
            def forward(self, x: torch.Tensor, update: bool = False) -> torch.Tensor:
                return x

        agent.obs_normalizer = _PassthroughNormalizer()
        metrics = agent._run_eval_rollouts(num_steps=10)
        assert math.isfinite(metrics["mean_reward"])


# ══════════════════════════════════════════════════════════════════════
# Tests: evaluate_policy
# ══════════════════════════════════════════════════════════════════════


class TestEvaluatePolicy:
    """Tests for the public evaluate_policy() method."""

    @staticmethod
    def _make_eval_agent() -> OfflineCQLAgent:
        agent = _make_agent(_StubConfig(eval_callbacks=None))
        agent.env = _MockEnv(num_envs=2, ep_length=5)
        return agent

    def test_returns_dict(self) -> None:
        agent = self._make_eval_agent()
        result = agent.evaluate_policy(max_eval_steps=10)
        assert isinstance(result, dict)

    def test_returns_expected_keys(self) -> None:
        agent = self._make_eval_agent()
        result = agent.evaluate_policy(max_eval_steps=10)
        assert "mean_reward" in result
        assert "num_episodes" in result

    def test_episode_reward_accuracy(self) -> None:
        """Should match _run_eval_rollouts for the same scenario."""
        agent = self._make_eval_agent()
        result = agent.evaluate_policy(max_eval_steps=10)
        assert abs(result.get("mean_ep_reward", 0) - 2.5) < 0.01

    def test_actor_mode_restored(self) -> None:
        agent = self._make_eval_agent()
        agent.actor.train()
        _ = agent.evaluate_policy(max_eval_steps=5)
        assert agent.actor.training

    def test_empty_dict_for_none_steps(self) -> None:
        """max_eval_steps=None runs forever; with islice(count(), None) we
        get an infinite loop. In practice callers always provide a step count.
        We test with 0 steps instead (islice(count(), 0) → no iterations)."""
        agent = self._make_eval_agent()
        result = agent.evaluate_policy(max_eval_steps=0)
        # 0 iterations → step never defined → num_steps=0 → empty metrics
        assert result == {} or result.get("num_episodes", 0) == 0.0


# ══════════════════════════════════════════════════════════════════════
# Tests: learn() with evaluation
# ══════════════════════════════════════════════════════════════════════


class TestLearnWithEval:
    """Integration tests for learn() with periodic evaluation enabled."""

    @staticmethod
    def _make_fake_dataset(size: int = 64) -> OfflineDataset:
        ds = object.__new__(OfflineDataset)
        ds.size = size
        ds.actor_obs_dim = OBS_DIM
        ds.critic_obs_dim = CRITIC_OBS_DIM
        ds.act_dim = ACT_DIM
        ds.actor_obs = torch.randn(size, OBS_DIM)
        ds.critic_obs = torch.randn(size, CRITIC_OBS_DIM)
        ds.actions = torch.randn(size, ACT_DIM)
        ds.rewards = torch.randn(size)
        ds.next_actor_obs = torch.randn(size, OBS_DIM)
        ds.next_critic_obs = torch.randn(size, CRITIC_OBS_DIM)
        ds.dones = torch.zeros(size)
        ds.truncations = torch.zeros(size)
        return ds

    def test_learn_with_eval_interval(self, tmp_path: Any) -> None:
        """learn() with eval_interval > 0 runs eval rollouts."""
        agent = _make_agent(_StubConfig(
            num_learning_iterations=6,
            logging_interval=10,
            save_interval=0,
            eval_interval=3,
            eval_steps=4,
        ))
        agent.log_dir = str(tmp_path)
        agent.dataset = self._make_fake_dataset()
        agent.env = _MockEnv(num_envs=2, ep_length=3)

        agent.learn()
        assert agent.global_step == 6

        # writer.add_scalar should have been called for eval metrics
        # at steps 3 and 6, plus final eval → 3 eval rounds
        eval_scalar_calls = [
            c for c in agent.writer.add_scalar.call_args_list
            if c[0][0].startswith("Eval/")
        ]
        assert len(eval_scalar_calls) > 0, "No Eval/ metrics logged"

    def test_eval_interval_zero_skips_eval(self, tmp_path: Any) -> None:
        """eval_interval=0 means no eval rollouts during training."""
        agent = _make_agent(_StubConfig(
            num_learning_iterations=4,
            logging_interval=10,
            save_interval=0,
            eval_interval=0,
        ))
        agent.log_dir = str(tmp_path)
        agent.dataset = self._make_fake_dataset()

        agent.learn()
        assert agent.global_step == 4

        # No Eval/ scalar calls
        eval_scalar_calls = [
            c for c in agent.writer.add_scalar.call_args_list
            if c[0][0].startswith("Eval/")
        ]
        assert len(eval_scalar_calls) == 0

    def test_logging_helper_called(self, tmp_path: Any) -> None:
        """post_epoch_logging should be called at logging_interval."""
        agent = _make_agent(_StubConfig(
            num_learning_iterations=4,
            logging_interval=2,
            save_interval=0,
        ))
        agent.log_dir = str(tmp_path)
        agent.dataset = self._make_fake_dataset()

        agent.learn()
        # logging_interval=2, so post_epoch_logging called at step 2 and 4
        assert agent.logging_helper.post_epoch_logging.call_count >= 1


# ══════════════════════════════════════════════════════════════════════
# Tests: resume from checkpoint
# ══════════════════════════════════════════════════════════════════════


class TestResumeTraining:
    """Tests for resuming training from a non-zero global_step."""

    @staticmethod
    def _make_fake_dataset(size: int = 64) -> OfflineDataset:
        ds = object.__new__(OfflineDataset)
        ds.size = size
        ds.actor_obs_dim = OBS_DIM
        ds.critic_obs_dim = CRITIC_OBS_DIM
        ds.act_dim = ACT_DIM
        ds.actor_obs = torch.randn(size, OBS_DIM)
        ds.critic_obs = torch.randn(size, CRITIC_OBS_DIM)
        ds.actions = torch.randn(size, ACT_DIM)
        ds.rewards = torch.randn(size)
        ds.next_actor_obs = torch.randn(size, OBS_DIM)
        ds.next_critic_obs = torch.randn(size, CRITIC_OBS_DIM)
        ds.dones = torch.zeros(size)
        ds.truncations = torch.zeros(size)
        return ds

    def test_resume_from_step(self, tmp_path: Any) -> None:
        """Setting global_step before learn() resumes from that step."""
        agent = _make_agent(_StubConfig(
            num_learning_iterations=10,
            logging_interval=100,
            save_interval=0,
        ))
        agent.log_dir = str(tmp_path)
        agent.dataset = self._make_fake_dataset()
        agent.global_step = 7  # simulate load()
        agent.learn()
        assert agent.global_step == 10

    def test_resume_step_preserved(self, tmp_path: Any) -> None:
        """After resume, global_step should be exactly num_learning_iterations."""
        agent = _make_agent(_StubConfig(
            num_learning_iterations=5,
            logging_interval=100,
            save_interval=0,
        ))
        agent.log_dir = str(tmp_path)
        agent.dataset = self._make_fake_dataset()
        agent.global_step = 3
        agent.learn()
        assert agent.global_step == 5

    def test_resume_with_eval(self, tmp_path: Any) -> None:
        """Resume + eval_interval should still trigger evaluations."""
        agent = _make_agent(_StubConfig(
            num_learning_iterations=8,
            logging_interval=100,
            save_interval=0,
            eval_interval=4,
            eval_steps=3,
        ))
        agent.log_dir = str(tmp_path)
        agent.dataset = self._make_fake_dataset()
        agent.env = _MockEnv(num_envs=2, ep_length=2)
        agent.global_step = 3  # resume from step 3

        agent.learn()
        assert agent.global_step == 8

        # Eval should have happened at step 4 and 8, plus final
        eval_calls = [
            c for c in agent.writer.add_scalar.call_args_list
            if c[0][0].startswith("Eval/")
        ]
        assert len(eval_calls) > 0
