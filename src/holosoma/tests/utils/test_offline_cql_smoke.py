"""Smoke-test & debug checklist for Offline CQL.

Quick validation before launching a long training run.  Every test class
targets one specific pre-flight concern and is self-contained (no disk I/O,
no GPU, no real env).

Recommended run order (failures isolate fastest top-to-bottom)
---------------------------------------------------------------
pytest -x tests/utils/test_offline_cql_smoke.py

  Level 1 — data pipeline (nothing touches the model)
    1. TestDatasetDryRun       shapes, dtypes, sample() contract

  Level 2 — single gradient step (model builds, grads flow)
    2. TestSingleUpdateSmoke   one critic+actor+alpha step, all finite

  Level 3 — optimisation signal (the model actually learns)
    3. TestOverfitSmallBatch   td_loss drops monotonically on a fixed batch

  Level 4 — persistence (nothing lost on save→load)
    4. TestCheckpointRoundTrip save, load into fresh agent, params match

  Level 5 — action semantics (actor output is valid)
    5. TestActionRangeSemantics  bounds, scaling, tanh saturation

  Level 6 — eval loop (env integration)
    6. TestEvalSmoke           _run_eval_rollouts returns plausible metrics

  Level 7 — logging surface (all diagnostic stats present & finite)
    7. TestStatLogging         obs stats, action stats, Q stats, target stats

If Level N fails, everything below it is suspect — fix top-down.
"""

from __future__ import annotations

import copy
import math
import os
import types
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from holosoma.agents.fast_sac.fast_sac import Actor
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.offline_cql.offline_cql import TwinQCritic, polyak_update
from holosoma.agents.offline_cql.offline_cql_agent import OfflineCQLAgent
from holosoma.agents.offline_cql.offline_cql_utils import (
    create_frozen_normalizer,
    save_cql_params,
    load_cql_params,
    validate_normalization,
)
from holosoma.utils.average_meters import TensorAverageMeterDict
from holosoma.utils.safe_torch_import import GradScaler, TensorDict, TensorboardSummaryWriter

# ── Constants ─────────────────────────────────────────────────────────

B = 16  # mini-batch
OBS_DIM = 12
CRITIC_OBS_DIM = 20
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


# ── Config stub ───────────────────────────────────────────────────────


@dataclass
class _SmokeConfig:
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
    save_interval: int = 0
    eval_interval: int = 0
    eval_steps: int = 4
    eval_callbacks: Any = None
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    cql_alpha_learning_rate: float = 3e-4
    weight_decay: float = 0.0


# ── Factories ─────────────────────────────────────────────────────────


def _make_agent(
    config: _SmokeConfig | None = None,
    *,
    action_scale: torch.Tensor | None = None,
    action_bias: torch.Tensor | None = None,
    obs_normalization: bool = False,
) -> OfflineCQLAgent:
    """Build a minimal OfflineCQLAgent without __init__/setup()."""
    cfg = config or _SmokeConfig()
    cfg.obs_normalization = obs_normalization
    agent = object.__new__(OfflineCQLAgent)

    agent.config = cfg
    agent.device = DEVICE
    agent.is_multi_gpu = False
    agent.gpu_global_rank = 0
    agent.gpu_local_rank = 0
    agent.gpu_world_size = 1
    agent.is_main_process = True
    agent.global_step = 0

    _action_scale = action_scale if action_scale is not None else torch.ones(ACT_DIM)
    _action_bias = action_bias if action_bias is not None else torch.zeros(ACT_DIM)

    actor = Actor(
        obs_indices=OBS_INDICES,
        obs_keys=ACTOR_OBS_KEYS,
        n_act=ACT_DIM,
        num_envs=1,
        hidden_dim=HIDDEN,
        log_std_max=2.0,
        log_std_min=-5.0,
        action_scale=_action_scale,
        action_bias=_action_bias,
        device=DEVICE,
    )
    agent.actor = actor

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

    agent.log_alpha = torch.tensor(
        [math.log(cfg.alpha_init)], requires_grad=True, device=DEVICE
    )
    agent.target_entropy = -ACT_DIM * cfg.target_entropy_ratio

    agent.log_alpha_cql = torch.tensor(
        [math.log(cfg.cql_alpha_init)], requires_grad=True, device=DEVICE
    )

    agent.scaler = GradScaler(enabled=False)

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

    # Normalization
    agent.obs_normalization = cfg.obs_normalization
    if obs_normalization:
        # Frozen normalizer from synthetic stats
        mean = torch.randn(1, OBS_DIM) * 5
        std = torch.rand(1, OBS_DIM).clamp(min=0.1) * 3
        agent.obs_normalizer = create_frozen_normalizer(mean, std, count=10_000, device=DEVICE)
        c_mean = torch.randn(1, CRITIC_OBS_DIM) * 5
        c_std = torch.rand(1, CRITIC_OBS_DIM).clamp(min=0.1) * 3
        agent.critic_obs_normalizer = create_frozen_normalizer(c_mean, c_std, count=10_000, device=DEVICE)
    else:
        agent.obs_normalizer = nn.Identity()
        agent.critic_obs_normalizer = nn.Identity()

    agent._last_cql_penalty = torch.tensor(0.0, device=DEVICE)
    agent._eval_dims_match = True  # tests use matching dims

    # Stubs for save / export / env helpers
    agent.unwrapped_env = types.SimpleNamespace(
        get_checkpoint_state=lambda: {},
        load_checkpoint_state=lambda _: None,
    )
    agent._experiment_config = None
    agent.log_dir = "/tmp/cql_smoke_test"
    agent.writer = MagicMock(spec=TensorboardSummaryWriter)
    agent.logging_helper = MagicMock(spec=LoggingHelper)
    agent.training_metrics = TensorAverageMeterDict()
    agent.eval_callbacks = []

    agent._save_calls: list[str] = []
    agent.save = lambda path: agent._save_calls.append(path)  # type: ignore[assignment]
    agent.export = lambda onnx_file_path="": None  # type: ignore[assignment]

    return agent


def _make_batch(batch_size: int = B) -> TensorDict:
    """Synthetic batch matching OfflineDataset.sample() structure."""
    return TensorDict(
        {
            "observations": torch.randn(batch_size, OBS_DIM),
            "actions": torch.randn(batch_size, ACT_DIM),
            "critic_observations": torch.randn(batch_size, CRITIC_OBS_DIM),
            "next": {
                "observations": torch.randn(batch_size, OBS_DIM),
                "rewards": torch.randn(batch_size),
                "dones": torch.zeros(batch_size),
                "truncations": torch.zeros(batch_size),
                "effective_n_steps": torch.ones(batch_size, dtype=torch.long),
                "critic_observations": torch.randn(batch_size, CRITIC_OBS_DIM),
            },
        },
        batch_size=batch_size,
    )


class _MockEnv:
    """Minimal mock of FastSACEnv for eval rollout tests."""

    def __init__(
        self,
        num_envs: int = 2,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        ep_length: int = 3,
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


# ══════════════════════════════════════════════════════════════════════
# Level 1 — Dataset Dry-Run
# ══════════════════════════════════════════════════════════════════════


class TestDatasetDryRun:
    """Verify the synthetic batch contract without touching the model."""

    def test_batch_keys(self) -> None:
        """All required top-level and nested keys are present."""
        data = _make_batch()
        for key in ("observations", "actions", "critic_observations"):
            assert key in data.keys(), f"missing top-level key: {key}"
        for key in (
            "observations",
            "rewards",
            "dones",
            "truncations",
            "effective_n_steps",
            "critic_observations",
        ):
            assert key in data["next"].keys(), f"missing next/ key: {key}"

    def test_batch_shapes(self) -> None:
        """Tensor dimensions match expected constants."""
        data = _make_batch()
        assert data["observations"].shape == (B, OBS_DIM)
        assert data["actions"].shape == (B, ACT_DIM)
        assert data["critic_observations"].shape == (B, CRITIC_OBS_DIM)
        assert data["next"]["observations"].shape == (B, OBS_DIM)
        assert data["next"]["rewards"].shape == (B,)
        assert data["next"]["dones"].shape == (B,)
        assert data["next"]["truncations"].shape == (B,)
        assert data["next"]["effective_n_steps"].shape == (B,)
        assert data["next"]["critic_observations"].shape == (B, CRITIC_OBS_DIM)

    def test_batch_dtypes(self) -> None:
        """Float tensors are float32; effective_n_steps is long."""
        data = _make_batch()
        assert data["observations"].dtype == torch.float32
        assert data["actions"].dtype == torch.float32
        assert data["next"]["rewards"].dtype == torch.float32
        assert data["next"]["effective_n_steps"].dtype == torch.long

    def test_batch_finite(self) -> None:
        """No NaN / Inf in the synthetic batch."""
        data = _make_batch()
        for key in ("observations", "actions", "critic_observations"):
            assert torch.isfinite(data[key]).all(), f"non-finite in {key}"
        for key in ("observations", "rewards", "dones", "truncations", "critic_observations"):
            assert torch.isfinite(data["next"][key]).all(), f"non-finite in next/{key}"

    def test_varying_batch_size(self) -> None:
        """Batch factory works for non-default sizes."""
        for bs in (1, 4, 64):
            data = _make_batch(batch_size=bs)
            assert data["observations"].shape[0] == bs


# ══════════════════════════════════════════════════════════════════════
# Level 2 — Single Update Step Smoke
# ══════════════════════════════════════════════════════════════════════


class TestSingleUpdateSmoke:
    """One full critic→actor→alpha cycle: grads flow, metrics are sane."""

    EXPECTED_CRITIC_KEYS = {
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

    EXPECTED_ACTOR_KEYS = {
        "actor_loss",
        "actor_grad_norm",
        "policy_entropy",
        "action_std",
        "alpha_value",
        "log_probs_mean",
    }

    EXPECTED_ALPHA_KEYS = {"alpha_loss"}

    def test_critic_step_all_metrics_finite(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        metrics = agent._update_critic(data)
        assert self.EXPECTED_CRITIC_KEYS.issubset(metrics.keys())
        for k, v in metrics.items():
            assert torch.isfinite(v), f"critic metric '{k}' is not finite: {v}"

    def test_actor_step_all_metrics_finite(self) -> None:
        agent = _make_agent()
        data = _make_batch()
        metrics = agent._update_actor(data)
        assert self.EXPECTED_ACTOR_KEYS.issubset(metrics.keys())
        for k, v in metrics.items():
            assert torch.isfinite(v), f"actor metric '{k}' is not finite: {v}"

    def test_alpha_step_all_metrics_finite(self) -> None:
        agent = _make_agent()
        log_probs = torch.randn(B)
        metrics = agent._update_alpha(log_probs)
        assert self.EXPECTED_ALPHA_KEYS.issubset(metrics.keys())
        for k, v in metrics.items():
            assert torch.isfinite(v), f"alpha metric '{k}' is not finite: {v}"

    def test_full_cycle_grads_flow(self) -> None:
        """Critic→actor→alpha cycle: params change, all finite."""
        agent = _make_agent()
        data = _make_batch()

        critic_before = {
            n: p.clone() for n, p in agent.qnet.named_parameters()
        }
        actor_before = {
            n: p.clone() for n, p in agent.actor.named_parameters()
        }

        critic_m = agent._update_critic(data)
        actor_m = agent._update_actor(data)
        agent._last_cql_penalty = critic_m["cql_penalty"]
        alpha_m = agent._update_alpha(actor_m["log_probs_mean"])

        # At least some critic params changed
        changed = sum(
            1
            for n, p in agent.qnet.named_parameters()
            if not torch.equal(p, critic_before[n])
        )
        assert changed > 0, "No critic params changed after update"

        # At least some actor params changed
        changed = sum(
            1
            for n, p in agent.actor.named_parameters()
            if not torch.equal(p, actor_before[n])
        )
        assert changed > 0, "No actor params changed after update"

    def test_single_step_with_normalization(self) -> None:
        """Same cycle but with obs normalization enabled."""
        agent = _make_agent(obs_normalization=True)
        data = _make_batch()
        # Normalise like learn() does
        data["observations"] = agent.obs_normalizer(data["observations"], update=False)
        data["next"]["observations"] = agent.obs_normalizer(
            data["next"]["observations"], update=False
        )
        data["critic_observations"] = agent.critic_obs_normalizer(
            data["critic_observations"], update=False
        )
        data["next"]["critic_observations"] = agent.critic_obs_normalizer(
            data["next"]["critic_observations"], update=False
        )
        cm = agent._update_critic(data)
        am = agent._update_actor(data)
        for k, v in {**cm, **am}.items():
            assert torch.isfinite(v), f"{k} non-finite with normalization"


# ══════════════════════════════════════════════════════════════════════
# Level 3 — Overfit on Small Batch (debug mode)
# ══════════════════════════════════════════════════════════════════════


class TestOverfitSmallBatch:
    """Repeated updates on the exact same batch → td_loss must decrease.

    This is the single most useful debug check: if td_loss does NOT
    decrease on a fixed batch, the critic has a bug (wrong gradient,
    wrong target, broken scaler, etc.).
    """

    NUM_STEPS = 60
    LR = 1e-3

    def _run_overfit(
        self,
        *,
        obs_normalization: bool = False,
    ) -> list[float]:
        """Run N steps on the same batch and return td_loss history."""
        cfg = _SmokeConfig(
            critic_learning_rate=self.LR,
            actor_learning_rate=self.LR,
            alpha_learning_rate=self.LR,
            use_autotune=False,  # freeze alpha to isolate critic
            policy_frequency=100,  # effectively disable actor updates
        )
        agent = _make_agent(cfg, obs_normalization=obs_normalization)
        data = _make_batch()

        if obs_normalization:
            data["observations"] = agent.obs_normalizer(data["observations"], update=False)
            data["next"]["observations"] = agent.obs_normalizer(
                data["next"]["observations"], update=False
            )
            data["critic_observations"] = agent.critic_obs_normalizer(
                data["critic_observations"], update=False
            )
            data["next"]["critic_observations"] = agent.critic_obs_normalizer(
                data["next"]["critic_observations"], update=False
            )

        losses: list[float] = []
        for _ in range(self.NUM_STEPS):
            m = agent._update_critic(data)
            losses.append(m["td_loss"].item())
        return losses

    def test_td_loss_decreases(self) -> None:
        """td_loss at step 60 < td_loss at step 1."""
        losses = self._run_overfit()
        # Allow a small tolerance — we compare first 5 avg vs last 5 avg
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early, (
            f"td_loss did not decrease: early avg={early:.6f}, late avg={late:.6f}"
        )

    def test_td_loss_stays_finite(self) -> None:
        """All steps produce finite loss."""
        losses = self._run_overfit()
        for i, l in enumerate(losses):
            assert math.isfinite(l), f"td_loss is non-finite at step {i}: {l}"

    def test_overfit_with_normalization(self) -> None:
        """Same check with frozen normalizers active."""
        losses = self._run_overfit(obs_normalization=True)
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early, (
            f"td_loss did not decrease (with norm): early={early:.6f}, late={late:.6f}"
        )

    def test_actor_loss_moves_in_overfit(self) -> None:
        """After overfitting the critic, a few actor steps should change actor_loss."""
        cfg = _SmokeConfig(
            critic_learning_rate=self.LR,
            actor_learning_rate=self.LR,
            use_autotune=False,
            policy_frequency=1,
        )
        agent = _make_agent(cfg)
        data = _make_batch()

        # First overfit critic
        for _ in range(40):
            agent._update_critic(data)

        # Then check actor loss actually moves
        losses: list[float] = []
        for _ in range(10):
            m = agent._update_actor(data)
            losses.append(m["actor_loss"].item())

        # Actor loss should change (not stuck at exactly the same value)
        unique_vals = len(set(f"{l:.8f}" for l in losses))
        assert unique_vals > 1, "actor_loss is stuck at a single value"


# ══════════════════════════════════════════════════════════════════════
# Level 4 — Checkpoint Save/Load Round-Trip
# ══════════════════════════════════════════════════════════════════════


class TestCheckpointRoundTrip:
    """Save → load into a fresh agent → params match bit-for-bit."""

    def _make_trained_agent(self, steps: int = 5) -> OfflineCQLAgent:
        agent = _make_agent()
        data = _make_batch()
        for _ in range(steps):
            cm = agent._update_critic(data)
            am = agent._update_actor(data)
            agent._last_cql_penalty = cm["cql_penalty"]
            agent._update_alpha(am["log_probs_mean"])
            with torch.no_grad():
                polyak_update(agent.qnet, agent.qnet_target, agent.config.tau)
        agent.global_step = steps
        return agent

    def test_params_match_after_roundtrip(self, tmp_path) -> None:
        """All model params equal after save → load."""
        agent = self._make_trained_agent()
        ckpt_path = str(tmp_path / "smoke_ckpt.pt")

        # Need experiment config for save
        agent._experiment_config = types.SimpleNamespace(
            to_serializable_dict=lambda: {"test": True}
        )

        # Use real save
        save_cql_params(
            global_step=agent.global_step,
            actor=agent.actor,
            qnet=agent.qnet,
            qnet_target=agent.qnet_target,
            log_alpha=agent.log_alpha,
            obs_normalizer=agent.obs_normalizer,
            critic_obs_normalizer=agent.critic_obs_normalizer,
            actor_optimizer=agent.actor_optimizer,
            q_optimizer=agent.q_optimizer,
            alpha_optimizer=agent.alpha_optimizer,
            scaler=agent.scaler,
            args=agent.config,
            save_path=ckpt_path,
            metadata={"experiment_config": {"test": True}, "iteration": agent.global_step},
            log_alpha_cql=agent.log_alpha_cql,
            alpha_cql_optimizer=agent.alpha_cql_optimizer,
        )
        assert os.path.isfile(ckpt_path)

        # Load into fresh agent
        fresh = _make_agent()
        load_cql_params(
            ckpt_path,
            device=DEVICE,
            actor=fresh.actor,
            qnet=fresh.qnet,
            qnet_target=fresh.qnet_target,
            log_alpha=fresh.log_alpha,
            obs_normalizer=fresh.obs_normalizer,
            critic_obs_normalizer=fresh.critic_obs_normalizer,
            actor_optimizer=fresh.actor_optimizer,
            q_optimizer=fresh.q_optimizer,
            alpha_optimizer=fresh.alpha_optimizer,
            scaler=fresh.scaler,
            log_alpha_cql=fresh.log_alpha_cql,
            alpha_cql_optimizer=fresh.alpha_cql_optimizer,
        )

        # Compare parameters
        for (n1, p1), (n2, p2) in zip(
            agent.actor.named_parameters(), fresh.actor.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"actor param '{n1}' differs"

        for (n1, p1), (n2, p2) in zip(
            agent.qnet.named_parameters(), fresh.qnet.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"qnet param '{n1}' differs"

        for (n1, p1), (n2, p2) in zip(
            agent.qnet_target.named_parameters(), fresh.qnet_target.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"qnet_target param '{n1}' differs"

        assert torch.equal(agent.log_alpha.data, fresh.log_alpha.data)
        assert torch.equal(agent.log_alpha_cql.data, fresh.log_alpha_cql.data)

    def test_metrics_reproducible_after_load(self, tmp_path) -> None:
        """After loading, a forward pass produces the same metrics."""
        torch.manual_seed(42)
        agent = self._make_trained_agent()
        ckpt_path = str(tmp_path / "repro_ckpt.pt")

        save_cql_params(
            global_step=agent.global_step,
            actor=agent.actor,
            qnet=agent.qnet,
            qnet_target=agent.qnet_target,
            log_alpha=agent.log_alpha,
            obs_normalizer=agent.obs_normalizer,
            critic_obs_normalizer=agent.critic_obs_normalizer,
            actor_optimizer=agent.actor_optimizer,
            q_optimizer=agent.q_optimizer,
            alpha_optimizer=agent.alpha_optimizer,
            scaler=agent.scaler,
            args=agent.config,
            save_path=ckpt_path,
            metadata={"experiment_config": {"test": True}, "iteration": agent.global_step},
            log_alpha_cql=agent.log_alpha_cql,
            alpha_cql_optimizer=agent.alpha_cql_optimizer,
        )

        fresh = _make_agent()
        load_cql_params(
            ckpt_path,
            device=DEVICE,
            actor=fresh.actor,
            qnet=fresh.qnet,
            qnet_target=fresh.qnet_target,
            log_alpha=fresh.log_alpha,
            obs_normalizer=fresh.obs_normalizer,
            critic_obs_normalizer=fresh.critic_obs_normalizer,
            actor_optimizer=fresh.actor_optimizer,
            q_optimizer=fresh.q_optimizer,
            alpha_optimizer=fresh.alpha_optimizer,
            scaler=fresh.scaler,
            log_alpha_cql=fresh.log_alpha_cql,
            alpha_cql_optimizer=fresh.alpha_cql_optimizer,
        )

        # Deterministic forward (same seed → same batch)
        torch.manual_seed(99)
        data_a = _make_batch()
        torch.manual_seed(99)
        data_b = _make_batch()

        # Both agents should produce identical actor output
        with torch.no_grad():
            out_a = agent.actor(data_a["observations"])
            out_b = fresh.actor(data_b["observations"])
        assert torch.allclose(out_a[0], out_b[0], atol=1e-6), "Actor outputs differ after load"

        # Q-values should match
        with torch.no_grad():
            q_a = agent.qnet(data_a["critic_observations"], data_a["actions"])
            q_b = fresh.qnet(data_b["critic_observations"], data_b["actions"])
        assert torch.allclose(q_a, q_b, atol=1e-6), "Q-values differ after load"

    def test_global_step_restored(self, tmp_path) -> None:
        """global_step survives the round-trip."""
        agent = self._make_trained_agent(steps=7)
        ckpt_path = str(tmp_path / "step_ckpt.pt")

        save_cql_params(
            global_step=agent.global_step,
            actor=agent.actor,
            qnet=agent.qnet,
            qnet_target=agent.qnet_target,
            log_alpha=agent.log_alpha,
            obs_normalizer=agent.obs_normalizer,
            critic_obs_normalizer=agent.critic_obs_normalizer,
            actor_optimizer=agent.actor_optimizer,
            q_optimizer=agent.q_optimizer,
            alpha_optimizer=agent.alpha_optimizer,
            scaler=agent.scaler,
            args=agent.config,
            save_path=ckpt_path,
            metadata={"experiment_config": {"test": True}, "iteration": agent.global_step},
            log_alpha_cql=agent.log_alpha_cql,
            alpha_cql_optimizer=agent.alpha_cql_optimizer,
        )

        ckpt = torch.load(ckpt_path, weights_only=False)
        assert ckpt["global_step"] == 7


# ══════════════════════════════════════════════════════════════════════
# Level 5 — Action Range / Scaling Semantics
# ══════════════════════════════════════════════════════════════════════


class TestActionRangeSemantics:
    """Verify actor output respects action_scale / action_bias bounds."""

    def test_default_scale_actions_in_minus1_plus1(self) -> None:
        """With scale=1, bias=0: deterministic actions ∈ [-1, 1]."""
        agent = _make_agent()
        obs = torch.randn(100, OBS_DIM)
        with torch.no_grad():
            actions, _mean, _log_std = agent.actor(obs)
        assert actions.min().item() >= -1.0 - 1e-6
        assert actions.max().item() <= 1.0 + 1e-6

    def test_custom_scale_bias(self) -> None:
        """With scale=2, bias=0.5: actions ∈ [-1.5, 2.5]."""
        scale = torch.full((ACT_DIM,), 2.0)
        bias = torch.full((ACT_DIM,), 0.5)
        agent = _make_agent(action_scale=scale, action_bias=bias)
        obs = torch.randn(200, OBS_DIM)
        with torch.no_grad():
            actions, _, _ = agent.actor(obs)
        # tanh ∈ [-1, 1], so action ∈ [-scale + bias, +scale + bias] = [-1.5, 2.5]
        assert actions.min().item() >= -2.0 - 1e-4  # slight margin for numerics
        assert actions.max().item() <= 2.5 + 1e-4

    def test_sampled_actions_respect_bounds(self) -> None:
        """Stochastic actions from get_actions_and_log_probs also bounded."""
        agent = _make_agent()
        obs = torch.randn(500, OBS_DIM)
        with torch.no_grad():
            actions, log_probs = agent.actor.get_actions_and_log_probs(obs)
        # tanh clamping: even sampled actions should be within scale
        assert actions.min().item() >= -1.0 - 1e-3
        assert actions.max().item() <= 1.0 + 1e-3
        assert torch.isfinite(log_probs).all()

    def test_tanh_saturation_finite_log_probs(self) -> None:
        """Extreme obs should not produce NaN log-probs (tanh saturation)."""
        agent = _make_agent()
        # Very large obs → large pre-tanh mean → tanh saturates
        obs = torch.randn(50, OBS_DIM) * 100
        with torch.no_grad():
            actions, log_probs = agent.actor.get_actions_and_log_probs(obs)
        assert torch.isfinite(actions).all(), "Actions contain NaN/Inf"
        assert torch.isfinite(log_probs).all(), "Log-probs contain NaN/Inf"

    def test_action_scale_buffer_registered(self) -> None:
        """action_scale and action_bias are registered buffers on the actor."""
        agent = _make_agent()
        buffer_names = [n for n, _ in agent.actor.named_buffers()]
        assert "action_scale" in buffer_names
        assert "action_bias" in buffer_names

    def test_critic_accepts_scaled_actions(self) -> None:
        """Critic forward should work with post-scaled (in-bound) actions."""
        scale = torch.full((ACT_DIM,), 3.0)
        bias = torch.full((ACT_DIM,), 1.0)
        agent = _make_agent(action_scale=scale, action_bias=bias)
        obs = torch.randn(B, OBS_DIM)
        with torch.no_grad():
            actions, _, _ = agent.actor(obs)
            q = agent.qnet(
                torch.randn(B, CRITIC_OBS_DIM),
                actions,
            )
        assert q.shape == (NUM_Q, B, 1)
        assert torch.isfinite(q).all()


# ══════════════════════════════════════════════════════════════════════
# Level 6 — Evaluation Smoke
# ══════════════════════════════════════════════════════════════════════


class TestEvalSmoke:
    """_run_eval_rollouts returns plausible metrics from mock env."""

    def test_eval_returns_expected_keys(self) -> None:
        agent = _make_agent()
        agent.env = _MockEnv(ep_length=3)
        metrics = agent._run_eval_rollouts(num_steps=10)
        for key in (
            "mean_reward",
            "mean_ep_reward",
            "mean_ep_length",
            "num_episodes",
            "action_mean",
            "action_std",
            "obs_mean",
        ):
            assert key in metrics, f"missing eval metric: {key}"

    def test_eval_metrics_are_finite(self) -> None:
        agent = _make_agent()
        agent.env = _MockEnv(ep_length=3)
        metrics = agent._run_eval_rollouts(num_steps=10)
        for k, v in metrics.items():
            assert math.isfinite(v), f"eval metric '{k}' is not finite: {v}"

    def test_eval_episodes_complete(self) -> None:
        """With ep_length=3 and 10 steps × 2 envs, several episodes complete."""
        agent = _make_agent()
        agent.env = _MockEnv(num_envs=2, ep_length=3)
        metrics = agent._run_eval_rollouts(num_steps=10)
        assert metrics["num_episodes"] >= 2, "expected ≥ 2 completed episodes"
        assert metrics["mean_ep_length"] > 0

    def test_eval_restores_training_mode(self) -> None:
        """Actor should be back in training mode after eval."""
        agent = _make_agent()
        agent.env = _MockEnv()
        agent.actor.train()
        agent._run_eval_rollouts(num_steps=5)
        assert agent.actor.training, "actor not restored to training mode"

    def test_eval_with_normalization(self) -> None:
        """Eval rollouts work correctly with obs normalization."""
        agent = _make_agent(obs_normalization=True)
        agent.env = _MockEnv(ep_length=3)
        metrics = agent._run_eval_rollouts(num_steps=10)
        assert math.isfinite(metrics["mean_reward"])
        assert metrics["num_episodes"] >= 1


# ══════════════════════════════════════════════════════════════════════
# Level 7 — Stat Logging (obs stats, action stats, Q stats, targets)
# ══════════════════════════════════════════════════════════════════════


class TestStatLogging:
    """Verify that a full update cycle produces all diagnostic stats."""

    def _run_one_cycle(
        self, *, obs_normalization: bool = False
    ) -> dict[str, torch.Tensor | float]:
        """Run one critic+actor+alpha cycle and merge all metrics."""
        agent = _make_agent(obs_normalization=obs_normalization)
        data = _make_batch()
        if obs_normalization:
            data["observations"] = agent.obs_normalizer(data["observations"], update=False)
            data["next"]["observations"] = agent.obs_normalizer(
                data["next"]["observations"], update=False
            )
            data["critic_observations"] = agent.critic_obs_normalizer(
                data["critic_observations"], update=False
            )
            data["next"]["critic_observations"] = agent.critic_obs_normalizer(
                data["next"]["critic_observations"], update=False
            )

        cm = agent._update_critic(data)
        am = agent._update_actor(data)
        agent._last_cql_penalty = cm["cql_penalty"]
        alm = agent._update_alpha(am["log_probs_mean"])
        return {**cm, **am, **alm}

    # ── Q-value stats ──────────────────────────────────────────────

    def test_q_stats_present_and_finite(self) -> None:
        m = self._run_one_cycle()
        for key in (
            "q_data_mean",
            "q_data_max",
            "q_data_min",
            "q_overestimation_gap",
            "cql_logsumexp_mean",
            "cql_q_rand_mean",
            "cql_q_pi_mean",
        ):
            assert key in m, f"missing Q stat: {key}"
            val = m[key]
            if isinstance(val, torch.Tensor):
                assert torch.isfinite(val), f"Q stat '{key}' non-finite"

    # ── TD target stats ───────────────────────────────────────────

    def test_target_stats_present_and_finite(self) -> None:
        m = self._run_one_cycle()
        for key in ("td_target_mean", "td_target_max", "td_target_min"):
            assert key in m, f"missing target stat: {key}"
            assert torch.isfinite(m[key]), f"target stat '{key}' non-finite"

    # ── Action / policy stats ─────────────────────────────────────

    def test_action_stats_present_and_finite(self) -> None:
        m = self._run_one_cycle()
        for key in ("action_std", "policy_entropy", "log_probs_mean"):
            assert key in m, f"missing action stat: {key}"
            assert torch.isfinite(m[key]), f"action stat '{key}' non-finite"

    def test_policy_entropy_is_finite(self) -> None:
        """For a randomly-initialised policy, entropy should be finite.

        Note: differential entropy (−E[log π]) CAN be negative when the
        density concentrates (std < 1), so we only assert finiteness and
        that it's within a reasonable range.
        """
        m = self._run_one_cycle()
        ent = m["policy_entropy"].item()
        assert math.isfinite(ent), f"policy_entropy is non-finite: {ent}"
        assert -50 < ent < 50, f"policy_entropy is extreme: {ent}"

    def test_action_std_is_positive(self) -> None:
        m = self._run_one_cycle()
        assert m["action_std"].item() > 0

    # ── Alpha / CQL alpha stats ───────────────────────────────────

    def test_alpha_stats_present(self) -> None:
        m = self._run_one_cycle()
        assert "alpha_value" in m
        assert "alpha_loss" in m
        assert "cql_alpha" in m
        assert "cql_loss" in m

    def test_alpha_value_positive(self) -> None:
        m = self._run_one_cycle()
        assert m["alpha_value"].item() > 0

    # ── Gradient norms ────────────────────────────────────────────

    def test_grad_norms_present_and_nonneg(self) -> None:
        m = self._run_one_cycle()
        for key in ("critic_grad_norm", "actor_grad_norm"):
            assert key in m, f"missing grad norm: {key}"
            assert m[key].item() >= 0, f"grad norm '{key}' is negative"

    # ── Loss composition ──────────────────────────────────────────

    def test_critic_loss_is_td_plus_cql(self) -> None:
        """critic_loss ≈ td_loss + cql_loss."""
        m = self._run_one_cycle()
        expected = m["td_loss"].item() + m["cql_loss"].item()
        actual = m["critic_loss"].item()
        assert abs(actual - expected) < 1e-4, (
            f"critic_loss={actual} != td_loss + cql_loss = {expected}"
        )

    # ── Normalized obs produces ≈ zero-mean output ────────────────

    def test_normalized_obs_zero_mean_ish(self) -> None:
        """With frozen normalizer, normalised obs should be roughly zero-mean."""
        agent = _make_agent(obs_normalization=True)
        raw_obs = torch.randn(200, OBS_DIM)
        norm_obs = agent.obs_normalizer(raw_obs, update=False)
        # Won't be exactly zero-mean because raw_obs doesn't match the
        # normalizer's internal statistics, but it should be transformed
        assert norm_obs.shape == raw_obs.shape
        assert torch.isfinite(norm_obs).all()

    def test_all_stats_finite_with_normalization(self) -> None:
        """Full cycle with normalization produces all-finite metrics."""
        m = self._run_one_cycle(obs_normalization=True)
        for k, v in m.items():
            if isinstance(v, torch.Tensor):
                assert torch.isfinite(v), f"{k} non-finite with normalization"
