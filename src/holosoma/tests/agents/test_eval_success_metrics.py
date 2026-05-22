"""Unit tests for eval_utils.py success-rate measurement (Step Eval-1).

These tests run entirely without a simulator by directly exercising the
internal logic of ``run_evaluate_policy`` via a lightweight stub that
replays a fixed episode schedule.

Synthetic scenario (4 envs):
    env0  → succeeds   (clip_ended at step 5)
    env1  → bad_tracking (done at step 3)
    env2  → timeout     (done at step 4)
    env3  → never finishes before max_eval_steps

Expected results:
    success_count                  = 1
    envs_finished                  = 3
    envs_unfinished                = 1
    success_rate                   = 1 / 4 = 0.25
    success_rate_attempted         = 0.25
    success_rate_finished          = 1 / 3 ≈ 0.3333
    bad_tracking_count             = 1
    timeout_count                  = 1
    max_eval_steps_unfinished_count = 1
    per_env_reason[3]              = "max_eval_steps_unfinished"
    per_env_success[3]             = False
    success_count cannot be > 1    (no double-counting)
"""
from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Helpers to build a minimal agent + env stub
# ---------------------------------------------------------------------------

def _make_motion_cmd_stub(env, clip_end_mask_at_step: dict[int, list[int]]):
    """Return a MotionCommand-like stub that fires clip_ended for given envs
    at specific steps (simulating a looping motion clip).

    ``clip_end_mask_at_step`` maps step → list of env indices that wrap.
    The stub exposes a ``time_steps`` tensor that decreases from a high value
    to 0 for the specified envs at each step, triggering clip_ended detection.
    """
    time_steps = torch.ones(env.num_envs, dtype=torch.float32) * 100.0

    class _Stub:
        pass

    stub = _Stub()
    stub.time_steps = time_steps
    stub._clip_end_map = clip_end_mask_at_step
    stub._current_step = [0]

    _orig_clone = torch.Tensor.clone

    def advance(step: int):
        """Simulate motion clip wrapping for scheduled envs."""
        stub._current_step[0] = step
        # Reset to high value by default (no event)
        stub.time_steps = torch.ones(env.num_envs, dtype=torch.float32) * 100.0
        if step in stub._clip_end_map:
            for i in stub._clip_end_map[step]:
                # time_steps drops below previous value → clip_ended fires
                stub.time_steps[i] = 0.0

    stub.advance = advance
    return stub


def _build_agent_and_env(
    num_envs: int,
    done_at: dict[int, list[tuple[int, str]]],  # step → [(env_idx, reason_name)]
    clip_end_at: dict[int, list[int]],           # step → [env_idx] (success)
    max_steps: int = 10,
):
    """Build minimal mocked agent + env for eval_utils testing.

    Parameters
    ----------
    num_envs:
        Number of parallel envs.
    done_at:
        Maps a step index to a list of (env_idx, reason_name) tuples that
        should have ``dones[env_idx] = True`` at that step with the given
        termination reason active in ``term_mgr.active_terms``.
    clip_end_at:
        Maps a step index to env indices whose motion clip wraps (success).
    max_steps:
        ``max_eval_steps`` passed to ``run_evaluate_policy``.
    """
    # -- env stub ----------------------------------------------------------------
    env = MagicMock()
    env.num_envs = num_envs
    device = torch.device("cpu")

    obs_dim = 4

    def _reset():
        return torch.zeros(num_envs, obs_dim)

    env.reset.side_effect = _reset

    # Mutable state for the step function
    _step_count = [0]

    def _step(actions):
        step = _step_count[0]
        _step_count[0] += 1

        obs = torch.zeros(num_envs, obs_dim)
        rewards = torch.ones(num_envs)  # reward = 1 per step
        dones = torch.zeros(num_envs, dtype=torch.bool)

        current_done = done_at.get(step, [])
        active_terms_per_env: dict[int, str] = {}
        for env_idx, reason in current_done:
            dones[env_idx] = True
            active_terms_per_env[env_idx] = reason

        extras: dict[str, Any] = {}
        return obs, rewards, dones, extras, active_terms_per_env

    # Termination manager stub
    term_mgr_stub = MagicMock()

    class _TermMgr:
        def __init__(self):
            self._active: dict[int, str] = {}

        @property
        def active_terms(self):
            # Return dict[term_name → bool_mask] matching the current done set
            terms: dict[str, Any] = {}
            for env_idx, reason in self._active.items():
                if reason not in terms:
                    terms[reason] = torch.zeros(num_envs, dtype=torch.bool)
                terms[reason][env_idx] = True
            return terms

    _term_mgr = _TermMgr()

    # Wrap env._env
    env_inner = MagicMock()
    env_inner.termination_manager = _term_mgr
    env._env = env_inner

    # Build motion cmd stub
    motion_cmd_stub = _make_motion_cmd_stub(env, clip_end_at)

    # Override env.step to also advance the motion stub and term_mgr
    def _patched_step(actions):
        step = _step_count[0]
        _step_count[0] += 1

        obs = torch.zeros(num_envs, obs_dim)
        rewards = torch.ones(num_envs)
        dones = torch.zeros(num_envs, dtype=torch.bool)

        current_done = done_at.get(step, [])
        _term_mgr._active = {}
        for env_idx, reason in current_done:
            dones[env_idx] = True
            _term_mgr._active[env_idx] = reason

        # Advance motion clip state for this step
        motion_cmd_stub.advance(step)

        extras: dict[str, Any] = {}
        return obs, rewards, dones, extras

    env.step.side_effect = _patched_step

    # -- actor stub --------------------------------------------------------------
    actor = MagicMock()
    actor.training = False

    def _actor_forward(obs):
        actions = torch.zeros(num_envs, 2)
        return actions, torch.zeros(num_envs, 2), torch.zeros(num_envs, 2)

    actor.side_effect = _actor_forward

    # -- agent stub --------------------------------------------------------------
    agent = MagicMock()
    agent.device = device
    agent.env = env
    agent.actor = actor
    agent.obs_normalization = False
    agent.eval_callbacks = []
    agent.config = MagicMock()
    agent.config.eval_callbacks = None
    # _single_episode_per_env defaults to False; tests override as needed
    agent._single_episode_per_env = False

    # unwrapped_env.command_manager.get_state("motion_command") → motion_cmd_stub
    cmd_mgr = MagicMock()
    cmd_mgr.get_state.return_value = motion_cmd_stub
    inner_env = MagicMock()
    inner_env.command_manager = cmd_mgr
    agent.unwrapped_env = inner_env

    return agent, env, motion_cmd_stub, max_steps


# ---------------------------------------------------------------------------
# Shared scenario builder (4-env canonical example)
# ---------------------------------------------------------------------------

def _build_canonical_scenario(single_episode_per_env: bool):
    """
    env0: clip wraps at step 5  → success
    env1: done at step 3, reason=bad_tracking
    env2: done at step 4, reason=timeout
    env3: never done before max_steps=8
    """
    agent, env, motion_cmd_stub, max_steps = _build_agent_and_env(
        num_envs=4,
        done_at={
            3: [(1, "bad_tracking")],
            4: [(2, "timeout")],
        },
        clip_end_at={5: [0]},
        max_steps=8,
    )
    agent._single_episode_per_env = single_episode_per_env
    return agent, max_steps


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvalSuccessMetrics:
    """Success-rate metric tests for run_evaluate_policy."""

    def _run(self, single_episode_per_env: bool):
        from holosoma.agents.offline_rl.common.eval_utils import run_evaluate_policy

        agent, max_steps = _build_canonical_scenario(single_episode_per_env)

        # Patch out callback helpers (no-ops)
        with (
            patch(
                "holosoma.agents.offline_rl.common.eval_utils.create_eval_callbacks",
                return_value=None,
            ),
            patch(
                "holosoma.agents.offline_rl.common.eval_utils.pre_evaluate_policy",
                return_value=None,
            ),
            patch(
                "holosoma.agents.offline_rl.common.eval_utils.post_evaluate_policy",
                return_value=None,
            ),
            patch(
                "holosoma.agents.offline_rl.common.eval_utils.pre_eval_env_step",
                side_effect=lambda a, s: s,
            ),
            patch(
                "holosoma.agents.offline_rl.common.eval_utils.post_eval_env_step",
                side_effect=lambda a, s: s,
            ),
        ):
            run_evaluate_policy(agent, max_eval_steps=max_steps)

        stats = getattr(agent, "_last_per_env_stats", None)
        assert stats is not None, "_last_per_env_stats must always be set"
        return stats

    # -- single_episode_per_env=True ------------------------------------------

    def test_single_mode_stats_populated(self):
        stats = self._run(single_episode_per_env=True)
        assert stats is not None

    def test_single_mode_success_count(self):
        stats = self._run(single_episode_per_env=True)
        assert stats["success_count"] == 1

    def test_single_mode_envs_finished(self):
        stats = self._run(single_episode_per_env=True)
        assert stats["envs_finished"] == 3

    def test_single_mode_envs_unfinished(self):
        stats = self._run(single_episode_per_env=True)
        assert stats["envs_unfinished"] == 1

    def test_single_mode_success_rate_uses_num_envs(self):
        stats = self._run(single_episode_per_env=True)
        assert stats["success_rate"] == pytest.approx(1 / 4, abs=1e-6)

    def test_single_mode_success_rate_attempted_equals_success_rate(self):
        stats = self._run(single_episode_per_env=True)
        assert stats["success_rate_attempted"] == pytest.approx(stats["success_rate"], abs=1e-9)

    def test_single_mode_success_rate_finished(self):
        stats = self._run(single_episode_per_env=True)
        assert stats["success_rate_finished"] == pytest.approx(1 / 3, abs=1e-6)

    def test_single_mode_bad_tracking_count(self):
        stats = self._run(single_episode_per_env=True)
        assert stats["bad_tracking_count"] == 1

    def test_single_mode_timeout_count(self):
        stats = self._run(single_episode_per_env=True)
        assert stats["timeout_count"] == 1

    def test_single_mode_max_eval_steps_unfinished_count(self):
        stats = self._run(single_episode_per_env=True)
        assert stats["max_eval_steps_unfinished_count"] == 1

    def test_single_mode_unfinished_reason(self):
        stats = self._run(single_episode_per_env=True)
        # env3 is the unfinished env
        assert stats["reason"][3] == "max_eval_steps_unfinished"

    def test_single_mode_unfinished_success_false(self):
        stats = self._run(single_episode_per_env=True)
        assert stats["success"][3] is False or stats["success"][3] == 0

    def test_single_mode_denominator_annotation(self):
        stats = self._run(single_episode_per_env=True)
        assert stats["success_rate_denominator"] == "num_envs"
        assert stats["success_rate_finished_denominator"] == "envs_finished"
        assert stats["unfinished_envs_counted_as_failure"] is True

    def test_single_mode_no_double_counting(self):
        """success_count must not exceed 1 even if multiple steps fire for env0."""
        stats = self._run(single_episode_per_env=True)
        assert stats["success_count"] <= 1

    # -- single_episode_per_env=False -----------------------------------------

    def test_default_mode_stats_populated(self):
        """_last_per_env_stats must be set in default (False) mode too."""
        stats = self._run(single_episode_per_env=False)
        assert stats is not None

    def test_default_mode_success_count(self):
        stats = self._run(single_episode_per_env=False)
        assert stats["success_count"] == 1

    def test_default_mode_success_rate(self):
        stats = self._run(single_episode_per_env=False)
        assert stats["success_rate"] == pytest.approx(1 / 4, abs=1e-6)

    def test_default_mode_envs_unfinished(self):
        stats = self._run(single_episode_per_env=False)
        assert stats["envs_unfinished"] == 1

    def test_default_mode_unfinished_reason(self):
        stats = self._run(single_episode_per_env=False)
        assert stats["reason"][3] == "max_eval_steps_unfinished"

    def test_default_mode_unfinished_counted_as_failure(self):
        stats = self._run(single_episode_per_env=False)
        assert stats["unfinished_envs_counted_as_failure"] is True

    # -- failure reason consistency -------------------------------------------

    def test_failure_reason_counts_keys(self):
        stats = self._run(single_episode_per_env=True)
        counts = stats["failure_reason_counts"]
        assert "bad_tracking" in counts
        assert "timeout" in counts
        assert "max_eval_steps_unfinished" in counts
        assert "success" in counts

    def test_failure_reason_counts_sum(self):
        stats = self._run(single_episode_per_env=True)
        counts = stats["failure_reason_counts"]
        assert sum(counts.values()) == 4  # one entry per env

    # -- clip_ended success path unchanged ------------------------------------

    def test_clip_ended_maps_to_success(self):
        """env0 finishes via clip_ended; its reason must be 'success'."""
        stats = self._run(single_episode_per_env=True)
        assert stats["reason"][0] == "success"
        assert stats["success"][0] is True or stats["success"][0] == 1

    def test_bad_tracking_maps_to_failure(self):
        stats = self._run(single_episode_per_env=True)
        assert stats["reason"][1] == "bad_tracking"
        assert not (stats["success"][1])

    def test_timeout_maps_to_failure(self):
        stats = self._run(single_episode_per_env=True)
        assert stats["reason"][2] == "timeout"
        assert not (stats["success"][2])
