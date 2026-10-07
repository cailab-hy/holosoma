from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch

from holosoma.agents.aw_cql.aw_cql_agent import AWCQLAgent
from holosoma.agents.cql.cql_agent import CQLAgent
from holosoma.agents.hong_advantage_cql.hong_advantage_cql_agent import (
    HongAdvantageCQLAgent,
    episode_bounds_from_h5,
    fit_linear_value,
    hong_transition_probabilities,
)
from holosoma.config_values.algo import DEFAULTS as ALGO_DEFAULTS
from holosoma.config_values.experiment import DEFAULTS

TARGET = "holosoma.agents.hong_advantage_cql.hong_advantage_cql_agent.HongAdvantageCQLAgent"
HONG_FIELDS = {"hong_aw_enabled", "hong_aw_temperature", "hong_aw_eps", "hong_aw_obs_key"}


def test_hong_cql_is_plain_cql_objective() -> None:
    assert issubclass(HongAdvantageCQLAgent, CQLAgent)
    assert not issubclass(HongAdvantageCQLAgent, AWCQLAgent)
    # No loss / update hook is overridden: the CQL objective is inherited unchanged.
    for hook in (
        "_build_cql_per_sample_losses",
        "_transform_cql_per_sample_losses",
        "_build_sampled_conservative_losses",
        "_after_q_update",
        "_update_q",
        "_update_cql_lagrange",
        "_update_actor",
        "_sample_offline_batch",
    ):
        assert getattr(HongAdvantageCQLAgent, hook) is getattr(CQLAgent, hook), hook


def test_episode_bounds_and_returns() -> None:
    dones = np.array([0, 0, 1, 0, 0, 0, 0])
    truncs = np.array([0, 0, 0, 0, 1, 0, 0])
    starts, ends = episode_bounds_from_h5(dones, truncs)
    np.testing.assert_array_equal(starts, [0, 3, 5])
    np.testing.assert_array_equal(ends, [2, 4, 6])  # last row always closes an episode


def test_linear_value_fit_handles_rank_deficient_states() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 30))
    x[:, 3] = 2.0 * x[:, 2]  # collinear
    x[:, 7] = 0.764  # constant
    w = rng.standard_normal(30)
    y = x @ w + 1.5 + 0.01 * rng.standard_normal(2000)
    _, _, pred = fit_linear_value(x, y)
    r2 = 1.0 - np.square(y - pred).sum() / np.square(y - y.mean()).sum()
    assert r2 > 0.999


def test_hong_probabilities() -> None:
    g = np.array([1.0, 5.0, 3.0])
    v0 = np.array([1.0, 1.0, 1.0])
    lengths = np.array([2, 3, 1])
    adv, adv_norm, priority, prob = hong_transition_probabilities(g, v0, lengths, temperature=0.2, eps=1e-8)
    np.testing.assert_allclose(adv, [0.0, 4.0, 2.0])
    np.testing.assert_allclose(adv_norm, [0.0, 1.0, 0.5], atol=1e-7)
    np.testing.assert_allclose(priority, np.exp(adv_norm / 0.2))
    assert prob.shape == (6,)
    assert abs(prob.sum() - 1.0) < 1e-12
    # all transitions of an episode share q_i; p = q_i / sum_j T_j q_j
    np.testing.assert_allclose(prob[:2], priority[0] / (lengths * priority).sum())
    np.testing.assert_allclose(prob[2:5], priority[1] / (lengths * priority).sum())
    assert prob[2] > prob[5] > prob[0]


def _write_h5(path: Path) -> None:
    rng = np.random.default_rng(0)
    lengths = [5, 3, 8, 4, 6, 2, 7, 5]
    n = sum(lengths)
    dones = np.zeros(n, bool)
    dones[np.cumsum(lengths) - 1] = True
    with h5py.File(path, "w") as f:
        f["observations"] = rng.standard_normal((n, 4)).astype(np.float32)
        f["critic_observations"] = rng.standard_normal((n, 6)).astype(np.float32)
        f["rewards"] = rng.uniform(0, 1, n).astype(np.float32)
        f["dones"] = dones
        f["truncations"] = np.zeros(n, bool)


def test_sampler_follows_hong_distribution(tmp_path: Path) -> None:
    h5 = tmp_path / "d.h5"
    _write_h5(h5)
    n = 40
    agent = HongAdvantageCQLAgent.__new__(HongAdvantageCQLAgent)
    agent.config = ALGO_DEFAULTS["hong_advantage_cql"].config
    agent._offline_dataset_path = h5
    agent._offline_num_samples = n
    agent.device = "cpu"
    prob, stats = agent._compute_hong_probabilities()
    assert stats["num_episodes"] == 8 and 0 < stats["ess_ratio"] <= 1

    class _Cache:
        _storage = {"rewards": torch.arange(n, dtype=torch.float32)}

        def sample(self, batch_size):  # uniform draw that must be replaced
            raise AssertionError("uniform sampler used")

        def close(self):
            pass

    agent._offline_gpu_cache = _Cache()
    agent._hong_prob = torch.as_tensor(prob, dtype=torch.float64)
    agent._hong_generator = torch.Generator().manual_seed(0)
    agent._install_weighted_sampler()
    batch = agent._offline_gpu_cache.sample(200_000)
    idx = batch["dataset_index"]
    torch.testing.assert_close(batch["rewards"], idx.float())  # rows match their dataset index
    assert set(batch) == {"rewards", "dataset_index"}  # no weight attached to the batch
    freq = np.bincount(idx.numpy(), minlength=n) / idx.numel()
    np.testing.assert_allclose(freq, prob, atol=3e-3)


def test_hong_cql_registered_and_paired_with_cql() -> None:
    assert ALGO_DEFAULTS["hong_advantage_cql"]._target_ == TARGET
    assert ALGO_DEFAULTS["hong_advantage_cql"].config.hong_aw_temperature == 0.2
    for name in (
        "g1_29dof_wbt_hong_advantage_cql",
        "g1_29dof_wbt_lafan_dance1_hong_advantage_cql",
        "g1_29dof_wbt_lafan_kick_hong_advantage_cql",
        "g1_29dof_wbt_lafan_kick2_hong_advantage_cql",
    ):
        hong = DEFAULTS[name]
        cql = DEFAULTS[name.replace("hong_advantage_cql", "cql")]
        assert hong.algo._target_ == TARGET
        hong_cfg = asdict(hong.algo.config)
        assert set(hong_cfg) - set(asdict(cql.algo.config)) == HONG_FIELDS
        assert {k: v for k, v in hong_cfg.items() if k not in HONG_FIELDS} == asdict(cql.algo.config)
        assert hong.reward == cql.reward and hong.command == cql.command and hong.observation == cql.observation
