from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from holosoma.agents.cql.cql import Actor, DoubleQCritic
from holosoma.agents.cql.cql_utils import EmpiricalNormalization
from holosoma.agents.modules.q_probe import PROBE_KEYS, compute_q_probe_stats, load_probe_set, select_probe_indices

OBS, COBS, ACT = 6, 8, 3


def _indices(dim: int) -> dict[str, dict[str, int]]:
    return {"all": {"start": 0, "end": dim, "size": dim}}


def _models(seed: int = 0):
    torch.manual_seed(seed)
    actor = Actor(_indices(OBS), ["all"], n_act=ACT, num_envs=1, hidden_dim=16, log_std_max=2.0, log_std_min=-5.0)
    # non-trivial policy mean / std
    torch.nn.init.normal_(actor.fc_mu[0].weight, std=0.5)
    torch.nn.init.normal_(actor.fc_logstd.weight, std=0.1)
    qnet = DoubleQCritic(_indices(COBS), ["all"], n_act=ACT, hidden_dim=16)
    return actor, qnet


def _probe(n: int = 64, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "observations": torch.randn(n, OBS, generator=g),
        "next_observations": torch.randn(n, OBS, generator=g),
        "critic_observations": torch.randn(n, COBS, generator=g),
        "actions": torch.rand(n, ACT, generator=g) * 2 - 1,
    }


def _stats(actor, qnet, probe, obs_norm=None, cobs_norm=None, seed: int = 7):
    return compute_q_probe_stats(
        qnet,
        actor,
        obs_norm,
        cobs_norm,
        probe,
        num_action_samples=4,
        temperature=1.0,
        use_tanh=True,
        seed=seed,
        chunk_size=20,
    )


def test_select_probe_indices_is_deterministic_sorted_unique() -> None:
    a = select_probe_indices(1000, 100, seed=3)
    b = select_probe_indices(1000, 100, seed=3)
    assert np.array_equal(a, b)
    assert np.all(np.diff(a) > 0)
    assert a.size == 100 and a.max() < 1000
    assert not np.array_equal(a, select_probe_indices(1000, 100, seed=4))
    assert select_probe_indices(10, 100, seed=0).size == 10


def test_load_probe_set_reads_selected_rows(tmp_path) -> None:
    n = 50
    rng = np.random.default_rng(0)
    data = {
        "observations": rng.standard_normal((n, OBS)).astype(np.float32),
        "next_observations": rng.standard_normal((n, OBS)).astype(np.float32),
        "critic_observations": rng.standard_normal((n, COBS)).astype(np.float32),
        "actions": rng.standard_normal((n, ACT)).astype(np.float32),
    }
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)
    idx = select_probe_indices(n, 10, seed=1)
    probe = load_probe_set(path, idx, "cpu")
    assert set(probe) == set(PROBE_KEYS)
    for key in PROBE_KEYS:
        np.testing.assert_allclose(probe[key].numpy(), data[key][idx])
    with pytest.raises(ValueError, match="increasing"):
        load_probe_set(path, np.array([3, 1]), "cpu")


def test_compute_q_probe_stats_keys_deterministic_and_chunk_invariant() -> None:
    actor, qnet = _models()
    probe = _probe()
    s1 = _stats(actor, qnet, probe)
    s2 = _stats(actor, qnet, probe)
    assert set(s1) == {
        "probe/q_level",
        "probe/q_lse",
        "probe/lse_minus_q_data",
        "probe/q_pi",
        "probe/q_curr",
        "probe/q_rand",
    }
    for key in s1:
        assert torch.isfinite(s1[key])
        torch.testing.assert_close(s1[key], s2[key])
    # LSE dominates the mean of the twin critics at the dataset action only up to importance
    # correction, but the unweighted bracket must be LSE minus the twin-mean Q_D exactly.
    q1, q2 = qnet(probe["critic_observations"], probe["actions"])
    torch.testing.assert_close(s1["probe/lse_minus_q_data"], s1["probe/q_lse"] - (0.5 * (q1 + q2)).mean())
    s3 = compute_q_probe_stats(
        qnet, actor, None, None, probe, num_action_samples=4, temperature=1.0, use_tanh=True, seed=7, chunk_size=64
    )
    # different chunking draws different random actions, but the dataset-side levels are exact
    torch.testing.assert_close(s1["probe/q_level"], s3["probe/q_level"])
    torch.testing.assert_close(s1["probe/q_pi"], s3["probe/q_pi"])


def test_uniform_critic_shift_moves_levels_not_bracket() -> None:
    actor, qnet = _models()
    probe = _probe()
    base = _stats(actor, qnet, probe)
    shift = 3.0
    with torch.no_grad():
        qnet.q1.net[-1].bias += shift
        qnet.q2.net[-1].bias += shift
    shifted = _stats(actor, qnet, probe)
    for key in ("probe/q_level", "probe/q_lse", "probe/q_pi", "probe/q_curr", "probe/q_rand"):
        torch.testing.assert_close(shifted[key] - base[key], torch.tensor(shift))
    torch.testing.assert_close(shifted["probe/lse_minus_q_data"], base["probe/lse_minus_q_data"])


def test_normalizers_are_applied_without_update() -> None:
    actor, qnet = _models()
    probe = _probe()
    obs_norm = EmpiricalNormalization(shape=OBS, device="cpu")
    cobs_norm = EmpiricalNormalization(shape=COBS, device="cpu")
    obs_norm.train()
    cobs_norm.train()
    with torch.no_grad():
        obs_norm.update(torch.randn(100, OBS) * 5 + 1)
        cobs_norm.update(torch.randn(100, COBS) * 5 + 1)
    count_before = int(cobs_norm.count)
    mean_before = cobs_norm._mean.clone()
    with_norm = _stats(actor, qnet, probe, obs_norm, cobs_norm)
    assert int(cobs_norm.count) == count_before
    torch.testing.assert_close(cobs_norm._mean, mean_before)
    without_norm = _stats(actor, qnet, probe)
    assert not torch.allclose(with_norm["probe/q_level"], without_norm["probe/q_level"])
