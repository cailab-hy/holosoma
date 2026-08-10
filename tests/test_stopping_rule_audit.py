from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from aw_wall_probe import build_scorer
from holosoma.agents.cql.cql import Actor as CQLActor
from holosoma.agents.cql.cql import DoubleQCritic
from holosoma.agents.td3.td3 import Actor as TD3Actor
from stopping_rule_audit import ProbeData
from stopping_rule_audit import RunSpec
from stopping_rule_audit import _reward_hash
from stopping_rule_audit import discover_checkpoints
from stopping_rule_audit import load_fixed_probe_data
from stopping_rule_audit import load_level_log
from stopping_rule_audit import parse_window
from stopping_rule_audit import scan_stopping_rule


def _row(step: int, dhat: float, q_min: float) -> dict[str, float | int | str]:
    return {
        "step": step,
        "dhat_bin3": dhat,
        "span": 10.0,
        "q_min": q_min,
        "q_min_source": "test",
        "q_min_log_step": step,
    }


def test_contrast_requires_formed_peak_and_previous_above_tau() -> None:
    rows = [
        _row(5_000, 0.01, -10.0),
        _row(10_000, 0.11, -11.0),
        _row(20_000, 0.08, -9.0),
        _row(30_000, 0.04, -12.0),
    ]
    summary = scan_stopping_rule(
        rows,
        main_bin=3,
        tau=0.05,
        level_offset=-30.0,
        baseline_window=(5_000, 20_000),
    )

    assert summary["contrast_fire_step"] == 30_000
    assert summary["level_fire_step"] is None
    assert summary["t_fire"] == 30_000
    assert summary["adopted_ckpt"] == 20_000
    assert summary["gap"] == 10_000
    assert summary["warmup_peak_dhat"] == pytest.approx(0.11)
    assert [row["contrast_fired"] for row in rows] == [0, 0, 0, 1]


def test_level_channel_uses_early_median_and_can_fire_first() -> None:
    rows = [
        _row(5_000, 0.12, -10.0),
        _row(10_000, 0.11, -12.0),
        _row(20_000, 0.10, -11.0),
        _row(30_000, 0.09, -42.0),
    ]
    summary = scan_stopping_rule(
        rows,
        main_bin=3,
        tau=0.05,
        level_offset=-30.0,
        baseline_window=(5_000, 20_000),
    )

    assert summary["level_baseline"] == pytest.approx(-11.0)
    assert summary["level_threshold"] == pytest.approx(-41.0)
    assert summary["level_fire_step"] == 30_000
    assert summary["fired_channel"] == "level"
    assert summary["adopted_ckpt"] == 20_000


def test_no_fire_adopts_final_checkpoint() -> None:
    rows = [
        _row(5_000, 0.12, -10.0),
        _row(10_000, 0.11, -11.0),
        _row(20_000, 0.09, -12.0),
        _row(30_000, 0.08, -13.0),
    ]
    summary = scan_stopping_rule(
        rows,
        main_bin=3,
        tau=0.05,
        level_offset=-30.0,
        baseline_window=(5_000, 20_000),
    )

    assert summary["t_fire"] is None
    assert summary["fired_channel"] == "none"
    assert summary["adopted_ckpt"] == 30_000
    assert summary["gap"] is None


def test_fixed_probe_cache_is_reused_and_rhash_guarded(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cell.h5"
    rewards = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    critic_obs = np.arange(36, dtype=np.float32).reshape(12, 3)
    actions = np.arange(24, dtype=np.float32).reshape(12, 2)
    with h5py.File(dataset_path, "w") as h5:
        h5.create_dataset("rewards", data=rewards)
        h5.create_dataset("critic_observations", data=critic_obs)
        h5.create_dataset("actions", data=actions)
    with h5py.File(dataset_path, "r") as h5:
        reward_hash = _reward_hash(h5["rewards"])

    cache_path = tmp_path / "probe_rows.npz"
    cell_indices = np.empty(2, dtype=object)
    cell_indices[0] = np.array([1, 2], dtype=np.int64)
    cell_indices[1] = np.array([3, 4], dtype=np.int64)
    np.savez_compressed(
        cache_path,
        cell_keys=np.array([(3, "SURV"), (3, "FAIL")]),
        cell_idx=cell_indices,
        span_idx=np.array([0, 5, 11], dtype=np.int64),
        rhash=reward_hash,
    )
    spec = RunSpec(
        label="test",
        ckpt_dir=tmp_path,
        dataset=dataset_path,
        probe_rows=cache_path,
        wall_bins=(3,),
        agent_type="cql",
        tau=0.05,
        level_offset=-30.0,
        baseline_window=(5_000, 20_000),
        level_from_log=None,
        out=tmp_path / "out.csv",
        device="cpu",
        batch_size=32,
    )

    probe = load_fixed_probe_data(spec)
    assert isinstance(probe, ProbeData)
    np.testing.assert_array_equal(probe.span_actions, actions[[0, 5, 11]])
    np.testing.assert_array_equal(probe.cells[(3, "SURV")][1], actions[[1, 2]])

    with np.load(cache_path, allow_pickle=True) as cache:
        payload = {key: cache[key] for key in cache.files}
    payload["rhash"] = "wrong-hash"
    np.savez_compressed(cache_path, **payload)
    with pytest.raises(ValueError, match="rhash mismatch"):
        load_fixed_probe_data(spec)


def test_checkpoint_discovery_and_window_parser(tmp_path: Path) -> None:
    (tmp_path / "model_0001000.pt").touch()
    (tmp_path / "model_0010000.pt").touch()
    (tmp_path / "ignore.pt").touch()

    assert [step for step, _ in discover_checkpoints(tmp_path)] == [1_000, 10_000]
    assert parse_window("5k–20k") == (5_000, 20_000)


def test_level_log_prefers_q_target_min_column(tmp_path: Path) -> None:
    log_path = tmp_path / "level.csv"
    with log_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["global_step", "Loss/q_target_min"])
        writer.writeheader()
        writer.writerow({"global_step": 5_000, "Loss/q_target_min": -10.5})
        writer.writerow({"global_step": 6_000, "Loss/q_target_min": -11.5})

    assert load_level_log(log_path) == {5_000: -10.5, 6_000: -11.5}


def _obs_layout(size: int, key: str) -> dict[str, dict[str, int]]:
    return {key: {"start": 0, "end": size, "size": size}}


def _critic(obs_dim: int, action_dim: int) -> DoubleQCritic:
    return DoubleQCritic(
        obs_indices=_obs_layout(obs_dim, "critic_obs"),
        obs_keys=["critic_obs"],
        n_act=action_dim,
        hidden_dim=16,
        use_layer_norm=True,
        device="cpu",
    )


def test_build_scorer_can_return_target_twin_q(tmp_path: Path) -> None:
    actor = CQLActor(
        obs_indices=_obs_layout(3, "actor_obs"),
        obs_keys=["actor_obs"],
        n_act=2,
        num_envs=1,
        hidden_dim=16,
        log_std_max=0.0,
        log_std_min=-5.0,
        use_tanh=True,
        use_layer_norm=True,
        device="cpu",
    )
    critic = _critic(4, 2)
    target = _critic(4, 2)
    checkpoint = tmp_path / "model_0001000.pt"
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "qnet_state_dict": critic.state_dict(),
            "qnet_target_state_dict": target.state_dict(),
            "args": {"obs_normalization": False, "use_layer_norm": True},
        },
        checkpoint,
    )

    q_fn, pi_fn, target_q_fn = build_scorer(
        "cql",
        str(checkpoint),
        "cpu",
        include_target_q=True,
    )
    critic_obs = np.zeros((5, 4), dtype=np.float32)
    actor_obs = np.zeros((5, 3), dtype=np.float32)
    actions = np.zeros((5, 2), dtype=np.float32)

    assert q_fn(critic_obs, actions).shape == (5,)
    assert target_q_fn(critic_obs, actions).shape == (5,)
    assert pi_fn(actor_obs).shape == (5, 2)


def test_build_scorer_supports_td3bc_actor_layout(tmp_path: Path) -> None:
    actor = TD3Actor(
        obs_indices=_obs_layout(3, "actor_obs"),
        obs_keys=["actor_obs"],
        n_act=2,
        num_envs=1,
        hidden_dim=16,
        use_tanh=True,
        use_layer_norm=True,
        device="cpu",
    )
    critic = _critic(4, 2)
    target = _critic(4, 2)
    checkpoint = tmp_path / "model_0001000.pt"
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "qnet_state_dict": critic.state_dict(),
            "qnet_target_state_dict": target.state_dict(),
            "args": {"obs_normalization": False, "use_layer_norm": True},
        },
        checkpoint,
    )

    q_fn, pi_fn, target_q_fn = build_scorer(
        "td3bc",
        str(checkpoint),
        "cpu",
        include_target_q=True,
    )
    critic_obs = np.zeros((2, 4), dtype=np.float32)
    actor_obs = np.zeros((2, 3), dtype=np.float32)
    actions = np.zeros((2, 2), dtype=np.float32)

    assert q_fn(critic_obs, actions).shape == (2,)
    assert target_q_fn(critic_obs, actions).shape == (2,)
    assert pi_fn(actor_obs).shape == (2, 2)
