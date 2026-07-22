from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from holosoma.agents.cql.cql import Actor, DoubleQCritic


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "aw_wall_bin_q_probe.py"
SPEC = importlib.util.spec_from_file_location("aw_wall_bin_q_probe", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


def _layout(size: int, key: str) -> dict[str, dict[str, int]]:
    return {key: {"start": 0, "end": size, "size": size}}


def test_measurement_c_episode_labels_match_wall_rule() -> None:
    phase_bins = np.array([4, 5, 5, 4, 5, 6])
    dones = np.array([False, False, True, False, False, True])
    truncations = np.zeros(6, dtype=bool)
    bad_tracking = np.array([False, False, True, False, False, False])

    fail_bin4, surv_bin4 = PROBE.measurement_c_episode_labels(
        phase_bins,
        dones,
        truncations,
        bad_tracking,
        bin_index=4,
        wall_span=1,
    )
    fail_bin5, surv_bin5 = PROBE.measurement_c_episode_labels(
        phase_bins,
        dones,
        truncations,
        bad_tracking,
        bin_index=5,
        wall_span=1,
    )

    np.testing.assert_array_equal(fail_bin4, np.array([0]))
    np.testing.assert_array_equal(surv_bin4, np.array([3]))
    np.testing.assert_array_equal(fail_bin5, np.array([1, 2]))
    np.testing.assert_array_equal(surv_bin5, np.array([4]))


def test_checkpoint_model_restore_and_probe_inference(tmp_path: Path) -> None:
    actor_obs_dim = 3
    critic_obs_dim = 4
    action_dim = 2
    actor = Actor(
        obs_indices=_layout(actor_obs_dim, "actor_obs"),
        obs_keys=["actor_obs"],
        n_act=action_dim,
        num_envs=1,
        hidden_dim=8,
        log_std_max=0.0,
        log_std_min=-5.0,
        use_tanh=True,
        use_layer_norm=True,
        device="cpu",
    )
    critic = DoubleQCritic(
        obs_indices=_layout(critic_obs_dim, "critic_obs"),
        obs_keys=["critic_obs"],
        n_act=action_dim,
        hidden_dim=8,
        use_layer_norm=True,
        device="cpu",
    )
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "qnet_state_dict": critic.state_dict(),
            "obs_normalizer_state": {
                "_mean": torch.zeros(1, actor_obs_dim),
                "_std": torch.ones(1, actor_obs_dim),
            },
            "critic_obs_normalizer_state": {
                "_mean": torch.zeros(1, critic_obs_dim),
                "_std": torch.ones(1, critic_obs_dim),
            },
            "args": {
                "obs_normalization": True,
                "use_layer_norm": True,
                "use_tanh": True,
                "log_std_max": 0.0,
                "log_std_min": -5.0,
                "use_cnn_encoder": False,
            },
        },
        checkpoint_path,
    )

    model = PROBE.load_probe_model(checkpoint_path, torch.device("cpu"))
    arrays = PROBE.CellArrays(
        observations=np.zeros((5, actor_obs_dim), dtype=np.float32),
        critic_observations=np.arange(20, dtype=np.float32).reshape(5, critic_obs_dim) / 20.0,
        actions=np.zeros((5, action_dim), dtype=np.float32),
    )
    q_values, drift = PROBE.infer_q_and_drift(model, arrays, torch.device("cpu"), batch_size=2)

    assert q_values.shape == (5,)
    assert drift.shape == (5,)
    assert np.isfinite(q_values).all()
    assert np.isfinite(drift).all()
