from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from holosoma.agents.asym_cql.asym_cql_agent import AsymCQLAgent
from holosoma.config_values.experiment import DEFAULTS
from scripts import asym_precompute_weights as asym
from scripts import aw_precompute_weights as aw_precompute
from scripts.aw_precompute_weights import make_weights


def test_asym_cql_places_independent_weights_on_the_two_terms() -> None:
    plus = torch.tensor([0.5, 2.0])
    minus = torch.tensor([1.5, 0.25])
    q1_lse = torch.tensor([3.0, 5.0])
    q2_lse = torch.tensor([4.0, 6.0])
    q1_data = torch.tensor([1.0, 2.0])
    q2_data = torch.tensor([2.0, 3.0])
    agent = AsymCQLAgent.__new__(AsymCQLAgent)
    agent._aw_batch_weight = plus
    agent._asym_batch_minus_weight = minus
    agent._asym_last_metrics = {}

    q1_loss, q2_loss = agent._build_cql_per_sample_losses(q1_lse, q2_lse, q1_data, q2_data)
    q1_loss, q2_loss = agent._transform_cql_per_sample_losses(q1_loss, q2_loss)

    torch.testing.assert_close(q1_loss, minus * q1_lse - plus * q1_data)
    torch.testing.assert_close(q2_loss, minus * q2_lse - plus * q2_data)


def test_asym_precompute_preserves_aw_positive_formula_and_normalizes_both(tmp_path: Path) -> None:
    h5_path = tmp_path / "dataset.h5"
    rewards = np.asarray([1.0, -0.2, 0.4, 1.3, -0.7, 0.8], dtype=np.float32)
    with h5py.File(h5_path, "w") as h5_file:
        h5_file["rewards"] = rewards
        h5_file["motion_phase"] = np.asarray([0.0, 0.2, 0.4, 0.0, 0.2, 0.4], dtype=np.float32)
        h5_file["dones"] = np.asarray([0, 0, 1, 0, 0, 1], dtype=np.uint8)
        h5_file["truncations"] = np.zeros(6, dtype=np.uint8)
    out = tmp_path / "asym.npz"
    assert aw_precompute.main([str(h5_path), "--H", "2", "--n-bins", "3"]) in {0, 2}
    assert asym.main([str(h5_path), "--out", str(out)]) == 0

    with np.load(out) as sidecar:
        with np.load(f"{h5_path}.aw_weights.npz") as aw_sidecar:
            expected_plus = aw_sidecar["weight"]
        np.testing.assert_array_equal(sidecar["weight"], sidecar["weight_plus"])
        np.testing.assert_array_equal(sidecar["weight_plus"], expected_plus)
        np.testing.assert_allclose(sidecar["weight_plus"].mean(), 1.0, atol=1e-6)
        np.testing.assert_allclose(sidecar["weight_minus"].mean(), 1.0, atol=1e-6)
        np.testing.assert_array_equal(np.sort(sidecar["weight_plus"]), np.sort(sidecar["weight_minus"]))
        assert str(sidecar["minus_mode"]) == "mirror"
        order = np.argsort(sidecar["advantage"], kind="stable")
        assert np.all(np.diff(sidecar["weight_minus"][order]) <= 0.0)


def test_mirrored_weights_reverse_only_the_assignment() -> None:
    advantage = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    weight_plus = np.asarray([0.2, 0.5, 0.8, 1.3, 2.2], dtype=np.float32)
    expected = np.asarray([2.2, 1.3, 0.8, 0.5, 0.2], dtype=np.float32)
    np.testing.assert_array_equal(asym.make_mirrored_weights(advantage, weight_plus), expected)


def test_exp_minus_mode_remains_available() -> None:
    advantage = np.asarray([-1.0, 0.0, 2.0])
    expected, _ = make_weights(-advantage, beta=1.0, w_max=10.0)
    mirrored = asym.make_mirrored_weights(advantage, make_weights(advantage, 1.0, 10.0)[0])
    assert not np.array_equal(mirrored, expected)


def test_wbt_asym_experiment_is_registered_and_paired_with_aw() -> None:
    aw = DEFAULTS["g1_29dof_wbt_aw_cql"]
    asym_config = DEFAULTS["g1_29dof_wbt_asym_cql"]
    assert asym_config.algo._target_ == "holosoma.agents.asym_cql.asym_cql_agent.AsymCQLAgent"
    assert asym_config.algo.config.offline_dataset_path == aw.algo.config.offline_dataset_path
    assert asym_config.algo.config.asym_weights_path == ""
