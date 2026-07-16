from __future__ import annotations

import h5py
import numpy as np

from scripts.analyze_h5_phase_failure_hazard import analyze_phase_failure_hazard


def test_phase_failure_hazard_counts_unique_entered_episodes(tmp_path):
    path = tmp_path / "episode_data.h5"
    episode_id = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3], dtype=np.int64)
    phase = np.array([0.05, 0.15, 0.25, 0.05, 0.15, 0.35, 0.25, 0.35, 0.05], dtype=np.float32)
    bad = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1], dtype=np.uint8)
    done = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8)
    complete = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8)

    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["num_samples"] = episode_id.size
        h5_file.attrs["motion_phase_semantics"] = "pre_step"
        h5_file.create_dataset("episode_id", data=episode_id)
        h5_file.create_dataset("motion_phase", data=phase[:, None])
        h5_file.create_dataset("next_done_bad_tracking", data=bad)
        h5_file.create_dataset("dones", data=done)
        h5_file.create_dataset("episode_data_complete", data=complete)

    result = analyze_phase_failure_hazard(path, num_bins=4, chunk_size=3)

    np.testing.assert_array_equal(result.entered_episodes, np.array([2, 3, 0, 0]))
    np.testing.assert_array_equal(result.bad_tracking_failures, np.array([0, 2, 0, 0]))
    np.testing.assert_allclose(result.hazard[:2], np.array([0.0, 2.0 / 3.0]))
    assert np.isnan(result.hazard[2:]).all()
    assert result.total_episodes == 4
    assert result.analyzed_episodes == 3
    assert result.incomplete_episodes == 1
    assert result.bad_tracking_episodes == 2
    assert result.phase_semantics == "pre_step"


def test_legacy_post_step_phase_uses_preterminal_row(tmp_path):
    path = tmp_path / "legacy_episode_data.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["num_samples"] = 3
        h5_file.create_dataset("episode_id", data=np.zeros(3, dtype=np.int64))
        h5_file.create_dataset("motion_phase", data=np.array([0.05, 0.15, 0.90], dtype=np.float32))
        h5_file.create_dataset("next_done_bad_tracking", data=np.array([0, 0, 1], dtype=np.uint8))
        h5_file.create_dataset("dones", data=np.array([0, 0, 1], dtype=np.uint8))
        h5_file.create_dataset("episode_data_complete", data=np.ones(3, dtype=np.uint8))

    result = analyze_phase_failure_hazard(path, num_bins=4, chunk_size=2)

    np.testing.assert_array_equal(result.entered_episodes, np.array([1, 0, 0, 0]))
    np.testing.assert_array_equal(result.bad_tracking_failures, np.array([1, 0, 0, 0]))
    assert result.hazard[0] == 1.0
    assert result.phase_semantics == "legacy_post_step"
    assert result.nonmonotonic_phase_episodes == 0


def test_phase_failure_hazard_rejects_noncontiguous_episode_ids(tmp_path):
    path = tmp_path / "invalid_episode_data.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["num_samples"] = 3
        h5_file.create_dataset("episode_id", data=np.array([0, 1, 0], dtype=np.int64))
        h5_file.create_dataset("motion_phase", data=np.array([0.1, 0.2, 0.3], dtype=np.float32))
        h5_file.create_dataset("next_done_bad_tracking", data=np.zeros(3, dtype=np.uint8))
        h5_file.create_dataset("dones", data=np.ones(3, dtype=np.uint8))
        h5_file.create_dataset("episode_data_complete", data=np.ones(3, dtype=np.uint8))

    try:
        analyze_phase_failure_hazard(path, num_bins=4, chunk_size=2)
    except ValueError as error:
        assert "non-contiguous" in str(error)
    else:
        raise AssertionError("Expected non-contiguous episode IDs to be rejected.")
