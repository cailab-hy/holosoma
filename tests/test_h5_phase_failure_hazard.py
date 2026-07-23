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


def test_age_filters_and_start_phase_cohorts(tmp_path):
    path = tmp_path / "cohort_episode_data.h5"
    episode_lengths = (5, 15, 25, 12)
    episode_id = np.concatenate(
        [np.full(length, episode_idx, dtype=np.int64) for episode_idx, length in enumerate(episode_lengths)]
    )
    phase = np.concatenate(
        (
            np.linspace(0.01, 0.20, episode_lengths[0], dtype=np.float32),
            np.linspace(0.00, 0.99, episode_lengths[1], dtype=np.float32),
            np.linspace(0.51, 0.99, episode_lengths[2], dtype=np.float32),
            np.linspace(0.30, 0.45, episode_lengths[3], dtype=np.float32),
        )
    )
    done = np.zeros(episode_id.size, dtype=np.uint8)
    terminal_rows = np.cumsum(episode_lengths) - 1
    done[terminal_rows] = 1
    bad = np.zeros_like(done)
    bad[terminal_rows[[0, 2]]] = 1
    motion_end = np.zeros_like(done)
    motion_end[terminal_rows[1]] = 1
    timeout = np.zeros_like(done)
    timeout[terminal_rows[3]] = 1

    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["num_samples"] = episode_id.size
        h5_file.attrs["motion_phase_semantics"] = "pre_step"
        h5_file.create_dataset("episode_id", data=episode_id)
        h5_file.create_dataset("motion_phase", data=phase)
        h5_file.create_dataset("next_done_bad_tracking", data=bad)
        h5_file.create_dataset("next_done_motion_ends", data=motion_end)
        h5_file.create_dataset("next_done_timeout", data=timeout)
        h5_file.create_dataset("dones", data=done)
        h5_file.create_dataset("episode_data_complete", data=np.ones_like(done))

    result = analyze_phase_failure_hazard(
        path,
        num_bins=4,
        exclude_episode_age_le=(10, 20),
        chunk_size=7,
    )

    np.testing.assert_array_equal(result.min_episode_ages, np.array([0, 10, 20]))
    np.testing.assert_array_equal(
        result.entered_by_min_episode_age,
        np.array(
            [
                [2, 2, 2, 2],
                [1, 2, 2, 2],
                [0, 0, 1, 1],
            ]
        ),
    )
    np.testing.assert_array_equal(
        result.bad_tracking_failures_by_min_episode_age,
        np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ]
        ),
    )
    np.testing.assert_array_equal(result.start_cohort_episodes, np.array([2, 1, 1, 0]))
    np.testing.assert_array_equal(result.start_cohort_passed_next_bin, np.array([1, 0, 1, 0]))
    np.testing.assert_array_equal(result.start_cohort_motion_ends, np.array([1, 0, 0, 0]))
    np.testing.assert_array_equal(result.start_cohort_timeouts, np.array([0, 1, 0, 0]))
    assert result.start_cohort_early_bad_tracking[1, 0] == 1
    assert result.start_cohort_early_bad_tracking[2, 0] == 1
    assert result.start_cohort_bad_failure_bins[2, 3] == 1
    assert result.exact_phase_zero_episodes == 1
    assert result.exact_phase_zero_motion_ends == 1
    assert result.exact_phase_zero_non_bad_reached_final == 1
    assert result.motion_end_key == "next_done_motion_ends"
    assert result.timeout_key == "next_done_timeout"
