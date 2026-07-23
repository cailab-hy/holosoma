#!/usr/bin/env python3
"""Estimate phase-conditioned bad-tracking failure hazards from episode HDF5 data.

For phase bin k, the discrete hazard is

    h_k = bad-tracking episodes that terminate in bin k
          / complete episodes that enter bin k.

Episodes, rather than transitions, are counted in both terms. This keeps long
episodes from receiving more weight and correctly handles random-phase starts.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


@dataclass(frozen=True)
class PhaseFailureHazardResult:
    phase_semantics: str
    phase_edges: np.ndarray
    entered_episodes: np.ndarray
    bad_tracking_failures: np.ndarray
    min_episode_ages: np.ndarray
    entered_by_min_episode_age: np.ndarray
    bad_tracking_failures_by_min_episode_age: np.ndarray
    start_cohort_episodes: np.ndarray
    start_cohort_bad_tracking: np.ndarray
    start_cohort_early_bad_tracking: np.ndarray
    start_cohort_passed_next_bin: np.ndarray
    start_cohort_reached_final_bin: np.ndarray
    start_cohort_motion_ends: np.ndarray
    start_cohort_timeouts: np.ndarray
    start_cohort_non_bad_terminal: np.ndarray
    start_cohort_non_bad_reached_final: np.ndarray
    start_cohort_episode_age_sum: np.ndarray
    start_cohort_max_bin_sum: np.ndarray
    start_cohort_bad_failure_bins: np.ndarray
    exact_phase_zero_episodes: int
    exact_phase_zero_motion_ends: int
    exact_phase_zero_timeouts: int
    exact_phase_zero_non_bad_terminal: int
    exact_phase_zero_non_bad_reached_final: int
    motion_end_key: str | None
    timeout_key: str | None
    total_episodes: int
    analyzed_episodes: int
    incomplete_episodes: int
    bad_tracking_episodes: int
    invalid_phase_rows: int
    nonmonotonic_phase_episodes: int
    multiple_bad_terminal_episodes: int
    bad_reason_without_done_rows: int
    unresolved_bad_failure_episodes: int

    @property
    def hazard(self) -> np.ndarray:
        return np.divide(
            self.bad_tracking_failures,
            self.entered_episodes,
            out=np.full(self.entered_episodes.shape, np.nan, dtype=np.float64),
            where=self.entered_episodes > 0,
        )

    @property
    def hazards_by_min_episode_age(self) -> np.ndarray:
        return np.divide(
            self.bad_tracking_failures_by_min_episode_age,
            self.entered_by_min_episode_age,
            out=np.full(self.entered_by_min_episode_age.shape, np.nan, dtype=np.float64),
            where=self.entered_by_min_episode_age > 0,
        )


def _read_scalar_rows(dataset: h5py.Dataset, start: int, end: int, dtype: np.dtype) -> np.ndarray:
    values = np.asarray(dataset[start:end])
    if values.shape[0] != end - start:
        raise ValueError(f"Unexpected row count for HDF5 dataset '{dataset.name}'.")
    if values.size != end - start:
        raise ValueError(f"Expected scalar rows in HDF5 dataset '{dataset.name}', got shape {values.shape}.")
    return values.reshape(-1).astype(dtype, copy=False)


def _iter_episode_blocks(
    h5_file: h5py.File,
    keys: dict[str, str],
    num_samples: int,
    chunk_size: int,
) -> Iterator[dict[str, np.ndarray]]:
    carry: dict[str, np.ndarray] | None = None

    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        batch = {
            "episode_id": _read_scalar_rows(h5_file[keys["episode_id"]], start, end, np.int64),
            "phase": _read_scalar_rows(h5_file[keys["phase"]], start, end, np.float64),
            "bad": _read_scalar_rows(h5_file[keys["bad"]], start, end, np.bool_),
            "done": _read_scalar_rows(h5_file[keys["done"]], start, end, np.bool_),
        }
        if "complete" in keys:
            batch["complete"] = _read_scalar_rows(h5_file[keys["complete"]], start, end, np.bool_)
        else:
            batch["complete"] = np.ones(end - start, dtype=bool)
        for optional_reason in ("motion_end", "timeout"):
            if optional_reason in keys:
                batch[optional_reason] = _read_scalar_rows(
                    h5_file[keys[optional_reason]],
                    start,
                    end,
                    np.bool_,
                )

        if carry is not None:
            batch = {name: np.concatenate((carry[name], values)) for name, values in batch.items()}

        changes = np.flatnonzero(np.diff(batch["episode_id"]) != 0) + 1
        starts = np.concatenate((np.array([0]), changes))
        ends = np.concatenate((changes, np.array([batch["episode_id"].shape[0]])))

        for block_start, block_end in zip(starts[:-1], ends[:-1]):
            yield {name: values[block_start:block_end] for name, values in batch.items()}

        last_start = int(starts[-1])
        carry = {name: values[last_start:] for name, values in batch.items()}

    if carry is not None and carry["episode_id"].size > 0:
        yield carry


def _phase_bin_indices(
    phases: np.ndarray,
    phase_min: float,
    phase_max: float,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    tolerance = np.finfo(np.float32).eps * 8.0
    valid = (
        np.isfinite(phases)
        & (phases >= phase_min - tolerance)
        & (phases <= phase_max + tolerance)
    )
    clipped = np.clip(phases[valid], phase_min, phase_max)
    scaled = (clipped - phase_min) / (phase_max - phase_min)
    indices = np.floor(scaled * num_bins).astype(np.int64)
    return np.minimum(indices, num_bins - 1), valid


def _resolve_optional_key(
    h5_file: h5py.File,
    explicit_key: str | None,
    candidates: tuple[str, ...],
    label: str,
) -> str | None:
    if explicit_key is not None:
        if explicit_key not in h5_file:
            raise KeyError(f"Missing requested {label} HDF5 key '{explicit_key}'.")
        return explicit_key
    return next((candidate for candidate in candidates if candidate in h5_file), None)


def analyze_phase_failure_hazard(
    path: Path,
    *,
    num_bins: int = 20,
    phase_min: float = 0.0,
    phase_max: float = 1.0,
    episode_key: str = "episode_id",
    phase_key: str = "motion_phase",
    bad_key: str = "next_done_bad_tracking",
    done_key: str = "dones",
    complete_key: str = "episode_data_complete",
    motion_end_key: str | None = None,
    timeout_key: str | None = None,
    phase_semantics: str = "auto",
    include_incomplete: bool = False,
    exclude_episode_age_le: tuple[int, ...] = (10, 20),
    chunk_size: int = 1_000_000,
) -> PhaseFailureHazardResult:
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    if phase_max <= phase_min:
        raise ValueError("phase_max must be greater than phase_min.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if any(age < 0 for age in exclude_episode_age_le):
        raise ValueError("exclude_episode_age_le values must be non-negative.")

    age_thresholds = np.asarray(
        (0, *sorted(set(int(age) for age in exclude_episode_age_le if age > 0))),
        dtype=np.int64,
    )
    entered_by_age = np.zeros((age_thresholds.size, num_bins), dtype=np.int64)
    failures_by_age = np.zeros_like(entered_by_age)
    start_cohort_episodes = np.zeros(num_bins, dtype=np.int64)
    start_cohort_bad_tracking = np.zeros(num_bins, dtype=np.int64)
    start_cohort_early_bad_tracking = np.zeros((age_thresholds.size, num_bins), dtype=np.int64)
    start_cohort_passed_next_bin = np.zeros(num_bins, dtype=np.int64)
    start_cohort_reached_final_bin = np.zeros(num_bins, dtype=np.int64)
    start_cohort_motion_ends = np.zeros(num_bins, dtype=np.int64)
    start_cohort_timeouts = np.zeros(num_bins, dtype=np.int64)
    start_cohort_non_bad_terminal = np.zeros(num_bins, dtype=np.int64)
    start_cohort_non_bad_reached_final = np.zeros(num_bins, dtype=np.int64)
    start_cohort_episode_age_sum = np.zeros(num_bins, dtype=np.int64)
    start_cohort_max_bin_sum = np.zeros(num_bins, dtype=np.int64)
    start_cohort_bad_failure_bins = np.zeros((num_bins, num_bins), dtype=np.int64)
    exact_phase_zero_episodes = 0
    exact_phase_zero_motion_ends = 0
    exact_phase_zero_timeouts = 0
    exact_phase_zero_non_bad_terminal = 0
    exact_phase_zero_non_bad_reached_final = 0
    total_episodes = 0
    analyzed_episodes = 0
    incomplete_episodes = 0
    bad_tracking_episodes = 0
    invalid_phase_rows = 0
    nonmonotonic_phase_episodes = 0
    multiple_bad_terminal_episodes = 0
    bad_reason_without_done_rows = 0
    unresolved_bad_failure_episodes = 0
    seen_episode_ids: set[int] = set()
    resolved_motion_end_key: str | None = None
    resolved_timeout_key: str | None = None

    with h5py.File(path, "r") as h5_file:
        required_keys = {
            "episode_id": episode_key,
            "phase": phase_key,
            "bad": bad_key,
            "done": done_key,
        }
        missing = [key for key in required_keys.values() if key not in h5_file]
        if missing:
            raise KeyError(
                f"Missing required HDF5 keys: {missing}. "
                "Use a FastSACEpisodeDataAgent dataset or override the key arguments."
            )

        keys = dict(required_keys)
        if complete_key in h5_file:
            keys["complete"] = complete_key
        elif not include_incomplete:
            raise KeyError(
                f"Missing required HDF5 key '{complete_key}'. "
                "Pass --include-incomplete only if every stored episode is known to be complete."
            )
        resolved_motion_end_key = _resolve_optional_key(
            h5_file,
            motion_end_key,
            ("next_done_motion_ends", "next.done_motion_ends", "done_motion_ends"),
            "motion-end",
        )
        resolved_timeout_key = _resolve_optional_key(
            h5_file,
            timeout_key,
            (
                "next_done_timeout",
                "next.done_timeout",
                "done_timeout",
                "next_truncations",
                "next.truncations",
                "truncations",
            ),
            "timeout",
        )
        if resolved_motion_end_key is not None:
            keys["motion_end"] = resolved_motion_end_key
        if resolved_timeout_key is not None:
            keys["timeout"] = resolved_timeout_key

        inferred_samples = int(h5_file[episode_key].shape[0])
        num_samples = int(h5_file.attrs.get("num_samples", inferred_samples))
        stored_phase_semantics = h5_file.attrs.get("motion_phase_semantics", "legacy_post_step")
        if isinstance(stored_phase_semantics, bytes):
            stored_phase_semantics = stored_phase_semantics.decode("utf-8")
        resolved_phase_semantics = str(stored_phase_semantics) if phase_semantics == "auto" else phase_semantics
        if resolved_phase_semantics not in {"legacy_post_step", "pre_step"}:
            raise ValueError("phase_semantics must be one of: auto, legacy_post_step, pre_step.")
        for key in keys.values():
            if h5_file[key].shape[0] < num_samples:
                raise ValueError(
                    f"HDF5 key '{key}' has {h5_file[key].shape[0]:,} rows, fewer than num_samples={num_samples:,}."
                )

        for episode in _iter_episode_blocks(h5_file, keys, num_samples, chunk_size):
            total_episodes += 1
            episode_id = int(episode["episode_id"][0])
            if episode_id in seen_episode_ids:
                raise ValueError(
                    f"episode_id={episode_id} appears in multiple non-contiguous blocks; "
                    "episode-level hazard would be ambiguous."
                )
            seen_episode_ids.add(episode_id)

            if not include_incomplete and not bool(episode["complete"].all()):
                incomplete_episodes += 1
                continue

            analyzed_episodes += 1
            episode_age = int(episode["episode_id"].size)
            _, valid_phase = _phase_bin_indices(
                episode["phase"],
                phase_min,
                phase_max,
                num_bins,
            )
            invalid_phase_rows += int((~valid_phase).sum())
            entry_mask = valid_phase.copy()
            if resolved_phase_semantics == "legacy_post_step":
                entry_mask &= ~episode["done"]
            phase_bins, _ = _phase_bin_indices(
                episode["phase"][entry_mask],
                phase_min,
                phase_max,
                num_bins,
            )
            unique_phase_bins = np.unique(phase_bins)
            qualifying_age_profiles = episode_age > age_thresholds
            if unique_phase_bins.size > 0:
                for profile_idx in np.flatnonzero(qualifying_age_profiles):
                    entered_by_age[profile_idx, unique_phase_bins] += 1

            ordered_phases = episode["phase"][entry_mask]
            if ordered_phases.size > 1 and bool((np.diff(ordered_phases) < -1e-6).any()):
                nonmonotonic_phase_episodes += 1

            bad_reason_without_done_rows += int((episode["bad"] & ~episode["done"]).sum())
            bad_terminal_rows = np.flatnonzero(episode["bad"] & episode["done"])
            is_bad_tracking = bad_terminal_rows.size > 0
            if is_bad_tracking:
                bad_tracking_episodes += 1
                if bad_terminal_rows.size > 1:
                    multiple_bad_terminal_episodes += 1

            failure_bin: int | None = None
            if is_bad_tracking:
                failure_row = int(bad_terminal_rows[-1])
                if resolved_phase_semantics == "legacy_post_step":
                    prior_valid_rows = np.flatnonzero(valid_phase[:failure_row] & ~episode["done"][:failure_row])
                    if prior_valid_rows.size == 0:
                        unresolved_bad_failure_episodes += 1
                    else:
                        failure_row = int(prior_valid_rows[-1])
                if resolved_phase_semantics != "legacy_post_step" or prior_valid_rows.size > 0:
                    failure_bins, failure_valid = _phase_bin_indices(
                        episode["phase"][failure_row : failure_row + 1],
                        phase_min,
                        phase_max,
                        num_bins,
                    )
                    if bool(failure_valid[0]):
                        failure_bin = int(failure_bins[0])
                        for profile_idx in np.flatnonzero(qualifying_age_profiles):
                            failures_by_age[profile_idx, failure_bin] += 1

            if phase_bins.size == 0:
                continue

            start_bin = int(phase_bins[0])
            max_bin = int(unique_phase_bins[-1])
            reached_final_bin = max_bin == num_bins - 1
            passed_next_bin = max_bin > start_bin
            is_motion_end = bool(episode.get("motion_end", np.zeros(1, dtype=bool)).any())
            is_timeout = bool(episode.get("timeout", np.zeros(1, dtype=bool)).any())
            has_terminal = bool(episode["done"].any() or is_motion_end or is_timeout)
            is_non_bad_terminal = has_terminal and not is_bad_tracking
            is_non_bad_reached_final = is_non_bad_terminal and reached_final_bin

            start_cohort_episodes[start_bin] += 1
            start_cohort_episode_age_sum[start_bin] += episode_age
            start_cohort_max_bin_sum[start_bin] += max_bin
            start_cohort_passed_next_bin[start_bin] += int(passed_next_bin)
            start_cohort_reached_final_bin[start_bin] += int(reached_final_bin)
            start_cohort_bad_tracking[start_bin] += int(is_bad_tracking)
            start_cohort_motion_ends[start_bin] += int(is_motion_end)
            start_cohort_timeouts[start_bin] += int(is_timeout)
            start_cohort_non_bad_terminal[start_bin] += int(is_non_bad_terminal)
            start_cohort_non_bad_reached_final[start_bin] += int(is_non_bad_reached_final)
            if is_bad_tracking:
                for profile_idx, age_threshold in enumerate(age_thresholds):
                    if episode_age <= int(age_threshold):
                        start_cohort_early_bad_tracking[profile_idx, start_bin] += 1
                if failure_bin is not None:
                    start_cohort_bad_failure_bins[start_bin, failure_bin] += 1

            first_phase = float(ordered_phases[0])
            phase_zero_tolerance = np.finfo(np.float32).eps * 8.0
            starts_at_exact_phase_zero = abs(first_phase - phase_min) <= phase_zero_tolerance
            if starts_at_exact_phase_zero:
                exact_phase_zero_episodes += 1
                exact_phase_zero_motion_ends += int(is_motion_end)
                exact_phase_zero_timeouts += int(is_timeout)
                exact_phase_zero_non_bad_terminal += int(is_non_bad_terminal)
                exact_phase_zero_non_bad_reached_final += int(is_non_bad_reached_final)

    return PhaseFailureHazardResult(
        phase_semantics=resolved_phase_semantics,
        phase_edges=np.linspace(phase_min, phase_max, num_bins + 1),
        entered_episodes=entered_by_age[0],
        bad_tracking_failures=failures_by_age[0],
        min_episode_ages=age_thresholds,
        entered_by_min_episode_age=entered_by_age,
        bad_tracking_failures_by_min_episode_age=failures_by_age,
        start_cohort_episodes=start_cohort_episodes,
        start_cohort_bad_tracking=start_cohort_bad_tracking,
        start_cohort_early_bad_tracking=start_cohort_early_bad_tracking,
        start_cohort_passed_next_bin=start_cohort_passed_next_bin,
        start_cohort_reached_final_bin=start_cohort_reached_final_bin,
        start_cohort_motion_ends=start_cohort_motion_ends,
        start_cohort_timeouts=start_cohort_timeouts,
        start_cohort_non_bad_terminal=start_cohort_non_bad_terminal,
        start_cohort_non_bad_reached_final=start_cohort_non_bad_reached_final,
        start_cohort_episode_age_sum=start_cohort_episode_age_sum,
        start_cohort_max_bin_sum=start_cohort_max_bin_sum,
        start_cohort_bad_failure_bins=start_cohort_bad_failure_bins,
        exact_phase_zero_episodes=exact_phase_zero_episodes,
        exact_phase_zero_motion_ends=exact_phase_zero_motion_ends,
        exact_phase_zero_timeouts=exact_phase_zero_timeouts,
        exact_phase_zero_non_bad_terminal=exact_phase_zero_non_bad_terminal,
        exact_phase_zero_non_bad_reached_final=exact_phase_zero_non_bad_reached_final,
        motion_end_key=resolved_motion_end_key,
        timeout_key=resolved_timeout_key,
        total_episodes=total_episodes,
        analyzed_episodes=analyzed_episodes,
        incomplete_episodes=incomplete_episodes,
        bad_tracking_episodes=bad_tracking_episodes,
        invalid_phase_rows=invalid_phase_rows,
        nonmonotonic_phase_episodes=nonmonotonic_phase_episodes,
        multiple_bad_terminal_episodes=multiple_bad_terminal_episodes,
        bad_reason_without_done_rows=bad_reason_without_done_rows,
        unresolved_bad_failure_episodes=unresolved_bad_failure_episodes,
    )


def _derived_csv_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix or '.csv'}")


def _write_csv(path: Path, result: PhaseFailureHazardResult) -> tuple[Path, Path, Path]:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "excluded_episode_age_le",
                "phase_bin",
                "phase_left",
                "phase_right",
                "entered_episodes",
                "bad_tracking_failures",
                "hazard",
            ]
        )
        for profile_idx, age_threshold in enumerate(result.min_episode_ages):
            for bin_idx, (left, right, entered, failures, hazard) in enumerate(
                zip(
                    result.phase_edges[:-1],
                    result.phase_edges[1:],
                    result.entered_by_min_episode_age[profile_idx],
                    result.bad_tracking_failures_by_min_episode_age[profile_idx],
                    result.hazards_by_min_episode_age[profile_idx],
                )
            ):
                writer.writerow(
                    [
                        int(age_threshold),
                        bin_idx,
                        left,
                        right,
                        int(entered),
                        int(failures),
                        hazard,
                    ]
                )

    cohort_path = _derived_csv_path(path, "start_cohorts")
    early_age_headers = [
        f"bad_tracking_age_le_{int(age_threshold)}"
        for age_threshold in result.min_episode_ages
        if age_threshold > 0
    ]
    with cohort_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "start_phase_bin",
                "phase_left",
                "phase_right",
                "episodes",
                "mean_episode_age",
                "bad_tracking",
                "bad_tracking_rate",
                *early_age_headers,
                "passed_next_bin",
                "passed_next_bin_rate",
                "reached_final_bin",
                "reached_final_bin_rate",
                "motion_ends",
                "timeouts",
                "non_bad_terminal",
                "non_bad_reached_final",
                "mean_max_phase_bin",
            ]
        )
        for start_bin, cohort_count in enumerate(result.start_cohort_episodes):
            count = int(cohort_count)
            denominator = max(count, 1)
            early_counts = [
                int(result.start_cohort_early_bad_tracking[profile_idx, start_bin])
                for profile_idx, threshold in enumerate(result.min_episode_ages)
                if threshold > 0
            ]
            writer.writerow(
                [
                    start_bin,
                    result.phase_edges[start_bin],
                    result.phase_edges[start_bin + 1],
                    count,
                    result.start_cohort_episode_age_sum[start_bin] / denominator if count else np.nan,
                    int(result.start_cohort_bad_tracking[start_bin]),
                    result.start_cohort_bad_tracking[start_bin] / denominator if count else np.nan,
                    *early_counts,
                    int(result.start_cohort_passed_next_bin[start_bin]),
                    result.start_cohort_passed_next_bin[start_bin] / denominator if count else np.nan,
                    int(result.start_cohort_reached_final_bin[start_bin]),
                    result.start_cohort_reached_final_bin[start_bin] / denominator if count else np.nan,
                    int(result.start_cohort_motion_ends[start_bin]),
                    int(result.start_cohort_timeouts[start_bin]),
                    int(result.start_cohort_non_bad_terminal[start_bin]),
                    int(result.start_cohort_non_bad_reached_final[start_bin]),
                    result.start_cohort_max_bin_sum[start_bin] / denominator if count else np.nan,
                ]
            )

    failure_path = _derived_csv_path(path, "cohort_failures")
    with failure_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "start_phase_bin",
                "failure_phase_bin",
                "failure_phase_left",
                "failure_phase_right",
                "bad_tracking_failures",
                "fraction_of_start_cohort",
                "fraction_of_cohort_bad_tracking",
            ]
        )
        for start_bin in range(result.start_cohort_bad_failure_bins.shape[0]):
            cohort_count = int(result.start_cohort_episodes[start_bin])
            bad_count = int(result.start_cohort_bad_tracking[start_bin])
            for failure_bin in np.flatnonzero(result.start_cohort_bad_failure_bins[start_bin]):
                failure_count = int(result.start_cohort_bad_failure_bins[start_bin, failure_bin])
                writer.writerow(
                    [
                        start_bin,
                        int(failure_bin),
                        result.phase_edges[failure_bin],
                        result.phase_edges[failure_bin + 1],
                        failure_count,
                        failure_count / cohort_count if cohort_count else np.nan,
                        failure_count / bad_count if bad_count else np.nan,
                    ]
                )
    return path, cohort_path, failure_path


def _format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.3f}%"


def _print_hazard_profile(
    result: PhaseFailureHazardResult,
    profile_idx: int,
    age_threshold: int,
) -> None:
    if age_threshold == 0:
        print("Phase failure hazard — all complete episodes")
    else:
        print(f"Phase failure hazard — excluding episode_age <= {age_threshold}")
    print("  bin  phase_range        entered     bad_fail      hazard")
    for bin_idx, (left, right, entered, failures, hazard) in enumerate(
        zip(
            result.phase_edges[:-1],
            result.phase_edges[1:],
            result.entered_by_min_episode_age[profile_idx],
            result.bad_tracking_failures_by_min_episode_age[profile_idx],
            result.hazards_by_min_episode_age[profile_idx],
        )
    ):
        right_bracket = "]" if bin_idx == result.entered_episodes.size - 1 else ")"
        hazard_text = "n/a" if np.isnan(hazard) else f"{hazard:.6f} ({100.0 * hazard:.3f}%)"
        print(
            f"  {bin_idx:>3d}  [{left:>6.3f}, {right:>6.3f}{right_bracket} "
            f"{int(entered):>10,} {int(failures):>12,}  {hazard_text}"
        )


def _print_result(path: Path, result: PhaseFailureHazardResult, gate_bin: int = 6) -> None:
    print(f"path: {path}")
    print(f"motion_phase_semantics: {result.phase_semantics}")
    print(f"motion_end_key: {result.motion_end_key or 'not found'}")
    print(f"timeout_key: {result.timeout_key or 'not found'}")
    print(f"episodes: analyzed={result.analyzed_episodes:,} total={result.total_episodes:,}")
    print(f"bad_tracking_episodes: {result.bad_tracking_episodes:,}")
    for profile_idx, age_threshold in enumerate(result.min_episode_ages):
        print()
        _print_hazard_profile(result, profile_idx, int(age_threshold))

    early_profile_indices = [
        profile_idx
        for profile_idx, threshold in enumerate(result.min_episode_ages)
        if threshold > 0
    ]
    early_headers = " ".join(
        f"bad<=age{int(result.min_episode_ages[profile_idx]):<3d}"
        for profile_idx in early_profile_indices
    )
    print()
    print("Start-phase cohorts")
    print(
        "  bin  phase_range       episodes mean_age bad(rate)       "
        f"{early_headers} pass_next(rate) reach_final motion_end timeout nonbad_final mean_max_bin"
    )
    for start_bin, cohort_count_value in enumerate(result.start_cohort_episodes):
        cohort_count = int(cohort_count_value)
        if cohort_count == 0:
            continue
        left = result.phase_edges[start_bin]
        right = result.phase_edges[start_bin + 1]
        right_bracket = "]" if start_bin == result.start_cohort_episodes.size - 1 else ")"
        mean_age = result.start_cohort_episode_age_sum[start_bin] / cohort_count
        mean_max_bin = result.start_cohort_max_bin_sum[start_bin] / cohort_count
        bad_count = int(result.start_cohort_bad_tracking[start_bin])
        passed_count = int(result.start_cohort_passed_next_bin[start_bin])
        early_text = " ".join(
            f"{int(result.start_cohort_early_bad_tracking[profile_idx, start_bin]):>10,}"
            for profile_idx in early_profile_indices
        )
        print(
            f"  {start_bin:>3d}  [{left:>6.3f}, {right:>6.3f}{right_bracket} "
            f"{cohort_count:>8,} {mean_age:>8.2f} "
            f"{bad_count:>6,}({_format_ratio(bad_count, cohort_count):>8}) "
            f"{early_text} "
            f"{passed_count:>8,}({_format_ratio(passed_count, cohort_count):>8}) "
            f"{int(result.start_cohort_reached_final_bin[start_bin]):>11,} "
            f"{int(result.start_cohort_motion_ends[start_bin]):>10,} "
            f"{int(result.start_cohort_timeouts[start_bin]):>7,} "
            f"{int(result.start_cohort_non_bad_reached_final[start_bin]):>12,} "
            f"{mean_max_bin:>12.2f}"
        )

    print()
    print("Start cohort -> bad-tracking failure bins")
    for start_bin in range(result.start_cohort_bad_failure_bins.shape[0]):
        failure_bins = np.flatnonzero(result.start_cohort_bad_failure_bins[start_bin])
        if failure_bins.size == 0:
            continue
        destinations = ", ".join(
            f"bin{int(failure_bin)}={int(result.start_cohort_bad_failure_bins[start_bin, failure_bin]):,}"
            for failure_bin in failure_bins
        )
        print(f"  start_bin{start_bin}: {destinations}")

    gate_episodes = int(result.start_cohort_episodes[gate_bin])
    gate_passed = int(result.start_cohort_passed_next_bin[gate_bin])
    gate_bad = int(result.start_cohort_bad_tracking[gate_bin])
    gate_early = {
        int(result.min_episode_ages[profile_idx]): int(
            result.start_cohort_early_bad_tracking[profile_idx, gate_bin]
        )
        for profile_idx in early_profile_indices
    }
    print()
    print("Gate diagnostics")
    print(
        f"  start_bin{gate_bin}_passed_next: {gate_passed:,}/{gate_episodes:,} "
        f"({_format_ratio(gate_passed, gate_episodes)})"
    )
    print(
        f"  start_bin{gate_bin}_bad_tracking: {gate_bad:,}/{gate_episodes:,} "
        f"({_format_ratio(gate_bad, gate_episodes)}), early_bad={gate_early}"
    )
    print(
        "  exact_phase_zero: "
        f"episodes={result.exact_phase_zero_episodes:,}, "
        f"motion_ends={result.exact_phase_zero_motion_ends:,}, "
        f"timeouts={result.exact_phase_zero_timeouts:,}, "
        f"non_bad_terminal={result.exact_phase_zero_non_bad_terminal:,}, "
        f"non_bad_reached_final={result.exact_phase_zero_non_bad_reached_final:,}"
    )
    print(
        "  phase_zero_full_completer_exists: "
        f"{result.exact_phase_zero_motion_ends > 0 or result.exact_phase_zero_non_bad_reached_final > 0}"
    )

    print()
    print("Sanity")
    print(f"  incomplete_episodes_skipped:       {result.incomplete_episodes:,}")
    print(f"  invalid_phase_rows:                {result.invalid_phase_rows:,}")
    print(f"  nonmonotonic_phase_episodes:       {result.nonmonotonic_phase_episodes:,}")
    print(f"  multiple_bad_terminal_episodes:    {result.multiple_bad_terminal_episodes:,}")
    print(f"  bad_reason_without_done_rows:      {result.bad_reason_without_done_rows:,}")
    print(f"  unresolved_bad_failure_episodes:  {result.unresolved_bad_failure_episodes:,}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute phase-conditioned bad-tracking episode hazards.")
    parser.add_argument("path", type=Path, help="FastSAC episode-data HDF5 file.")
    parser.add_argument("--num-bins", type=int, default=20, help="Number of equal-width phase bins.")
    parser.add_argument("--phase-min", type=float, default=0.0)
    parser.add_argument("--phase-max", type=float, default=1.0)
    parser.add_argument("--episode-key", default="episode_id")
    parser.add_argument("--phase-key", default="motion_phase")
    parser.add_argument("--bad-key", default="next_done_bad_tracking")
    parser.add_argument("--done-key", default="dones")
    parser.add_argument("--complete-key", default="episode_data_complete")
    parser.add_argument(
        "--motion-end-key",
        default=None,
        help="Optional override; otherwise common motion-end keys are auto-detected.",
    )
    parser.add_argument(
        "--timeout-key",
        default=None,
        help="Optional override; otherwise common timeout/truncation keys are auto-detected.",
    )
    parser.add_argument(
        "--phase-semantics",
        choices=("auto", "legacy_post_step", "pre_step"),
        default="auto",
        help="Auto uses the HDF5 attribute; old files without it use terminal-preceding phase.",
    )
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument(
        "--exclude-episode-age-le",
        nargs="*",
        type=int,
        default=[10, 20],
        metavar="N",
        help="Also report hazards after removing whole episodes with age <= N.",
    )
    parser.add_argument(
        "--gate-bin",
        type=int,
        default=6,
        help="Start-phase cohort bin highlighted in the gate diagnostics.",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--csv",
        type=Path,
        help="Writes hazard CSV plus derived *_start_cohorts.csv and *_cohort_failures.csv files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.path.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {args.path}")
    if not 0 <= args.gate_bin < args.num_bins:
        raise ValueError(f"--gate-bin must be in [0, {args.num_bins - 1}].")

    result = analyze_phase_failure_hazard(
        args.path,
        num_bins=args.num_bins,
        phase_min=args.phase_min,
        phase_max=args.phase_max,
        episode_key=args.episode_key,
        phase_key=args.phase_key,
        bad_key=args.bad_key,
        done_key=args.done_key,
        complete_key=args.complete_key,
        motion_end_key=args.motion_end_key,
        timeout_key=args.timeout_key,
        phase_semantics=args.phase_semantics,
        include_incomplete=args.include_incomplete,
        exclude_episode_age_le=tuple(args.exclude_episode_age_le),
        chunk_size=args.chunk_size,
    )
    _print_result(args.path, result, gate_bin=args.gate_bin)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_paths = _write_csv(args.csv, result)
        for csv_path in csv_paths:
            print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
