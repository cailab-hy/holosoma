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
    phase_semantics: str = "auto",
    include_incomplete: bool = False,
    chunk_size: int = 1_000_000,
) -> PhaseFailureHazardResult:
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    if phase_max <= phase_min:
        raise ValueError("phase_max must be greater than phase_min.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    entered = np.zeros(num_bins, dtype=np.int64)
    failures = np.zeros(num_bins, dtype=np.int64)
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
            if phase_bins.size > 0:
                entered[np.unique(phase_bins)] += 1

            finite_phases = episode["phase"][np.isfinite(episode["phase"])]
            if finite_phases.size > 1 and bool((np.diff(finite_phases) < -1e-6).any()):
                nonmonotonic_phase_episodes += 1

            bad_reason_without_done_rows += int((episode["bad"] & ~episode["done"]).sum())
            bad_terminal_rows = np.flatnonzero(episode["bad"] & episode["done"])
            if bad_terminal_rows.size == 0:
                continue

            bad_tracking_episodes += 1
            if bad_terminal_rows.size > 1:
                multiple_bad_terminal_episodes += 1
            failure_row = int(bad_terminal_rows[-1])
            if resolved_phase_semantics == "legacy_post_step":
                prior_valid_rows = np.flatnonzero(valid_phase[:failure_row] & ~episode["done"][:failure_row])
                if prior_valid_rows.size == 0:
                    unresolved_bad_failure_episodes += 1
                    continue
                failure_row = int(prior_valid_rows[-1])
            failure_bins, failure_valid = _phase_bin_indices(
                episode["phase"][failure_row : failure_row + 1],
                phase_min,
                phase_max,
                num_bins,
            )
            if bool(failure_valid[0]):
                failures[int(failure_bins[0])] += 1

    return PhaseFailureHazardResult(
        phase_semantics=resolved_phase_semantics,
        phase_edges=np.linspace(phase_min, phase_max, num_bins + 1),
        entered_episodes=entered,
        bad_tracking_failures=failures,
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


def _write_csv(path: Path, result: PhaseFailureHazardResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["phase_bin", "phase_left", "phase_right", "entered_episodes", "bad_tracking_failures", "hazard"]
        )
        for bin_idx, (left, right, entered, failures, hazard) in enumerate(
            zip(
                result.phase_edges[:-1],
                result.phase_edges[1:],
                result.entered_episodes,
                result.bad_tracking_failures,
                result.hazard,
            )
        ):
            writer.writerow([bin_idx, left, right, int(entered), int(failures), hazard])


def _print_result(path: Path, result: PhaseFailureHazardResult) -> None:
    print(f"path: {path}")
    print(f"motion_phase_semantics: {result.phase_semantics}")
    print(f"episodes: analyzed={result.analyzed_episodes:,} total={result.total_episodes:,}")
    print(f"bad_tracking_episodes: {result.bad_tracking_episodes:,}")
    print()
    print("Phase failure hazard")
    print("  bin  phase_range        entered     bad_fail      hazard")
    for bin_idx, (left, right, entered, failures, hazard) in enumerate(
        zip(
            result.phase_edges[:-1],
            result.phase_edges[1:],
            result.entered_episodes,
            result.bad_tracking_failures,
            result.hazard,
        )
    ):
        right_bracket = "]" if bin_idx == result.entered_episodes.size - 1 else ")"
        hazard_text = "n/a" if np.isnan(hazard) else f"{hazard:.6f} ({100.0 * hazard:.3f}%)"
        print(
            f"  {bin_idx:>3d}  [{left:>6.3f}, {right:>6.3f}{right_bracket} "
            f"{int(entered):>10,} {int(failures):>12,}  {hazard_text}"
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
        "--phase-semantics",
        choices=("auto", "legacy_post_step", "pre_step"),
        default="auto",
        help="Auto uses the HDF5 attribute; old files without it use terminal-preceding phase.",
    )
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.path.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {args.path}")

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
        phase_semantics=args.phase_semantics,
        include_incomplete=args.include_incomplete,
        chunk_size=args.chunk_size,
    )
    _print_result(args.path, result)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(args.csv, result)
        print(f"csv: {args.csv}")


if __name__ == "__main__":
    main()
