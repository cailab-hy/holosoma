#!/usr/bin/env python3
"""Build a fragmented-success WBT offline dataset from per-clip FastSAC replays.

Why
===
Any single time-slice of one training replay contains "a few full completions
plus many failures" — success arrives episode-wise, so weighted cloning (AWR /
IQL) can simply amplify the few complete demos and the stitching challenge
evaporates. This tool CONSTRUCTS fragmentation instead: for every motion clip
(= one collection run = one H5 file) it selects replay from a different
training PERIOD, chosen so that the clip has (almost) no complete episodes,
while partial competence for different clips/phases coexists across the merged
dataset. Completion must then be assembled by value stitching, not cloning.

Inputs
======
Per-clip H5 files written by the FastSAC episode-data exporter
(``fast_sac_episode_data``): episodes are stored as contiguous row blocks with
a globally unique ``episode_id``, plus per-row ``next_global_step`` (collection
time — the period axis), ``next_done_motion_ends`` (completion flag at the
episode's last row) and ``motion_phase`` (progress inside the clip).

An episode counts as a COMPLETE DEMONSTRATION only when it traversed the
motion end-to-end: min motion_phase <= --demo-start-phase (default 0.05, i.e.
it started at the beginning) AND (last row has done_motion_ends=1 OR max
motion_phase >= --complete-phase, default 0.98 — near-timeout episodes that
tracked essentially the whole motion are demos in all but name). The span
condition matters for random-start collections (start_at_timestep_zero_prob
< 1): an episode that starts at phase 0.6 and reaches the end is a valuable
late-phase FRAGMENT, not a demonstration, and must survive the cut.

Usage
=====
    # 1) inspect the clip x period completion matrix (no writes)
    python scripts/build_fragmented_wbt_dataset.py audit \
        --clip largebox=offline_data/clip_largebox.h5 \
        --clip walk=offline_data/clip_walk.h5 --num-periods 10

    # 2) build the merged fragmented dataset
    python scripts/build_fragmented_wbt_dataset.py build \
        --clip largebox=offline_data/clip_largebox.h5 \
        --clip walk=offline_data/clip_walk.h5 \
        --output offline_data/wbt_fragmented.h5 \
        --max-complete-rate 0.05 --periods-per-clip 2

Selection: per clip, periods (equal-width bins of that clip's global_step
range) whose completion rate is below --max-complete-rate are eligible;
--periods-per-clip of them are picked with a rotating offset per clip so
different clips contribute different training eras (override any clip with
--periods "clip=3,4"). Complete episodes surviving inside selected periods are
dropped down to --max-complete-per-clip (default 0). Whole episodes only —
transitions, n-step fields and mc_return stay valid.

Output: same flat schema as the input (readable by HDF5BlockReader /
GPUTransitionCache unchanged) plus a per-row ``motion_id`` column, remapped
globally-unique episode_ids, and the full selection spec + per-clip stats
frozen into HDF5 attrs (``fragment_spec``) and a sibling .json file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

FORMAT_ATTR = "holosoma_offpolicy_transition_v1"
WRITE_CHUNK_ROWS = 65536


@dataclass
class EpisodeTable:
    clip: str
    path: Path
    episode_id: np.ndarray  # [E]
    start: np.ndarray  # [E] first row (inclusive)
    end: np.ndarray  # [E] last row (exclusive)
    global_step: np.ndarray  # [E] first-row next_global_step
    complete: np.ndarray  # [E] bool, full-traversal demo: started near phase 0 AND reached the end
    motion_ends: np.ndarray  # [E] bool
    max_phase: np.ndarray  # [E]
    min_phase: np.ndarray  # [E]
    length: np.ndarray  # [E]
    period: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))  # [E]
    period_edges: np.ndarray = field(default_factory=lambda: np.empty(0))


def _read_full(h5: h5py.File, key: str, fallback: float | None = None) -> np.ndarray:
    if key in h5:
        return np.asarray(h5[key][:]).reshape(len(h5[key]), -1).squeeze(-1)
    if fallback is None:
        raise KeyError(f"required key {key!r} missing (is this an episode-data export?)")
    return np.full(int(h5.attrs.get("num_samples", h5["observations"].shape[0])), fallback)


def load_episode_table(clip: str, path: Path, complete_phase: float, demo_start_phase: float = 0.05) -> EpisodeTable:
    with h5py.File(path, "r") as h5:
        episode_id = _read_full(h5, "episode_id").astype(np.int64)
        global_step = _read_full(h5, "next_global_step", fallback=0).astype(np.int64)
        motion_ends = _read_full(h5, "next_done_motion_ends", fallback=0).astype(bool)
        phase = _read_full(h5, "motion_phase", fallback=0.0).astype(np.float64)
        data_complete = _read_full(h5, "episode_data_complete", fallback=1).astype(bool)

    # Episodes are flushed whole, so each id must be one contiguous block.
    change = np.flatnonzero(np.diff(episode_id) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [episode_id.shape[0]]])
    block_ids = episode_id[starts]
    if np.unique(block_ids).shape[0] != block_ids.shape[0]:
        raise ValueError(
            f"{path}: episode_id blocks are not contiguous/unique — this file was not "
            "written by the episode-data exporter; whole-episode selection is unsafe."
        )

    n_eps = starts.shape[0]
    max_phase = np.array([phase[s:e].max() for s, e in zip(starts, ends)])
    min_phase = np.array([phase[s:e].min() for s, e in zip(starts, ends)])
    ends_flag = motion_ends[ends - 1]
    keep = np.array([bool(data_complete[s:e].all()) for s, e in zip(starts, ends)])
    if not keep.all():
        print(f"[{clip}] dropping {int((~keep).sum())}/{n_eps} episodes with episode_data_complete=0")
    # Full-traversal demo only: started near phase 0 AND reached the end. A late-start
    # episode that hits done_motion_ends is a late-phase fragment, not a demonstration.
    complete = (min_phase <= demo_start_phase) & (ends_flag | (max_phase >= complete_phase))
    table = EpisodeTable(
        clip=clip,
        path=path,
        episode_id=block_ids[keep],
        start=starts[keep],
        end=ends[keep],
        global_step=global_step[starts[keep]],
        complete=complete[keep],
        motion_ends=ends_flag[keep],
        max_phase=max_phase[keep],
        min_phase=min_phase[keep],
        length=(ends - starts)[keep],
    )
    return table


def assign_periods(table: EpisodeTable, num_periods: int) -> None:
    lo, hi = int(table.global_step.min()), int(table.global_step.max())
    edges = np.linspace(lo, hi + 1, num_periods + 1)
    table.period = np.clip(np.searchsorted(edges, table.global_step, side="right") - 1, 0, num_periods - 1)
    table.period_edges = edges


def completion_matrix(table: EpisodeTable, num_periods: int) -> list[dict[str, float]]:
    rows = []
    for period in range(num_periods):
        mask = table.period == period
        n = int(mask.sum())
        n_complete = int((mask & table.complete).sum())
        rows.append(
            {
                "period": period,
                "step_lo": float(table.period_edges[period]),
                "step_hi": float(table.period_edges[period + 1]),
                "episodes": n,
                "complete": n_complete,
                "complete_rate": (n_complete / n) if n else float("nan"),
                "phase_p50": float(np.median(table.max_phase[mask])) if n else float("nan"),
                "phase_p90": float(np.quantile(table.max_phase[mask], 0.9)) if n else float("nan"),
            }
        )
    return rows


def print_matrix(table: EpisodeTable, matrix: list[dict[str, float]]) -> None:
    print(f"\n=== clip '{table.clip}' ({table.path.name}): {table.episode_id.shape[0]} episodes, "
          f"{int(table.complete.sum())} complete ===")
    print(f"{'period':>6} {'global_step range':>24} {'eps':>6} {'complete':>9} {'rate':>7} {'phase_p50':>10} {'phase_p90':>10}")
    for row in matrix:
        rate = f"{100.0 * row['complete_rate']:.1f}%" if row["episodes"] else "-"
        print(
            f"{row['period']:>6} {int(row['step_lo']):>11,}-{int(row['step_hi']):<11,} {row['episodes']:>6} "
            f"{row['complete']:>9} {rate:>7} {row['phase_p50']:>10.3f} {row['phase_p90']:>10.3f}"
        )


def select_episodes(
    tables: list[EpisodeTable],
    num_periods: int,
    max_complete_rate: float,
    periods_per_clip: int,
    overrides: dict[str, list[int]],
    max_complete_per_clip: int,
    min_episodes_per_period: int,
    seed: int,
) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    selection: dict[str, dict] = {}
    for clip_index, table in enumerate(tables):
        matrix = completion_matrix(table, num_periods)
        if table.clip in overrides:
            chosen = overrides[table.clip]
            bad = [p for p in chosen if not (0 <= p < num_periods)]
            if bad:
                raise ValueError(f"--periods for clip {table.clip!r}: invalid period(s) {bad}")
        else:
            eligible = [
                row["period"]
                for row in matrix
                if row["episodes"] >= min_episodes_per_period and row["complete_rate"] <= max_complete_rate
            ]
            if not eligible:
                raise ValueError(
                    f"clip {table.clip!r}: no period with >= {min_episodes_per_period} episodes and "
                    f"completion rate <= {max_complete_rate:.1%}. Loosen --max-complete-rate or pass "
                    f"--periods '{table.clip}=...' explicitly."
                )
            offset = clip_index % len(eligible)
            chosen = [eligible[(offset + j) % len(eligible)] for j in range(min(periods_per_clip, len(eligible)))]
            chosen = sorted(set(chosen))

        in_period = np.isin(table.period, chosen)
        complete_idx = np.flatnonzero(in_period & table.complete)
        keep_complete = rng.permutation(complete_idx)[:max_complete_per_clip]
        keep_mask = in_period.copy()
        keep_mask[complete_idx] = False
        keep_mask[keep_complete] = True

        selection[table.clip] = {
            "clip_index": clip_index,
            "periods": [int(p) for p in chosen],
            "episode_mask": keep_mask,
            "episodes": int(keep_mask.sum()),
            "complete_kept": int((keep_mask & table.complete).sum()),
            "complete_dropped": int(len(complete_idx) - len(keep_complete)),
            "rows": int(table.length[keep_mask].sum()),
        }
    return selection


def _phase_coverage(phase_rows: np.ndarray, bins: int = 20) -> tuple[float, np.ndarray]:
    hist, _ = np.histogram(np.clip(phase_rows, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    return float((hist > 0).mean()), hist


def build_output(
    tables: list[EpisodeTable],
    selection: dict[str, dict],
    output: Path,
    spec: dict,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    next_episode_id = 0
    total_rows = 0
    report: dict = {"clips": {}, "spec": spec}
    with h5py.File(output, "w") as out:
        out.attrs["format"] = FORMAT_ATTR
        for table in tables:
            chosen = selection[table.clip]
            mask = chosen["episode_mask"]
            starts, ends = table.start[mask], table.end[mask]
            clip_rows = 0
            phase_rows_all: list[np.ndarray] = []
            with h5py.File(table.path, "r") as src:
                keys = [k for k in src.keys() if isinstance(src[k], h5py.Dataset)]
                # batch contiguous episode blocks into large write chunks
                pending: list[tuple[int, int]] = []
                pending_rows = 0

                def _flush_pending() -> None:
                    nonlocal pending, pending_rows, total_rows, clip_rows, next_episode_id
                    if not pending:
                        return
                    parts = {key: [] for key in keys}
                    episode_ids = []
                    for block_start, block_end in pending:
                        for key in keys:
                            parts[key].append(src[key][block_start:block_end])
                        length = block_end - block_start
                        episode_ids.append(np.full(length, next_episode_id, dtype=np.int64))
                        next_episode_id += 1
                    batch = {key: np.concatenate(parts[key], axis=0) for key in keys}
                    batch["episode_id"] = np.concatenate(episode_ids)
                    batch["motion_id"] = np.full(batch["episode_id"].shape[0], chosen["clip_index"], dtype=np.int32)
                    if "motion_phase" in batch:
                        phase_rows_all.append(np.asarray(batch["motion_phase"]).reshape(-1))
                    rows = batch["episode_id"].shape[0]
                    for key, value in batch.items():
                        if key not in out:
                            out.create_dataset(
                                key,
                                shape=(0,) + value.shape[1:],
                                maxshape=(None,) + value.shape[1:],
                                dtype=value.dtype,
                                chunks=True,
                                compression="gzip",
                                compression_opts=4,
                            )
                        ds = out[key]
                        ds.resize((total_rows + rows,) + ds.shape[1:])
                        ds[total_rows : total_rows + rows] = value
                    total_rows += rows
                    clip_rows += rows
                    pending, pending_rows = [], 0

                for block_start, block_end in zip(starts, ends):
                    pending.append((int(block_start), int(block_end)))
                    pending_rows += int(block_end - block_start)
                    if pending_rows >= WRITE_CHUNK_ROWS:
                        _flush_pending()
                _flush_pending()

            coverage, hist = (
                _phase_coverage(np.concatenate(phase_rows_all)) if phase_rows_all else (float("nan"), np.zeros(20))
            )
            report["clips"][table.clip] = {
                "clip_index": chosen["clip_index"],
                "source": str(table.path),
                "periods": chosen["periods"],
                "period_step_ranges": [
                    [float(table.period_edges[p]), float(table.period_edges[p + 1])] for p in chosen["periods"]
                ],
                "episodes": chosen["episodes"],
                "rows": clip_rows,
                "complete_kept": chosen["complete_kept"],
                "complete_dropped": chosen["complete_dropped"],
                "phase_coverage_frac": coverage,
                "phase_hist_20": hist.astype(int).tolist(),
            }
        out.attrs["num_samples"] = total_rows
        out.attrs["fragment_spec"] = json.dumps(report, sort_keys=True)
    report["total_rows"] = total_rows
    report["output"] = str(output)
    return report


def _parse_clips(items: list[str]) -> list[tuple[str, Path]]:
    clips = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            name, path = Path(item).stem, item
        clips.append((name, Path(path).expanduser()))
    names = [name for name, _ in clips]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate clip names: {names}")
    return clips


def _parse_period_overrides(items: list[str]) -> dict[str, list[int]]:
    overrides: dict[str, list[int]] = {}
    for item in items:
        name, spec = item.split("=", 1)
        overrides[name] = [int(p) for p in spec.replace(",", " ").split()]
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "build"):
        p = sub.add_parser(name)
        p.add_argument("--clip", action="append", required=True, metavar="NAME=PATH",
                       help="Per-clip replay H5 (repeat per clip). NAME optional (file stem).")
        p.add_argument("--num-periods", type=int, default=10, help="Equal-width global_step bins per clip.")
        p.add_argument("--complete-phase", type=float, default=0.98,
                       help="Episodes reaching this max motion_phase count as reaching the end even without done_motion_ends.")
        p.add_argument("--demo-start-phase", type=float, default=0.05,
                       help="Episodes only count as complete DEMOS if their min motion_phase is <= this "
                            "(random-start episodes finishing from mid-motion are fragments, not demos).")
        if name == "build":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--max-complete-rate", type=float, default=0.05,
                           help="Period eligibility: completion rate must be <= this.")
            p.add_argument("--periods-per-clip", type=int, default=1)
            p.add_argument("--periods", action="append", default=[], metavar="CLIP=P1,P2",
                           help="Explicit period override per clip (skips eligibility filtering).")
            p.add_argument("--max-complete-per-clip", type=int, default=0,
                           help="Complete episodes allowed to survive per clip (default 0).")
            p.add_argument("--min-episodes-per-period", type=int, default=20)
            p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    clips = _parse_clips(args.clip)
    tables = [load_episode_table(name, path, args.complete_phase, args.demo_start_phase) for name, path in clips]
    for table in tables:
        assign_periods(table, args.num_periods)
        print_matrix(table, completion_matrix(table, args.num_periods))

    if args.command == "audit":
        return

    selection = select_episodes(
        tables,
        num_periods=args.num_periods,
        max_complete_rate=args.max_complete_rate,
        periods_per_clip=args.periods_per_clip,
        overrides=_parse_period_overrides(args.periods),
        max_complete_per_clip=args.max_complete_per_clip,
        min_episodes_per_period=args.min_episodes_per_period,
        seed=args.seed,
    )
    spec = {
        "num_periods": args.num_periods,
        "complete_phase": args.complete_phase,
        "demo_start_phase": args.demo_start_phase,
        "max_complete_rate": args.max_complete_rate,
        "periods_per_clip": args.periods_per_clip,
        "max_complete_per_clip": args.max_complete_per_clip,
        "min_episodes_per_period": args.min_episodes_per_period,
        "seed": args.seed,
        "clips": {name: str(path) for name, path in clips},
    }
    report = build_output(tables, selection, args.output, spec)

    print(f"\n=== fragmented dataset written: {args.output} ({report['total_rows']:,} rows) ===")
    for clip, stats in report["clips"].items():
        print(
            f"  {clip}: periods={stats['periods']} episodes={stats['episodes']:,} rows={stats['rows']:,} "
            f"complete kept/dropped={stats['complete_kept']}/{stats['complete_dropped']} "
            f"phase_coverage={stats['phase_coverage_frac']:.2f}"
        )
    json_path = args.output.with_suffix(".fragment_spec.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"selection spec + stats: {json_path}")


if __name__ == "__main__":
    main()
