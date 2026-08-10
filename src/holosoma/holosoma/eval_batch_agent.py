from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import tyro
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.eval_agent import _log_repeated_eval_summary
from holosoma.utils.eval_utils import CheckpointConfig, init_eval_logging, load_saved_experiment_config
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


_STEP_TOKEN = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<suffix>[kKmM]?)$")
_COMPACT_STEP_OPTION = re.compile(r"^--(?P<steps>\d+(?:\.\d+)?[kKmM]?(?:,\d+(?:\.\d+)?[kKmM]?)+)$")
_CHECKPOINT_STEP = re.compile(r"model_(?P<step>\d+)\.pt$")


def _parse_step(token: str) -> int:
    match = _STEP_TOKEN.fullmatch(token.strip())
    if match is None:
        raise ValueError(f"invalid checkpoint step '{token}'; expected e.g. 80000, 80k, or 0.1m")
    value = float(match.group("value"))
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000}[match.group("suffix").lower()]
    step = int(round(value * multiplier))
    if step <= 0:
        raise ValueError(f"checkpoint step must be positive: {token}")
    return step


def _extract_compact_step_options(argv: list[str]) -> tuple[list[str], list[str]]:
    filtered: list[str] = []
    compact_steps: list[str] = []
    for argument in argv:
        match = _COMPACT_STEP_OPTION.fullmatch(argument)
        if match is None:
            filtered.append(argument)
        else:
            compact_steps.extend(match.group("steps").split(","))
    return filtered, compact_steps


def _parse_step_arguments(step_arguments: list[str] | None, compact_steps: list[str]) -> list[int]:
    tokens: list[str] = []
    for argument in step_arguments or []:
        tokens.extend(part for part in argument.split(",") if part)
    tokens.extend(compact_steps)
    return list(dict.fromkeys(_parse_step(token) for token in tokens))


def _checkpoint_step(path: Path) -> int | None:
    match = _CHECKPOINT_STEP.fullmatch(path.name)
    return int(match.group("step")) if match is not None else None


def _resolve_checkpoints(
    checkpoint_argument: str,
    steps: list[int],
    pattern: str,
    allow_missing: bool,
) -> list[tuple[int | None, Path]]:
    checkpoint_path = Path(checkpoint_argument).expanduser().resolve()
    if checkpoint_path.is_file():
        if steps:
            raise ValueError("--steps cannot be used when --checkpoint points to a single file")
        return [(_checkpoint_step(checkpoint_path), checkpoint_path)]
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"checkpoint path does not exist: {checkpoint_path}")

    if not steps:
        discovered = sorted(
            (
                (step, path)
                for path in checkpoint_path.glob("model_*.pt")
                if (step := _checkpoint_step(path)) is not None
            ),
            key=lambda item: item[0],
        )
        if not discovered:
            raise FileNotFoundError(f"no model_*.pt checkpoints found in {checkpoint_path}")
        return discovered

    resolved: list[tuple[int, Path]] = []
    missing: list[Path] = []
    for step in steps:
        path = checkpoint_path / pattern.format(step=step)
        if path.is_file():
            resolved.append((step, path))
        else:
            missing.append(path)
    if missing and not allow_missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"requested checkpoints are missing:\n{formatted}")
    for path in missing:
        logger.warning("[Batch Eval] skipping missing checkpoint: {}", path)
    if not resolved:
        raise FileNotFoundError("none of the requested checkpoints exist")
    return resolved


def _safe_folder_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "batch_eval"


def _create_result_directory(result_root: str, checkpoint_argument: str, result_name: str | None) -> Path:
    root = Path(result_root).expanduser().resolve()
    checkpoint_name = Path(checkpoint_argument).expanduser().resolve().name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = result_name or f"{checkpoint_name}_{timestamp}"
    result_directory = root / _safe_folder_name(folder_name)
    result_directory.mkdir(parents=True, exist_ok=False)
    return result_directory


def _json_dump(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2, sort_keys=True)


def _percent(value: float) -> float:
    return 100.0 * float(value)


def _checkpoint_report_text(record: dict[str, Any], eval_config: ExperimentConfig) -> str:
    report = record["report"]
    total = report["total_summary"]
    lines = [
        "Holosoma Batch Evaluation — Checkpoint Report",
        "=" * 54,
        f"checkpoint: {record['checkpoint']}",
        f"step: {record['step']}",
        f"num_envs: {eval_config.training.num_envs}",
        f"repeats: {report['num_repeats']}",
        f"total_episodes: {int(total['num_episodes'])}",
        "",
        "Repeat Results",
        "-" * 54,
    ]
    for repeat in report["repeat_summaries"]:
        lines.append(
            "repeat={repeat} seed={seed} episodes={episodes} "
            "success={success:.4f}% failure={failure:.4f}% "
            "bad_tracking={bad:.4f}% timeout={timeout:.4f}% "
            "return={return_mean:.6f}±{return_std:.6f} length={length_mean:.4f}±{length_std:.4f}".format(
                repeat=repeat["repeat"],
                seed=repeat["seed"] if repeat["seed"] is not None else "continued_rng",
                episodes=int(repeat["num_episodes"]),
                success=_percent(repeat["success_ratio"]),
                failure=_percent(1.0 - repeat["success_ratio"]),
                bad=_percent(repeat["bad_tracking_ratio"]),
                timeout=_percent(repeat["timeout_ratio"]),
                return_mean=repeat["episode_return_mean"],
                return_std=repeat["episode_return_std"],
                length_mean=repeat["episode_length_mean"],
                length_std=repeat["episode_length_std"],
            )
        )

    lines.extend(
        [
            "",
            "Checkpoint Aggregate",
            "-" * 54,
            f"success: {_percent(total['success_ratio']):.4f}% "
            f"± {_percent(report['success_ratio_std']):.4f}% across repeats",
            f"failure: {_percent(1.0 - total['success_ratio']):.4f}%",
            f"bad_tracking: {_percent(total['bad_tracking_ratio']):.4f}% "
            f"± {_percent(report['bad_tracking_ratio_std']):.4f}% across repeats",
            f"timeout: {_percent(total['timeout_ratio']):.4f}% "
            f"± {_percent(report['timeout_ratio_std']):.4f}% across repeats",
            f"episode_return: {total['episode_return_mean']:.6f} ± {total['episode_return_std']:.6f}",
            f"episode_length: {total['episode_length_mean']:.4f} ± {total['episode_length_std']:.4f}",
            f"stop_reason_counts: {report['stop_reason_counts']}",
            f"bad_tracking_detail_counts: {report['bad_tracking_detail_counts']}",
            f"bad_tracking_phase_counts: {report['bad_tracking_phase_counts']}",
            f"bad_tracking_phase_percentages: {report['bad_tracking_phase_percentages']}",
            f"bad_tracking_phase_unresolved: {report['bad_tracking_phase_unresolved']}",
            "",
        ]
    )
    return "\n".join(lines)


def _combine_population_stats(
    summaries: list[dict[str, float]],
    mean_key: str,
    std_key: str,
) -> tuple[float, float]:
    total_count = sum(float(summary["num_episodes"]) for summary in summaries)
    if total_count <= 0.0:
        return 0.0, 0.0
    weighted_sum = sum(
        float(summary["num_episodes"]) * float(summary[mean_key]) for summary in summaries
    )
    weighted_square_sum = sum(
        float(summary["num_episodes"])
        * (float(summary[std_key]) ** 2 + float(summary[mean_key]) ** 2)
        for summary in summaries
    )
    mean = weighted_sum / total_count
    variance = max(0.0, weighted_square_sum / total_count - mean**2)
    return mean, math.sqrt(variance)


def _sum_count_dict(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update({str(name): int(count) for name, count in record["report"][key].items()})
    return dict(sorted(counts.items()))


def _aggregate_records(records: list[dict[str, Any]], eval_config: ExperimentConfig) -> dict[str, Any]:
    totals = [record["report"]["total_summary"] for record in records]
    repeat_summaries = [
        repeat
        for record in records
        for repeat in record["report"]["repeat_summaries"]
    ]
    stop_reason_counts = _sum_count_dict(records, "stop_reason_counts")
    detail_counts = _sum_count_dict(records, "bad_tracking_detail_counts")
    phase_counts = _sum_count_dict(records, "bad_tracking_phase_counts")
    total_episodes = int(sum(float(summary["num_episodes"]) for summary in totals))
    successful_episodes = stop_reason_counts.get("motion_ends", 0) + stop_reason_counts.get("segment_ends", 0)
    bad_tracking_episodes = stop_reason_counts.get("bad_tracking", 0)
    timeout_episodes = stop_reason_counts.get("timeout", 0)
    return_mean, return_std = _combine_population_stats(
        totals, "episode_return_mean", "episode_return_std"
    )
    length_mean, length_std = _combine_population_stats(
        totals, "episode_length_mean", "episode_length_std"
    )

    checkpoint_success = [float(summary["success_ratio"]) for summary in totals]
    repeat_success = [float(summary["success_ratio"]) for summary in repeat_summaries]
    checkpoint_bad = [float(summary["bad_tracking_ratio"]) for summary in totals]
    repeat_bad = [float(summary["bad_tracking_ratio"]) for summary in repeat_summaries]

    def mean_std(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        return (
            float(statistics.fmean(values)),
            float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        )

    checkpoint_success_mean, checkpoint_success_std = mean_std(checkpoint_success)
    repeat_success_mean, repeat_success_std = mean_std(repeat_success)
    checkpoint_bad_mean, checkpoint_bad_std = mean_std(checkpoint_bad)
    repeat_bad_mean, repeat_bad_std = mean_std(repeat_bad)

    return {
        "num_checkpoints": len(records),
        "num_repeats_per_checkpoint": int(eval_config.training.eval_num_repeats),
        "num_envs": int(eval_config.training.num_envs),
        "expected_total_episodes": (
            len(records)
            * int(eval_config.training.eval_num_repeats)
            * int(eval_config.training.num_envs)
        ),
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "failed_episodes": total_episodes - successful_episodes,
        "bad_tracking_episodes": bad_tracking_episodes,
        "timeout_episodes": timeout_episodes,
        "pooled_success_ratio": successful_episodes / max(total_episodes, 1),
        "pooled_failure_ratio": (total_episodes - successful_episodes) / max(total_episodes, 1),
        "pooled_bad_tracking_ratio": bad_tracking_episodes / max(total_episodes, 1),
        "pooled_timeout_ratio": timeout_episodes / max(total_episodes, 1),
        "checkpoint_success_ratio_mean": checkpoint_success_mean,
        "checkpoint_success_ratio_std": checkpoint_success_std,
        "all_repeat_success_ratio_mean": repeat_success_mean,
        "all_repeat_success_ratio_std": repeat_success_std,
        "checkpoint_bad_tracking_ratio_mean": checkpoint_bad_mean,
        "checkpoint_bad_tracking_ratio_std": checkpoint_bad_std,
        "all_repeat_bad_tracking_ratio_mean": repeat_bad_mean,
        "all_repeat_bad_tracking_ratio_std": repeat_bad_std,
        "episode_return_mean": return_mean,
        "episode_return_std": return_std,
        "episode_length_mean": length_mean,
        "episode_length_std": length_std,
        "stop_reason_counts": stop_reason_counts,
        "bad_tracking_detail_counts": detail_counts,
        "bad_tracking_phase_counts": phase_counts,
        "checkpoints": [
            {"step": record["step"], "checkpoint": record["checkpoint"]}
            for record in records
        ],
    }


def _overall_report_text(overall: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Holosoma Batch Evaluation — Overall Report",
            "=" * 54,
            f"checkpoints: {overall['num_checkpoints']}",
            f"num_envs: {overall['num_envs']}",
            f"repeats_per_checkpoint: {overall['num_repeats_per_checkpoint']}",
            f"expected_total_episodes: {overall['expected_total_episodes']}",
            f"actual_total_episodes: {overall['total_episodes']}",
            "",
            f"pooled_success: {_percent(overall['pooled_success_ratio']):.4f}%",
            f"pooled_failure: {_percent(overall['pooled_failure_ratio']):.4f}%",
            f"pooled_bad_tracking: {_percent(overall['pooled_bad_tracking_ratio']):.4f}%",
            f"pooled_timeout: {_percent(overall['pooled_timeout_ratio']):.4f}%",
            f"checkpoint_success_mean: {_percent(overall['checkpoint_success_ratio_mean']):.4f}% "
            f"± {_percent(overall['checkpoint_success_ratio_std']):.4f}%",
            f"all_repeat_success_mean: {_percent(overall['all_repeat_success_ratio_mean']):.4f}% "
            f"± {_percent(overall['all_repeat_success_ratio_std']):.4f}%",
            f"checkpoint_bad_tracking_mean: {_percent(overall['checkpoint_bad_tracking_ratio_mean']):.4f}% "
            f"± {_percent(overall['checkpoint_bad_tracking_ratio_std']):.4f}%",
            f"all_repeat_bad_tracking_mean: {_percent(overall['all_repeat_bad_tracking_ratio_mean']):.4f}% "
            f"± {_percent(overall['all_repeat_bad_tracking_ratio_std']):.4f}%",
            f"episode_return: {overall['episode_return_mean']:.6f} ± {overall['episode_return_std']:.6f}",
            f"episode_length: {overall['episode_length_mean']:.4f} ± {overall['episode_length_std']:.4f}",
            f"stop_reason_counts: {overall['stop_reason_counts']}",
            f"bad_tracking_detail_counts: {overall['bad_tracking_detail_counts']}",
            f"bad_tracking_phase_counts: {overall['bad_tracking_phase_counts']}",
            "",
        ]
    )


def _write_checkpoint_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        report = record["report"]
        total = report["total_summary"]
        rows.append(
            {
                "step": record["step"],
                "checkpoint": record["checkpoint"],
                "repeats": report["num_repeats"],
                "episodes": int(total["num_episodes"]),
                "success_percent": _percent(total["success_ratio"]),
                "success_percent_std_across_repeats": _percent(report["success_ratio_std"]),
                "failure_percent": _percent(1.0 - total["success_ratio"]),
                "bad_tracking_percent": _percent(total["bad_tracking_ratio"]),
                "bad_tracking_percent_std_across_repeats": _percent(report["bad_tracking_ratio_std"]),
                "timeout_percent": _percent(total["timeout_ratio"]),
                "timeout_percent_std_across_repeats": _percent(report["timeout_ratio_std"]),
                "episode_return_mean": total["episode_return_mean"],
                "episode_return_std": total["episode_return_std"],
                "episode_length_mean": total["episode_length_mean"],
                "episode_length_std": total["episode_length_std"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_parser(default_result_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", help="checkpoint directory or a single .pt file")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=None,
        help="checkpoint steps, e.g. --steps 80k 85k 90k or --steps 80k,85k,90k",
    )
    parser.add_argument("--checkpoint-pattern", default="model_{step:07d}.pt")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--result-root", default=str(default_result_root))
    parser.add_argument("--result-name", default=None)
    parser.add_argument("--help-batch", action="store_true")
    return parser


def main() -> None:
    init_eval_logging()
    repo_root = Path(__file__).resolve().parents[3]
    filtered_argv, compact_steps = _extract_compact_step_options(sys.argv[1:])
    parser = _build_parser(repo_root / "test_result")
    batch_args, remaining_args = parser.parse_known_args(filtered_argv)
    if batch_args.help_batch:
        parser.print_help()
        return
    if not batch_args.checkpoint:
        parser.error("--checkpoint is required")

    steps = _parse_step_arguments(batch_args.steps, compact_steps)
    checkpoints = _resolve_checkpoints(
        batch_args.checkpoint,
        steps,
        batch_args.checkpoint_pattern,
        batch_args.allow_missing,
    )
    result_directory = _create_result_directory(
        batch_args.result_root,
        batch_args.checkpoint,
        batch_args.result_name,
    )
    logger.add(str(result_directory / "eval_batch.log"), level="INFO")
    logger.info("[Batch Eval] results will be saved to {}", result_directory)
    logger.info("[Batch Eval] resolved checkpoints={}", [str(path) for _, path in checkpoints])

    first_checkpoint_cfg = CheckpointConfig(checkpoint=str(checkpoints[0][1]))
    saved_config, saved_wandb_path = load_saved_experiment_config(first_checkpoint_cfg)
    eval_config = saved_config.get_eval_config()
    eval_config = dataclasses.replace(
        eval_config,
        training=dataclasses.replace(eval_config.training, headless=True),
    )
    eval_config = tyro.cli(
        ExperimentConfig,
        default=eval_config,
        args=remaining_args,
        description="Batch-eval overrides on top of the checkpoint's saved config.",
        config=TYRO_CONIFG,
    )
    eval_config.save_config(str(result_directory / "holosoma_config.yaml"))
    _json_dump(
        result_directory / "batch_manifest.json",
        {
            "checkpoint_argument": batch_args.checkpoint,
            "requested_steps": steps,
            "checkpoint_pattern": batch_args.checkpoint_pattern,
            "resolved_checkpoints": [
                {"step": step, "path": str(path)} for step, path in checkpoints
            ],
            "num_envs": int(eval_config.training.num_envs),
            "num_repeats": int(eval_config.training.eval_num_repeats),
            "seed": int(eval_config.training.seed),
            "seed_stride": int(eval_config.training.eval_seed_stride),
        },
    )

    env = None
    simulation_app = None
    records: list[dict[str, Any]] = []
    try:
        env, device, simulation_app = setup_simulation_environment(eval_config)
        algo_class = get_class(eval_config.algo._target_)
        algo: BaseAlgo = algo_class(
            device=device,
            env=env,
            config=eval_config.algo.config,
            log_dir=str(result_directory / "runtime"),
            multi_gpu_cfg=None,
        )
        algo.setup()
        algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)

        for checkpoint_index, (step, checkpoint_path) in enumerate(checkpoints, start=1):
            logger.info(
                "[Batch Eval] checkpoint {}/{} step={} path={}",
                checkpoint_index,
                len(checkpoints),
                step,
                checkpoint_path,
            )
            algo.load(str(checkpoint_path))
            report = _log_repeated_eval_summary(algo, eval_config)
            if report is None:
                raise AttributeError(
                    f"{type(algo).__name__} exposes neither evaluate_vectorized_episodes nor evaluate_one_episode"
                )
            record = {
                "step": step,
                "checkpoint": str(checkpoint_path),
                "report": report,
            }
            records.append(record)
            file_stem = f"checkpoint_{step:07d}" if step is not None else f"checkpoint_{checkpoint_index:03d}"
            (result_directory / f"{file_stem}.txt").write_text(
                _checkpoint_report_text(record, eval_config),
                encoding="utf-8",
            )
            _json_dump(result_directory / f"{file_stem}.json", record)
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)

    if not records:
        raise RuntimeError("batch evaluation completed without any checkpoint results")

    overall = _aggregate_records(records, eval_config)
    _write_checkpoint_csv(result_directory / "checkpoint_summary.csv", records)
    (result_directory / "overall.txt").write_text(
        _overall_report_text(overall),
        encoding="utf-8",
    )
    _json_dump(result_directory / "overall.json", overall)
    logger.info("[Batch Eval] completed {} checkpoints", len(records))
    logger.info("[Batch Eval] result directory: {}", result_directory)


if __name__ == "__main__":
    main()
