from __future__ import annotations

import os
import statistics
from typing import Any

import tyro
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import (
    close_simulation_app,
    setup_simulation_environment,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _stop_reason_counts(eval_results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in eval_results:
        stop_reason_value = result.get("stop_reason")
        stop_reason = str(stop_reason_value) if stop_reason_value is not None else "none"
        counts[stop_reason] = counts.get(stop_reason, 0) + 1
    return counts


def _summarize_eval_results(eval_results: list[dict[str, Any]]) -> dict[str, float]:
    if not eval_results:
        return {
            "num_episodes": 0.0,
            "episode_return_mean": 0.0,
            "episode_return_std": 0.0,
            "episode_length_mean": 0.0,
            "episode_length_std": 0.0,
            "success_ratio": 0.0,
            "bad_tracking_ratio": 0.0,
            "timeout_ratio": 0.0,
        }

    episode_returns = [float(result.get("episode_return", 0.0)) for result in eval_results]
    episode_lengths = [float(result.get("episode_length", 0.0)) for result in eval_results]
    counts = _stop_reason_counts(eval_results)
    num_episodes = float(len(eval_results))
    return {
        "num_episodes": num_episodes,
        "episode_return_mean": float(statistics.fmean(episode_returns)),
        "episode_return_std": float(statistics.pstdev(episode_returns)) if len(episode_returns) > 1 else 0.0,
        "episode_length_mean": float(statistics.fmean(episode_lengths)),
        "episode_length_std": float(statistics.pstdev(episode_lengths)) if len(episode_lengths) > 1 else 0.0,
        "success_ratio": float(counts.get("motion_ends", 0)) / num_episodes,
        "bad_tracking_ratio": float(counts.get("bad_tracking", 0)) / num_episodes,
        "timeout_ratio": float(counts.get("timeout", 0)) / num_episodes,
    }


def _set_defer_eval_resets(algo: BaseAlgo, enabled: bool) -> bool:
    env_candidates = [
        getattr(algo, "unwrapped_env", None),
        getattr(algo, "env", None),
        getattr(getattr(algo, "env", None), "_env", None),
    ]
    for env in env_candidates:
        if env is not None and hasattr(env, "set_defer_resets"):
            env.set_defer_resets(enabled)
            return True
    return False


def _log_repeated_eval_summary(algo: BaseAlgo, tyro_config: ExperimentConfig) -> bool:
    evaluate_vectorized_episodes_fn = getattr(algo, "evaluate_vectorized_episodes", None)
    evaluate_one_episode_fn = getattr(algo, "evaluate_one_episode", None)
    if not callable(evaluate_vectorized_episodes_fn) and not callable(evaluate_one_episode_fn):
        return False

    num_repeats = max(1, int(tyro_config.training.eval_num_repeats))
    max_eval_steps = tyro_config.training.max_eval_steps
    logger.info(
        "[Eval] starting repeated evaluation with num_envs={} repeats={} max_eval_steps={}",
        tyro_config.training.num_envs,
        num_repeats,
        max_eval_steps,
    )
    repeat_summaries: list[dict[str, float]] = []
    all_eval_results: list[dict[str, Any]] = []

    defer_resets = _set_defer_eval_resets(algo, True)
    if defer_resets:
        logger.info("[Eval] deferring automatic environment resets during vectorized evaluation.")
    try:
        for repeat_idx in range(num_repeats):
            if callable(evaluate_vectorized_episodes_fn):
                eval_batch_results = evaluate_vectorized_episodes_fn(
                    max_eval_steps=max_eval_steps,
                    use_early_termination=False,
                )
                if isinstance(eval_batch_results, list):
                    repeat_results = [result for result in eval_batch_results if isinstance(result, dict)]
                elif isinstance(eval_batch_results, dict):
                    repeat_results = [eval_batch_results]
                else:
                    repeat_results = []
            else:
                repeat_results = []
                eval_num_episodes = max(1, int(tyro_config.training.eval_num_episodes))
                for _ in range(eval_num_episodes):
                    eval_result = evaluate_one_episode_fn(
                        max_eval_steps=max_eval_steps,
                        use_early_termination=False,
                    )
                    if isinstance(eval_result, dict):
                        repeat_results.append(eval_result)

            all_eval_results.extend(repeat_results)
            repeat_summary = _summarize_eval_results(repeat_results)
            repeat_summaries.append(repeat_summary)
            logger.info(
                "[Eval Repeat] repeat={}/{} episodes={} success={:.2f}% bad_tracking={:.2f}% "
                "timeout={:.2f}% return_mean={:.4f} length_mean={:.2f}",
                repeat_idx + 1,
                num_repeats,
                int(repeat_summary["num_episodes"]),
                100.0 * repeat_summary["success_ratio"],
                100.0 * repeat_summary["bad_tracking_ratio"],
                100.0 * repeat_summary["timeout_ratio"],
                repeat_summary["episode_return_mean"],
                repeat_summary["episode_length_mean"],
            )
    finally:
        if defer_resets:
            _set_defer_eval_resets(algo, False)

    if not all_eval_results:
        logger.warning("No evaluation episodes were completed; cannot summarize eval ratios.")
        return True

    total_summary = _summarize_eval_results(all_eval_results)
    success_ratios = [summary["success_ratio"] for summary in repeat_summaries]
    bad_tracking_ratios = [summary["bad_tracking_ratio"] for summary in repeat_summaries]
    timeout_ratios = [summary["timeout_ratio"] for summary in repeat_summaries]
    success_std = float(statistics.pstdev(success_ratios)) if len(success_ratios) > 1 else 0.0
    bad_tracking_std = float(statistics.pstdev(bad_tracking_ratios)) if len(bad_tracking_ratios) > 1 else 0.0
    timeout_std = float(statistics.pstdev(timeout_ratios)) if len(timeout_ratios) > 1 else 0.0

    stop_reason_counts = _stop_reason_counts(all_eval_results)
    eval_metrics = {
        "Eval/num_repeats": float(num_repeats),
        "Eval/num_episodes": total_summary["num_episodes"],
        "Eval/success_ratio": total_summary["success_ratio"],
        "Eval/success_percent": 100.0 * total_summary["success_ratio"],
        "Eval/success_ratio_std": success_std,
        "Eval/success_percent_std": 100.0 * success_std,
        "Eval/bad_tracking_ratio": total_summary["bad_tracking_ratio"],
        "Eval/bad_tracking_percent": 100.0 * total_summary["bad_tracking_ratio"],
        "Eval/bad_tracking_ratio_std": bad_tracking_std,
        "Eval/bad_tracking_percent_std": 100.0 * bad_tracking_std,
        "Eval/timeout_ratio": total_summary["timeout_ratio"],
        "Eval/timeout_percent": 100.0 * total_summary["timeout_ratio"],
        "Eval/timeout_ratio_std": timeout_std,
        "Eval/timeout_percent_std": 100.0 * timeout_std,
        "Eval/episode_return_mean": total_summary["episode_return_mean"],
        "Eval/episode_return_std": total_summary["episode_return_std"],
        "Eval/episode_length_mean": total_summary["episode_length_mean"],
        "Eval/episode_length_std": total_summary["episode_length_std"],
    }
    for stop_reason, count in stop_reason_counts.items():
        ratio = float(count) / max(1.0, total_summary["num_episodes"])
        eval_metrics[f"Eval/stop_reason/{stop_reason}"] = ratio
        eval_metrics[f"Eval/stop_reason_percent/{stop_reason}"] = 100.0 * ratio

    logger.info(
        "[Eval Summary] repeats={} total_episodes={} success={:.2f}%±{:.2f}% "
        "bad_tracking={:.2f}%±{:.2f}% timeout={:.2f}%±{:.2f}% "
        "return_mean={:.4f} return_std={:.4f} length_mean={:.2f} length_std={:.2f}",
        num_repeats,
        int(total_summary["num_episodes"]),
        100.0 * total_summary["success_ratio"],
        100.0 * success_std,
        100.0 * total_summary["bad_tracking_ratio"],
        100.0 * bad_tracking_std,
        100.0 * total_summary["timeout_ratio"],
        100.0 * timeout_std,
        total_summary["episode_return_mean"],
        total_summary["episode_return_std"],
        total_summary["episode_length_mean"],
        total_summary["episode_length_std"],
    )
    logger.info("[Eval Summary] stop_reason_counts={}", stop_reason_counts)

    writer = getattr(algo, "writer", None)
    if writer is not None:
        for key, value in eval_metrics.items():
            writer.add_scalar(key, value, 0)
        writer.flush()

    return True


def run_eval_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
):
    # Use shared simulation environment setup
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(checkpoint_path)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    exported_policy_dir_path = os.path.join(checkpoint_dir, "exported")
    os.makedirs(exported_policy_dir_path, exist_ok=True)
    exported_policy_name = checkpoint_path.split("/")[-1]  # example: model_5000.pt
    exported_onnx_name = exported_policy_name.replace(".pt", ".onnx")  # example: model_5000.onnx

    if tyro_config.training.export_onnx:
        exported_onnx_path = os.path.join(exported_policy_dir_path, exported_onnx_name)
        if not hasattr(algo, "export"):
            raise AttributeError(
                f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
            )

        algo.export(onnx_file_path=exported_onnx_path)  # type: ignore[attr-defined]
        logger.info(f"Exported policy as onnx to: {exported_onnx_path}")

    used_repeated_eval = _log_repeated_eval_summary(algo, tyro_config)
    if not used_repeated_eval:
        algo.evaluate_policy(
            max_eval_steps=tyro_config.training.max_eval_steps,
        )

    # Cleanup simulation app
    if simulation_app:
        close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    print("overwritten_tyro_config: ", overwritten_tyro_config)
    run_eval_with_tyro(overwritten_tyro_config, checkpoint_cfg, saved_cfg, saved_wandb_path)


if __name__ == "__main__":
    main()
