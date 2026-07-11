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
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _as_detail_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _stop_reason_counts(eval_results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in eval_results:
        stop_reason_value = result.get("stop_reason")
        stop_reason = str(stop_reason_value) if stop_reason_value is not None else "none"
        counts[stop_reason] = counts.get(stop_reason, 0) + 1
    return counts


def _bad_tracking_detail_counts(eval_results: list[dict[str, Any]]) -> tuple[dict[str, int], int, int]:
    counts: dict[str, int] = {}
    bad_tracking_total = 0
    multi_detail_total = 0
    for result in eval_results:
        if result.get("stop_reason") != "bad_tracking":
            continue
        bad_tracking_total += 1
        details = sorted(set(_as_detail_list(result.get("bad_tracking_details"))))
        if not details:
            details = ["unknown"]
        if len(details) > 1:
            multi_detail_total += 1
        for detail in details:
            counts[detail] = counts.get(detail, 0) + 1
    return counts, bad_tracking_total, multi_detail_total


def _bad_tracking_detail_percentages(detail_counts: dict[str, int], denominator: int) -> dict[str, float]:
    if denominator <= 0:
        return {}
    return {detail: 100.0 * float(count) / float(denominator) for detail, count in detail_counts.items()}


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
        # D3 segment runs end successful episodes via segment_ends instead of motion_ends.
        "success_ratio": float(counts.get("motion_ends", 0) + counts.get("segment_ends", 0)) / num_episodes,
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


def _q_gradient_diagnostic_normalizer(normalizer):
    def _normalize(x):
        try:
            return normalizer(x, update=False)
        except TypeError:
            return normalizer(x)

    return _normalize


def _actions_for_critic(algo: BaseAlgo, actions: torch.Tensor) -> torch.Tensor:
    to_critic_actions = getattr(algo, "_to_critic_actions", None)
    if callable(to_critic_actions):
        return to_critic_actions(actions)
    return actions


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    values_f = values.detach().float()
    quantiles = torch.quantile(
        values_f,
        torch.tensor([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99], device=values_f.device),
    )
    return {
        "mean": float(values_f.mean().item()),
        "std": float(values_f.std(unbiased=False).item()) if values_f.numel() > 1 else 0.0,
        "min": float(values_f.min().item()),
        "p01": float(quantiles[0].item()),
        "p05": float(quantiles[1].item()),
        "p10": float(quantiles[2].item()),
        "p25": float(quantiles[3].item()),
        "p50": float(quantiles[4].item()),
        "p75": float(quantiles[5].item()),
        "p90": float(quantiles[6].item()),
        "p95": float(quantiles[7].item()),
        "p99": float(quantiles[8].item()),
        "max": float(values_f.max().item()),
    }


def _format_stats(stats: dict[str, float]) -> str:
    return (
        "mean={mean:.4f} std={std:.4f} min={min:.4f} p01={p01:.4f} p05={p05:.4f} "
        "p10={p10:.4f} p25={p25:.4f} p50={p50:.4f} p75={p75:.4f} p90={p90:.4f} "
        "p95={p95:.4f} p99={p99:.4f} max={max:.4f}"
    ).format(**stats)


def _cosine_histogram(values: torch.Tensor, bins: int = 10) -> list[tuple[float, float, int, float]]:
    if values.numel() == 0:
        return []
    values_cpu = values.detach().float().cpu().clamp(-1.0, 1.0)
    hist = torch.histc(values_cpu, bins=bins, min=-1.0, max=1.0)
    edges = torch.linspace(-1.0, 1.0, bins + 1)
    total = max(1, int(values_cpu.numel()))
    return [
        (float(edges[idx].item()), float(edges[idx + 1].item()), int(hist[idx].item()), 100.0 * float(hist[idx].item()) / total)
        for idx in range(bins)
    ]


def _log_cosine_histogram(prefix: str, values: torch.Tensor) -> None:
    for low, high, count, pct in _cosine_histogram(values):
        logger.info("{} hist [{:.1f}, {:.1f}] count={} percent={:.2f}%", prefix, low, high, count, pct)


def _cosine_from_grad_and_direction(
    grad_actions: torch.Tensor,
    direction: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_norm = grad_actions.norm(dim=1)
    direction_norm = direction.norm(dim=1)
    valid = (grad_norm > eps) & (direction_norm > eps)
    cosine = (grad_actions * direction).sum(dim=1) / (grad_norm * direction_norm).clamp_min(eps)
    return cosine[valid], valid


def _run_q_gradient_diagnostic(algo: BaseAlgo, checkpoint_cfg: CheckpointConfig) -> bool:
    actor = getattr(algo, "actor", None)
    qnet = getattr(algo, "qnet", None)
    sample_offline_batch = getattr(algo, "_sample_offline_batch", None)
    if actor is None or qnet is None or not callable(sample_offline_batch):
        logger.warning(
            "[QGradDiag] skipped: algo={} does not expose actor/qnet/_sample_offline_batch.",
            type(algo).__name__,
        )
        return False

    batch_size = max(1, int(checkpoint_cfg.q_gradient_batch_size))
    num_batches = max(1, int(checkpoint_cfg.q_gradient_num_batches))
    logger.info(
        "[QGradDiag] starting frozen-critic action-gradient diagnostic: batch_size={} num_batches={}",
        batch_size,
        num_batches,
    )

    was_actor_training = bool(getattr(actor, "training", False))
    was_qnet_training = bool(getattr(qnet, "training", False))
    obs_normalizer = getattr(algo, "obs_normalizer", None)
    critic_obs_normalizer = getattr(algo, "critic_obs_normalizer", None)
    was_obs_norm_training = bool(getattr(obs_normalizer, "training", False)) if obs_normalizer is not None else False
    was_critic_obs_norm_training = (
        bool(getattr(critic_obs_normalizer, "training", False)) if critic_obs_normalizer is not None else False
    )

    actor.eval()
    qnet.eval()
    if obs_normalizer is not None:
        obs_normalizer.eval()
    if critic_obs_normalizer is not None:
        critic_obs_normalizer.eval()

    normalize_obs = _q_gradient_diagnostic_normalizer(obs_normalizer) if obs_normalizer is not None else lambda x: x
    normalize_critic_obs = (
        _q_gradient_diagnostic_normalizer(critic_obs_normalizer) if critic_obs_normalizer is not None else lambda x: x
    )

    cosine_values: list[torch.Tensor] = []
    q_pi_values: list[torch.Tensor] = []
    q_data_values: list[torch.Tensor] = []
    grad_norm_values: list[torch.Tensor] = []
    direction_norm_values: list[torch.Tensor] = []
    invalid_count = 0
    total_count = 0
    group_cosines: dict[str, list[torch.Tensor]] = {}
    group_indices = getattr(algo, "bf_cql_group_indices", None)
    group_names = getattr(algo, "bf_cql_group_names", None)
    if group_indices is not None:
        if not group_names:
            group_names = [f"group_{idx}" for idx in range(len(group_indices))]
        group_cosines = {str(name): [] for name in group_names}

    try:
        for _ in range(num_batches):
            data = sample_offline_batch(
                batch_size=batch_size,
                normalize_obs=normalize_obs,
                normalize_critic_obs=normalize_critic_obs,
            )
            observations = data["observations"]
            critic_observations = data["critic_observations"]
            dataset_actions = _actions_for_critic(algo, data["actions"]).detach()

            with torch.no_grad():
                # Actor.forward returns the deterministic policy action in the critic action space.
                pi_actions = actor(observations)[0].detach()

            pi_actions_for_grad = pi_actions.clone().detach().requires_grad_(True)
            q1_pi, q2_pi = qnet(critic_observations, pi_actions_for_grad)
            q_pi = torch.minimum(q1_pi, q2_pi)
            grad_actions = torch.autograd.grad(
                q_pi.sum(),
                pi_actions_for_grad,
                retain_graph=False,
                create_graph=False,
            )[0].detach()

            direction = (dataset_actions - pi_actions).detach()
            cosine, valid = _cosine_from_grad_and_direction(grad_actions, direction)
            cosine_values.append(cosine.detach().cpu())
            invalid_count += int((~valid).sum().item())
            total_count += int(valid.numel())
            grad_norm_values.append(grad_actions.norm(dim=1).detach().cpu())
            direction_norm_values.append(direction.norm(dim=1).detach().cpu())

            with torch.no_grad():
                q1_data, q2_data = qnet(critic_observations, dataset_actions)
                q_data = torch.minimum(q1_data, q2_data)
                q_pi_values.append(q_pi.detach().cpu())
                q_data_values.append(q_data.detach().cpu())

            if group_indices is not None and group_names is not None:
                for group_name, indices in zip(group_names, group_indices, strict=True):
                    idx_tensor = torch.as_tensor(indices, device=direction.device, dtype=torch.long)
                    group_cosine, _ = _cosine_from_grad_and_direction(
                        grad_actions.index_select(1, idx_tensor),
                        direction.index_select(1, idx_tensor),
                    )
                    group_cosines[str(group_name)].append(group_cosine.detach().cpu())
    finally:
        if was_actor_training:
            actor.train()
        if was_qnet_training:
            qnet.train()
        if obs_normalizer is not None and was_obs_norm_training:
            obs_normalizer.train()
        if critic_obs_normalizer is not None and was_critic_obs_norm_training:
            critic_obs_normalizer.train()

    if not cosine_values:
        logger.warning("[QGradDiag] no valid batches were sampled.")
        return False

    cosines = torch.cat(cosine_values)
    q_pi_all = torch.cat(q_pi_values)
    q_data_all = torch.cat(q_data_values)
    grad_norms = torch.cat(grad_norm_values)
    direction_norms = torch.cat(direction_norm_values)
    logger.info("[QGradDiag] valid_samples={} invalid_samples={} total={}", int(cosines.numel()), invalid_count, total_count)
    logger.info("[QGradDiag] global cos(grad_a Q(a_pi), a_D-a_pi): {}", _format_stats(_tensor_stats(cosines)))
    logger.info(
        "[QGradDiag] ratios negative={:.2f}% near_zero(|cos|<0.1)={:.2f}% positive={:.2f}% aligned(cos>0.5)={:.2f}%",
        100.0 * float((cosines < 0.0).float().mean().item()),
        100.0 * float((cosines.abs() < 0.1).float().mean().item()),
        100.0 * float((cosines > 0.0).float().mean().item()),
        100.0 * float((cosines > 0.5).float().mean().item()),
    )
    logger.info("[QGradDiag] q_pi stats: {}", _format_stats(_tensor_stats(q_pi_all)))
    logger.info("[QGradDiag] q_data stats: {}", _format_stats(_tensor_stats(q_data_all)))
    logger.info("[QGradDiag] q_pi_minus_q_data_mean={:.4f}", float((q_pi_all - q_data_all).mean().item()))
    logger.info("[QGradDiag] grad_norm stats: {}", _format_stats(_tensor_stats(grad_norms)))
    logger.info("[QGradDiag] direction_norm stats: {}", _format_stats(_tensor_stats(direction_norms)))
    _log_cosine_histogram("[QGradDiag] global cosine", cosines)

    for group_name, tensors in group_cosines.items():
        if not tensors:
            continue
        group_values = torch.cat(tensors)
        if group_values.numel() == 0:
            continue
        logger.info(
            "[QGradDiag][group={}] cos stats: {} negative={:.2f}% near_zero={:.2f}% aligned={:.2f}%",
            group_name,
            _format_stats(_tensor_stats(group_values)),
            100.0 * float((group_values < 0.0).float().mean().item()),
            100.0 * float((group_values.abs() < 0.1).float().mean().item()),
            100.0 * float((group_values > 0.5).float().mean().item()),
        )

    return True


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
            repeat_detail_counts, repeat_bad_tracking_total, repeat_multi_detail_total = _bad_tracking_detail_counts(
                repeat_results
            )
            if repeat_bad_tracking_total > 0:
                logger.info(
                    "[Eval Repeat] bad_tracking_details={} percent_of_bad_tracking={} multi_detail={}/{}",
                    repeat_detail_counts,
                    _bad_tracking_detail_percentages(repeat_detail_counts, repeat_bad_tracking_total),
                    repeat_multi_detail_total,
                    repeat_bad_tracking_total,
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

    detail_counts, bad_tracking_total, multi_detail_total = _bad_tracking_detail_counts(all_eval_results)
    detail_percentages = _bad_tracking_detail_percentages(detail_counts, bad_tracking_total)
    for detail, count in detail_counts.items():
        all_episode_ratio = float(count) / max(1.0, total_summary["num_episodes"])
        bad_tracking_ratio = float(count) / max(1, bad_tracking_total)
        eval_metrics[f"Eval/bad_tracking_detail/{detail}"] = all_episode_ratio
        eval_metrics[f"Eval/bad_tracking_detail_percent/{detail}"] = 100.0 * all_episode_ratio
        eval_metrics[f"Eval/bad_tracking_detail_among_bad_tracking/{detail}"] = bad_tracking_ratio
        eval_metrics[f"Eval/bad_tracking_detail_percent_among_bad_tracking/{detail}"] = 100.0 * bad_tracking_ratio

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
    if bad_tracking_total > 0:
        logger.info(
            "[Eval Summary] bad_tracking_detail_counts={} percent_of_bad_tracking={} multi_detail={}/{}",
            detail_counts,
            detail_percentages,
            multi_detail_total,
            bad_tracking_total,
        )

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
    if checkpoint_cfg.q_gradient_diagnostic:
        _run_q_gradient_diagnostic(algo, checkpoint_cfg)

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
