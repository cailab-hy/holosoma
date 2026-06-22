from __future__ import annotations

import dataclasses
import logging
import os
import statistics
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypedDict, cast

import tqdm
import tyro
from loguru import logger

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    init_sim_imports,
    load_checkpoint,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG


class TrainingContext:
    """Context manager for training lifecycle and resource management."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.simulation_app: Any | None = None

    def __enter__(self):
        # Initialize simulation app
        self.simulation_app = init_sim_imports(self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean shutdown using the utility function
        close_simulation_app(self.simulation_app)

    def train(self) -> None:
        """Train using this context's sim app."""
        train(self.config, training_context=self)


@contextmanager
def training_context(config: ExperimentConfig):
    """Context manager function for training."""
    with TrainingContext(config) as ctx:
        yield ctx


class MultGPUConfig(TypedDict):
    global_rank: int
    local_rank: int
    world_size: int


def configure_multi_gpu() -> MultGPUConfig | None:
    """Configure multi-gpu training and return configuration dictionary, or `None` if single-GPU training."""
    import torch

    gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
    is_distributed = gpu_world_size > 1

    if not is_distributed:
        return None

    gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
    gpu_global_rank = int(os.getenv("RANK", "0"))

    if gpu_local_rank >= gpu_world_size:
        raise ValueError(f"Local rank '{gpu_local_rank}' is greater than or equal to world size '{gpu_world_size}'.")

    if gpu_global_rank >= gpu_world_size:
        raise ValueError(f"Global rank '{gpu_global_rank}' is greater than or equal to world size '{gpu_world_size}'.")

    torch.distributed.init_process_group(backend="nccl", rank=gpu_global_rank, world_size=gpu_world_size)
    torch.cuda.set_device(gpu_local_rank)

    multi_gpu_config: MultGPUConfig = {
        "global_rank": gpu_global_rank,
        "local_rank": gpu_local_rank,
        "world_size": gpu_world_size,
    }
    logger.info(f"Running with multi-GPU parameters: {multi_gpu_config}")

    return multi_gpu_config


def get_device(config, distributed_conf: MultGPUConfig | None) -> str:
    import torch

    is_config_device_specified = hasattr(config, "device") and config.device is not None
    is_multi_gpu = distributed_conf is not None

    if is_config_device_specified:
        if is_multi_gpu and config.device != cast("dict", distributed_conf)["local_rank"]:
            raise ValueError(
                f"Device specified in config ({config.device}) \
                              does not match expected local rank {cast('dict', distributed_conf)['local_rank']}"
            )
        device = config.device
    elif is_multi_gpu:
        device = f"cuda:{cast('dict', distributed_conf)['local_rank']}"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return device


def configure_logging(distributed_conf: MultGPUConfig | None = None, log_dir: Path | None = None):
    # Configure logging.
    from holosoma.utils.logging import LoguruLoggingBridge

    logger.remove()
    is_main_process = distributed_conf is None or distributed_conf["global_rank"] == 0

    # logging to file (from all ranks)
    if log_dir is not None:
        fname = f"train_rank_{distributed_conf['global_rank']:02d}.log" if distributed_conf is not None else "train.log"
        log_path = log_dir / fname
        logger.add(str(log_path), level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default in rank0
    if is_main_process:
        console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    else:
        console_log_level = "ERROR"
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logging.basicConfig(level=logging.DEBUG if is_main_process else logging.ERROR)
    logging.getLogger().addHandler(LoguruLoggingBridge())


def train(tyro_config: ExperimentConfig, training_context: TrainingContext | None = None) -> None:
    """Train an agent with optional context for sim app management.

    Parameters
    ----------
    training_context : Optional[TrainingContext]
        Optional training context with pre-initialized sim app.
        If None, creates and manages sim app automatically.
    """

    if training_context is not None:
        # Use the context's pre-initialized sim app
        simulation_app = training_context.simulation_app
        auto_close = False  # Context will handle closing
    else:
        # Default behavior - create and manage sim app ourselves
        simulation_app = init_sim_imports(tyro_config)
        auto_close = True

    try:
        # have to import torch after isaacgym
        import torch  # noqa: F401
        import torch.distributed as dist
        import wandb

        from holosoma.agents.base_algo.base_algo import BaseAlgo
        from holosoma.utils.common import seeding

        # unresolved_conf = dataclasses.asdict(tyro_config)
        # import ipdb; ipdb.set_trace()

        # Initialize process group
        distributed_conf: MultGPUConfig | None = configure_multi_gpu()
        device: str = get_device(tyro_config, distributed_conf)
        is_distributed = distributed_conf is not None
        is_main_process = distributed_conf is None or distributed_conf["local_rank"] == 0

        # Configure logger
        logger_cfg = tyro_config.logger
        wandb_enabled = logger_cfg.type == "wandb"

        # Compute experiment directory from logger and training config
        from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp

        timestamp = get_timestamp()
        experiment_dir = get_experiment_dir(logger_cfg, tyro_config.training, timestamp, task_name="locomotion")

        # Configure logging with experiment directory
        configure_logging(distributed_conf=distributed_conf, log_dir=experiment_dir)

        # Random seed
        seed = tyro_config.training.seed
        if distributed_conf is not None:
            seed += distributed_conf["global_rank"]
        seeding(seed, torch_deterministic=tyro_config.training.torch_deterministic)

        wandb_run_path: str | None = None

        # Configure wandb in rank 0
        if wandb_enabled and is_main_process:
            from holosoma.config_types.logger import WandbLoggerConfig

            assert isinstance(logger_cfg, WandbLoggerConfig), (
                "Logger config must be WandbLoggerConfig when type is wandb"
            )
            wandb_cfg = logger_cfg
            # Use training config for project/name, fallback to logger config, then defaults
            default_project = tyro_config.training.project or wandb_cfg.project or "default_project"
            default_run_name = (
                f"{timestamp}_{tyro_config.training.name or 'run'}_"
                f"{wandb_cfg.group or 'default'}_{tyro_config.robot.asset.robot_type}"
            )
            wandb_dir = Path(wandb_cfg.dir or (experiment_dir / ".wandb"))
            wandb_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Saving wandb logs to {wandb_dir}")

            # Only pass optional parameters when specified so wandb can fall back to environment defaults.
            wandb_kwargs: dict[str, Any] = {
                "project": wandb_cfg.project or default_project,
                "name": wandb_cfg.name or default_run_name,
                "config": dataclasses.asdict(tyro_config),
                "dir": str(wandb_dir),
                "mode": wandb_cfg.mode,
            }
            if wandb_cfg.entity:
                wandb_kwargs["entity"] = wandb_cfg.entity
            if wandb_cfg.group:
                wandb_kwargs["group"] = wandb_cfg.group
            if wandb_cfg.id:
                wandb_kwargs["id"] = wandb_cfg.id
            if wandb_cfg.tags:
                wandb_kwargs["tags"] = list(wandb_cfg.tags)
            if wandb_cfg.resume is not None:
                wandb_kwargs["resume"] = wandb_cfg.resume

            wandb.init(**wandb_kwargs)
            if wandb.run is not None:
                wandb_run_path = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"

        # Distribute environments across GPUs for proper multi-GPU training
        if distributed_conf is not None:
            original_num_envs = tyro_config.training.num_envs
            num_envs = original_num_envs // distributed_conf["world_size"]
            tyro_config = dataclasses.replace(
                tyro_config, training=dataclasses.replace(tyro_config.training, num_envs=num_envs)
            )
            logger.info(
                f"Distributed training: GPU {distributed_conf['global_rank']} will run {tyro_config.training.num_envs} "
                f"environments (total across all GPUs: {original_num_envs})"
            )

        env_target = tyro_config.env_class

        tyro_env_config = get_tyro_env_config(tyro_config)
        env = get_class(env_target)(tyro_env_config, device=device)

        # For manager system, pre-process config AFTER env creation
        # (need managers to compute dims)
        observation_manager = getattr(env, "observation_manager", None)
        if observation_manager is None:
            raise RuntimeError(
                f"Manager environment {env_target} is missing observation_manager attribute. "
                "This should not happen if the environment is properly configured."
            )

        experiment_save_dir = experiment_dir
        experiment_save_dir.mkdir(exist_ok=True, parents=True)

        if is_main_process:
            logger.info(f"Saving config file to {experiment_save_dir}")
            config_path = experiment_save_dir / CONFIG_NAME
            tyro_config.save_config(str(config_path))
            if wandb_enabled:
                wandb.save(str(config_path), base_path=experiment_save_dir)

        algo_class = get_class(tyro_config.algo._target_)
        algo: BaseAlgo = algo_class(
            device=device,
            env=env,
            config=tyro_config.algo.config,
            log_dir=experiment_save_dir,
            multi_gpu_cfg=distributed_conf,
        )
        algo.setup()
        algo.attach_checkpoint_metadata(tyro_config, wandb_run_path)
        if tyro_config.training.checkpoint is not None:
            loaded_checkpoint = load_checkpoint(tyro_config.training.checkpoint, str(experiment_save_dir))
            tyro_config = dataclasses.replace(
                tyro_config, training=dataclasses.replace(tyro_config.training, checkpoint=str(loaded_checkpoint))
            )
            algo.load(loaded_checkpoint)

        offline_learn_fn = getattr(algo, "offline_learn", None)
        if not callable(offline_learn_fn):
            raise RuntimeError(
                f"Selected algorithm '{tyro_config.algo._target_}' does not implement offline_learn(). "
                "Use an offline algorithm config (e.g., o2o cql, cal-ql)."
            )
        collect_online_steps_fn = getattr(algo, "collect_online_steps", None)
        if not callable(collect_online_steps_fn):
            raise RuntimeError(
                f"Selected algorithm '{tyro_config.algo._target_}' does not implement collect_online_steps(). "
                "Use an offline algorithm config (e.g., o2o cql, cal-ql)."
            )
        online_learn_fn = getattr(algo, "online_learn", None)
        if not callable(online_learn_fn):
            raise RuntimeError(
                f"Selected algorithm '{tyro_config.algo._target_}' does not implement online_learn_fn(). "
                "Use an offline algorithm config (e.g., o2o cql, cal-ql)."
            )

        evaluate_vectorized_episodes_fn = getattr(algo, "evaluate_vectorized_episodes", None)
        evaluate_one_episode_fn = getattr(algo, "evaluate_one_episode", None)
        max_eval_steps = tyro_config.training.max_eval_steps if tyro_config.training.max_eval_steps is not None else 1000
        eval_num_episodes = max(1, int(tyro_config.training.eval_num_episodes))

        def _get_online_replay_size() -> int | None:
            online_replay_buffer = getattr(algo, "online_replay_buffer", None)
            if online_replay_buffer is None:
                return None
            try:
                return int(len(online_replay_buffer))
            except TypeError:
                return None

        def _log_scalar_dict(metrics: dict[str, float]) -> None:
            if not is_main_process:
                return
            writer = getattr(algo, "writer", None)
            if writer is not None:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, algo.global_step)
            if wandb.run is not None:
                wandb.log(dict(metrics, global_step=algo.global_step), step=algo.global_step)

        def _run_eval_if_available(phase: str) -> None:
            if callable(evaluate_vectorized_episodes_fn) or callable(evaluate_one_episode_fn):
                eval_results: list[dict[str, Any]] = []
                if callable(evaluate_vectorized_episodes_fn):
                    for _ in range(eval_num_episodes):
                        eval_batch_results = evaluate_vectorized_episodes_fn(
                            max_eval_steps=max_eval_steps,
                            use_early_termination=False,
                        )
                        if isinstance(eval_batch_results, list):
                            eval_results.extend(
                                result for result in eval_batch_results if isinstance(result, dict)
                            )
                        elif isinstance(eval_batch_results, dict):
                            eval_results.append(eval_batch_results)
                elif callable(evaluate_one_episode_fn):
                    for _ in range(eval_num_episodes):
                        eval_result = evaluate_one_episode_fn(
                            max_eval_steps=max_eval_steps,
                            use_early_termination=False,
                        )
                        if isinstance(eval_result, dict):
                            eval_results.append(eval_result)

                if eval_results:
                    episode_returns = [float(result.get("episode_return", 0.0)) for result in eval_results]
                    episode_lengths = [float(result.get("episode_length", 0.0)) for result in eval_results]
                    stop_reason_counts: dict[str, int] = {}
                    for result in eval_results:
                        stop_reason_value = result.get("stop_reason")
                        stop_reason = str(stop_reason_value) if stop_reason_value is not None else "none"
                        stop_reason_counts[stop_reason] = stop_reason_counts.get(stop_reason, 0) + 1

                    eval_metrics: dict[str, float] = {
                        "Eval/episode_return_mean": float(statistics.fmean(episode_returns)),
                        "Eval/episode_return_std": float(statistics.pstdev(episode_returns))
                        if len(episode_returns) > 1
                        else 0.0,
                        "Eval/episode_return_min": float(min(episode_returns)),
                        "Eval/episode_return_max": float(max(episode_returns)),
                        "Eval/episode_length_mean": float(statistics.fmean(episode_lengths)),
                        "Eval/episode_length_std": float(statistics.pstdev(episode_lengths))
                        if len(episode_lengths) > 1
                        else 0.0,
                        "Eval/episode_length_min": float(min(episode_lengths)),
                        "Eval/episode_length_max": float(max(episode_lengths)),
                        "Eval/num_episodes": float(len(eval_results)),
                        "O2O/online_env_steps": float(getattr(algo, "online_env_steps", 0)),
                        "O2O/online_update_steps": float(getattr(algo, "online_update_steps", 0)),
                        "O2O/episode_return_mean": float(statistics.fmean(episode_returns)),
                        "O2O/episode_return_min": float(min(episode_returns)),
                        "O2O/episode_return_max": float(max(episode_returns)),
                        "O2O/episode_length_mean": float(statistics.fmean(episode_lengths)),
                        "O2O/episode_length_min": float(min(episode_lengths)),
                        "O2O/episode_length_max": float(max(episode_lengths)),
                        "O2O/phase_code": 1.0 if phase == "online" else 0.0,
                        "O2O/source_code": 0.0,
                    }
                    online_replay_size = _get_online_replay_size()
                    if online_replay_size is not None:
                        eval_metrics["O2O/online_buffer_size"] = float(online_replay_size)
                    for stop_reason, count in stop_reason_counts.items():
                        stop_reason_ratio = float(count) / float(len(eval_results))
                        eval_metrics[f"Eval/stop_reason/{stop_reason}"] = stop_reason_ratio
                        eval_metrics[f"Eval/stop_reason_percent/{stop_reason}"] = 100.0 * stop_reason_ratio

                    logger.info(
                        "[Eval:{}] step={} episodes={} return_mean={:.4f} return_std={:.4f} "
                        "return_min={:.4f} return_max={:.4f} length_mean={:.2f} "
                        "length_min={:.2f} length_max={:.2f}",
                        phase,
                        algo.global_step,
                        len(eval_results),
                        eval_metrics["Eval/episode_return_mean"],
                        eval_metrics["Eval/episode_return_std"],
                        eval_metrics["Eval/episode_return_min"],
                        eval_metrics["Eval/episode_return_max"],
                        eval_metrics["Eval/episode_length_mean"],
                        eval_metrics["Eval/episode_length_min"],
                        eval_metrics["Eval/episode_length_max"],
                    )

                    _log_scalar_dict(eval_metrics)

        offline_target_step = int(algo.config.offline_pretrain_steps)
        if algo.global_step < offline_target_step:
            logger.info(f"Starting offline pretraining until global_step={offline_target_step}")
        while algo.global_step < offline_target_step:
            remaining_steps = offline_target_step - algo.global_step
            chunk_steps = min(max(1, int(algo.config.eval_interval)), remaining_steps)
            offline_learn_fn(max_steps=chunk_steps)
            _run_eval_if_available(phase="offline")

        warmup_target_steps = int(algo.config.online_warmup_steps)
        if warmup_target_steps > 0:
            logger.info(f"Collecting online replay warmup: target_env_steps={warmup_target_steps}")
        while int(getattr(algo, "online_env_steps", 0)) < warmup_target_steps:
            remaining_warmup = warmup_target_steps - int(getattr(algo, "online_env_steps", 0))
            collect_steps = min(max(1, int(algo.config.online_collect_steps)), remaining_warmup)
            collected = collect_online_steps_fn(num_steps=collect_steps)
            if collected <= 0:
                raise RuntimeError("collect_online_steps() returned 0 during warmup; cannot fill online replay.")

        online_env_steps = int(getattr(algo, "online_env_steps", 0))
        online_replay_size = _get_online_replay_size()
        if algo.global_step < offline_target_step:
            raise RuntimeError(
                "O2O transition mismatch: online phase requested before offline pretraining target. "
                f"global_step={algo.global_step}, offline_target_step={offline_target_step}"
            )
        if online_env_steps < warmup_target_steps:
            raise RuntimeError(
                "O2O transition mismatch: online replay warmup did not finish. "
                f"online_env_steps={online_env_steps}, warmup_target_steps={warmup_target_steps}"
            )
        if warmup_target_steps > 0 and online_replay_size is not None:
            online_buffer_size = int(getattr(algo.config, "online_buffer_size", online_replay_size))
            num_envs = int(getattr(algo.env, "num_envs", 1))
            requires_complete_episodes = bool(getattr(algo, "online_replay_requires_complete_episodes", False))
            expected_min_replay_size = 1 if requires_complete_episodes else min(
                online_buffer_size,
                warmup_target_steps * num_envs,
            )
            if online_replay_size < expected_min_replay_size:
                raise RuntimeError(
                    "O2O transition mismatch: online replay contains fewer samples than warmup should produce. "
                    f"online_replay_size={online_replay_size}, expected_min={expected_min_replay_size}, "
                    f"warmup_target_steps={warmup_target_steps}, num_envs={num_envs}"
                )

        transition_metrics = {
            "O2O/phase_code": 1.0,
            "O2O/offline_pretrain_done": 1.0,
            "O2O/online_env_steps": float(online_env_steps),
            "O2O/online_update_steps": float(getattr(algo, "online_update_steps", 0)),
        }
        if online_replay_size is not None:
            transition_metrics["O2O/online_buffer_size"] = float(online_replay_size)
        logger.info(
            "O2O transition check passed: global_step={} online_env_steps={} online_replay_size={}",
            algo.global_step,
            online_env_steps,
            online_replay_size if online_replay_size is not None else "unknown",
        )
        _log_scalar_dict(transition_metrics)

        online_start_step = int(algo.global_step)
        online_target_step = online_start_step + int(algo.config.online_total_steps)
        online_eval_interval = max(0, int(getattr(algo.config, "online_eval_interval", 0)))
        next_online_eval_step = (
            online_start_step + online_eval_interval
            if online_eval_interval > 0
            else online_target_step + 1
        )
        logger.info(
            f"Starting online finetuning from global_step={online_start_step} "
            f"to global_step={online_target_step} "
            f"with online_eval_interval={online_eval_interval}"
        )
        online_pbar = tqdm.tqdm(
            total=online_target_step,
            initial=online_start_step,
            desc="O2O online",
            dynamic_ncols=True,
            disable=not is_main_process,
        )
        try:
            while algo.global_step < online_target_step:
                step_before_update = int(algo.global_step)
                collect_online_steps_fn(num_steps=max(1, int(algo.config.online_collect_steps)))
                remaining_online_updates = online_target_step - algo.global_step
                update_steps = min(max(1, int(algo.config.updates_per_collect)), remaining_online_updates)
                online_learn_fn(max_steps=update_steps)
                online_pbar.update(max(0, int(algo.global_step) - step_before_update))
                if algo.global_step >= next_online_eval_step:
                    _run_eval_if_available(phase="online")
                    while next_online_eval_step <= algo.global_step:
                        next_online_eval_step += online_eval_interval
        finally:
            online_pbar.close()

        if is_main_process:
            logger.info(f"Saving final O2O model at global step {algo.global_step}")
            algo.save(os.path.join(experiment_save_dir, f"model_{algo.global_step:07d}.pt"))
            export_fn = getattr(algo, "export", None)
            if callable(export_fn):
                export_fn(onnx_file_path=os.path.join(experiment_save_dir, f"model_{algo.global_step:07d}.onnx"))

        # teardown wandb before SimApp closes ungracefully (IsaacLab)
        if is_main_process and wandb_enabled:
            logger.info("Shutting down wandb...")
            wandb.teardown()

        # shutdown dist before SimApp closes ungracefully (IsaacLab)
        if is_distributed:
            logger.info("Shutting down distributed processes...")
            dist.destroy_process_group()
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Exception occurred during training: {e}\n{tb_str}")
        sys.exit(1)  # manually set exit code, not possible via isaacsim app.close()
    finally:
        if auto_close:
            close_simulation_app(simulation_app)

    logger.info("Training shutdown complete.")


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    print(tyro_cfg.curriculum)
    train(tyro_cfg)


if __name__ == "__main__":
    main()
