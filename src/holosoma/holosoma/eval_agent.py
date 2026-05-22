from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import tyro
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.eval_callback import EvalCallbacksConfig
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


@dataclass(frozen=True)
class EvalRunConfig:
    """CLI options controlling the eval_agent rollout behaviour."""

    single_episode_per_env: bool = False
    """If set, evaluate exactly one episode per env across all parallel envs and stop once every env has finished."""

    save_eval_results: bool = False
    """If set, save per-env results CSV and summary JSON under ``eval_results_dir`` (or the eval log dir)."""

    eval_results_dir: str | None = None
    """Directory where eval results files are written. Defaults to the auto-created eval log directory."""

    # ── Paired diagnostics tagging (no effect on rollout) ────────────
    algo_name: str = "unknown"
    """Free-form tag (e.g. 'smqr_anchor_only', 'vanilla_cql') written into
    each CSV row and the summary JSON so paired analyses across runs can be
    merged on (env_id, checkpoint_step, algo_name)."""

    checkpoint_step: int = -1
    """Training step of the checkpoint being evaluated.  If left at -1 the
    value is auto-parsed from the checkpoint file name (e.g. ``model_5000.pt``
    → 5000) when possible.  Purely a tag — does not affect rollout."""

    # ── Progress diagnostics thresholds (eval-path only) ──────────────
    eval_grasp_radius: float = 0.12
    """Hand ↔ object centre distance [m] below which the episode is marked
    as having achieved a grasp (v1 definition).  Only consulted when the
    active motion clip has an object (``_motion_cmd.motion.has_object``).
    Purely diagnostic."""

    eval_lift_height_margin: float = 0.05
    """Height gain [m] above the per-episode start object z at which the
    episode is marked as having achieved a lift.  Same scope as
    ``eval_grasp_radius``."""

    # ── v2 thresholds (threshold-free where possible, else liberal) ──
    eval_contact_radius: float = 0.18
    """Hand ↔ object distance [m] used by v2 ``first_contact_step``.  Looser
    than ``eval_grasp_radius`` to absorb wrist-frame ≠ palm-centre offsets
    on G1 (tracked wrist frame sits on the forearm).  Diagnostic only."""

    eval_object_moved_thresh: float = 0.02
    """XY displacement [m] of the object w.r.t. its episode-start XY above
    which the v2 diagnostics mark the object as ``moved``.  Used to gate
    ``grasp_achieved_v2 = first_contact AND object_moved`` so the metric
    cannot fire from a mere nearby-hand configuration.  Diagnostic only."""


def _write_eval_results(
    stats: dict,
    out_dir: Path,
    checkpoint_path: str,
    eval_run_cfg: "EvalRunConfig | None" = None,
) -> tuple[Path, Path, dict]:
    """Write per-env CSV and summary JSON; return (csv_path, summary_path, summary).

    The per-episode diagnostics (grasp, lift, object ↔ goal distance,
    bad-tracking timing, action norm) are read from ``stats`` when the
    agent's ``evaluate_policy`` populated them; missing keys fall back to
    sentinel values (``-1`` for integer step-indices, ``NaN`` for real-
    valued quantities) so the CSV shape stays constant across agents.
    """
    import math
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "eval_results.csv"
    summary_path = out_dir / "eval_summary.json"

    num_envs = int(stats["num_envs"])
    rewards = stats["reward"]
    lengths = stats["length"]
    successes = stats["success"]
    reasons = stats["reason"]
    done_flags = stats["done"]
    episode_reward_sums = stats.get("episode_reward_sum", rewards)
    episode_discounted_returns = stats.get("episode_discounted_return", rewards)
    return_gamma = float(stats.get("return_gamma", 0.99))

    # ── Paired-analysis tags ───────────────────────────────────
    _algo_name = getattr(eval_run_cfg, "algo_name", "unknown") if eval_run_cfg else "unknown"
    _ckpt_step = int(getattr(eval_run_cfg, "checkpoint_step", -1)) if eval_run_cfg else -1

    # ── Diagnostic tensors (optional) ───────────────────────────
    def _getf(key: str, default: float = float("nan")) -> list[float]:
        v = stats.get(key)
        if v is None:
            return [default] * num_envs
        return [float(x) for x in v]

    def _geti(key: str, default: int = -1) -> list[int]:
        v = stats.get(key)
        if v is None:
            return [default] * num_envs
        return [int(x) for x in v]

    min_obj2goal      = _getf("min_obj2goal_dist")
    max_obj_height    = _getf("max_obj_height")
    first_grasp_step  = _geti("first_grasp_step")
    first_lift_step   = _geti("first_lift_step")
    bad_tracking_step = _geti("bad_tracking_step")
    action_norm_mean  = _getf("action_norm_mean")
    action_abs_max    = _getf("action_abs_max")

    # ── v2 diagnostic tensors (NaN/-1 when algo did not populate) ──
    initial_obj2goal   = _getf("initial_obj2goal_dist")
    object_xy_disp     = _getf("object_xy_displacement")
    min_hand_obj_d     = _getf("min_hand_obj_dist")
    first_contact_step = _geti("first_contact_step")
    first_approach_step = _geti("first_approach_step")
    hand_obj_d_at_lift = _getf("min_hand_obj_dist_at_lift")
    reward_until_bt    = _getf("reward_until_bad_tracking")
    alive_steps        = _geti("alive_steps", default=0)

    # v2 thresholds
    _object_moved_thresh = float(
        getattr(eval_run_cfg, "eval_object_moved_thresh", 0.02)
    ) if eval_run_cfg else 0.02

    def _fmt(x):
        # CSV-friendly: NaN → empty, everything else passthrough.
        if isinstance(x, float) and math.isnan(x):
            return ""
        return x

    _EPS = 1e-6

    def _goal_progress(i: int) -> float:
        init_d, min_d = initial_obj2goal[i], min_obj2goal[i]
        if math.isnan(init_d) or math.isnan(min_d):
            return float("nan")
        # Clamp at 0 — no credit for the object drifting further than start.
        return max(init_d - min_d, 0.0)

    def _goal_progress_frac(i: int) -> float:
        init_d = initial_obj2goal[i]
        if math.isnan(init_d):
            return float("nan")
        gp = _goal_progress(i)
        if math.isnan(gp):
            return float("nan")
        return gp / max(init_d, _EPS)

    def _object_moved(i: int) -> int:
        v = object_xy_disp[i]
        if math.isnan(v):
            return 0
        return int(v > _object_moved_thresh)

    def _reward_per_alive_step(i: int) -> float:
        n = alive_steps[i]
        if n <= 0:
            return float("nan")
        return float(rewards[i]) / float(n)

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            # paired merge keys
            "env_id", "episode_id", "checkpoint_step", "algo_name",
            # core episode
            "episode_reward", "episode_length", "success", "done_reason", "episode_completed",
            "episode_reward_sum", "episode_discounted_return",
            # v1 diagnostics (preserved)
            "min_obj2goal_dist", "max_obj_height",
            "first_grasp_step", "first_lift_step",
            "grasp_achieved", "lift_achieved",
            "bad_tracking_step",
            "action_norm_mean", "action_abs_max",
            # v2 diagnostics
            "initial_obj2goal_dist", "goal_progress", "goal_progress_frac",
            "object_xy_displacement", "object_moved",
            "min_hand_obj_dist",
            "first_contact_step", "first_approach_step",
            "grasp_achieved_v2", "lift_after_contact", "bad_tracking_after_lift",
            "reward_until_bad_tracking", "reward_per_alive_step",
            "min_hand_obj_dist_at_lift",
            "alive_steps",
        ])
        for i in range(num_envs):
            grasp_ok = int(first_grasp_step[i] >= 0)
            lift_ok = int(first_lift_step[i] >= 0)
            obj_moved = _object_moved(i)
            grasp_v2 = int((first_contact_step[i] >= 0) and bool(obj_moved))
            lift_after = int(
                first_lift_step[i] > 0
                and first_contact_step[i] > 0
                and first_lift_step[i] > first_contact_step[i]
            )
            bt_after_lift = int(lift_after and bad_tracking_step[i] >= 0)
            w.writerow([
                i, i, _ckpt_step, _algo_name,
                float(rewards[i]),
                int(lengths[i]),
                int(bool(successes[i])),
                reasons[i] if reasons[i] else "unknown",
                int(bool(done_flags[i])),
                float(episode_reward_sums[i]),
                float(episode_discounted_returns[i]),
                _fmt(min_obj2goal[i]),
                _fmt(max_obj_height[i]),
                first_grasp_step[i],
                first_lift_step[i],
                grasp_ok,
                lift_ok,
                bad_tracking_step[i],
                _fmt(action_norm_mean[i]),
                _fmt(action_abs_max[i]),
                # v2
                _fmt(initial_obj2goal[i]),
                _fmt(_goal_progress(i)),
                _fmt(_goal_progress_frac(i)),
                _fmt(object_xy_disp[i]),
                obj_moved,
                _fmt(min_hand_obj_d[i]),
                first_contact_step[i],
                first_approach_step[i],
                grasp_v2,
                lift_after,
                bt_after_lift,
                _fmt(reward_until_bt[i]),
                _fmt(_reward_per_alive_step(i)),
                _fmt(hand_obj_d_at_lift[i]),
                int(alive_steps[i]),
            ])

    rewards_np = np.asarray(rewards, dtype=np.float64)
    reward_sums_np = np.asarray(episode_reward_sums, dtype=np.float64)
    discounted_returns_np = np.asarray(episode_discounted_returns, dtype=np.float64)
    lengths_np = np.asarray(lengths, dtype=np.float64)
    success_np = np.asarray(successes, dtype=np.bool_)
    success_count = int(success_np.sum())
    envs_finished = int(sum(1 for d in done_flags if d))
    envs_unfinished = num_envs - envs_finished

    def _percentile(arr, q):
        if arr.size == 0:
            return 0.0
        return float(np.percentile(arr, q))

    def _nan_mean(xs):
        arr = np.asarray(xs, dtype=np.float64)
        m = np.isfinite(arr)
        return float(arr[m].mean()) if m.any() else float("nan")

    def _pos_median(xs):
        arr = np.asarray(xs, dtype=np.float64)
        m = arr >= 0
        return float(np.median(arr[m])) if m.any() else float("nan")

    grasp_rate_v1 = float(np.mean([1.0 if s >= 0 else 0.0 for s in first_grasp_step]))
    lift_rate_v1  = float(np.mean([1.0 if s >= 0 else 0.0 for s in first_lift_step]))

    # v2 aggregates
    goal_progress_list      = [_goal_progress(i) for i in range(num_envs)]
    goal_progress_frac_list = [_goal_progress_frac(i) for i in range(num_envs)]
    object_moved_list       = [_object_moved(i) for i in range(num_envs)]
    grasp_v2_list = [
        int((first_contact_step[i] >= 0) and bool(object_moved_list[i]))
        for i in range(num_envs)
    ]
    lift_after_list = [
        int(
            first_lift_step[i] > 0
            and first_contact_step[i] > 0
            and first_lift_step[i] > first_contact_step[i]
        )
        for i in range(num_envs)
    ]
    bt_after_lift_list = [
        int(lift_after_list[i] and bad_tracking_step[i] >= 0)
        for i in range(num_envs)
    ]
    reward_per_alive_list = [_reward_per_alive_step(i) for i in range(num_envs)]

    first_contact_rate = float(np.mean([1.0 if s >= 0 else 0.0 for s in first_contact_step]))

    action_norms = np.asarray(action_norm_mean, dtype=np.float64)
    action_norms_finite = action_norms[np.isfinite(action_norms)]

    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_step": _ckpt_step,
        "algo_name": _algo_name,
        "num_envs": num_envs,
        # core
        "mean_reward": float(rewards_np.mean()) if num_envs > 0 else 0.0,
        "std_reward": float(rewards_np.std()) if num_envs > 0 else 0.0,
        "min_reward": float(rewards_np.min()) if num_envs > 0 else 0.0,
        "max_reward": float(rewards_np.max()) if num_envs > 0 else 0.0,
        "p10_reward": _percentile(rewards_np, 10.0),
        "p50_reward": _percentile(rewards_np, 50.0),
        "p90_reward": _percentile(rewards_np, 90.0),
        "mean_length": float(lengths_np.mean()) if num_envs > 0 else 0.0,
        "episode_reward_sum_mean": float(reward_sums_np.mean()) if num_envs > 0 else 0.0,
        "episode_reward_sum_std": float(reward_sums_np.std()) if num_envs > 0 else 0.0,
        "episode_reward_sum_min": float(reward_sums_np.min()) if num_envs > 0 else 0.0,
        "episode_reward_sum_max": float(reward_sums_np.max()) if num_envs > 0 else 0.0,
        "episode_discounted_return_mean": float(discounted_returns_np.mean()) if num_envs > 0 else 0.0,
        "episode_discounted_return_std": float(discounted_returns_np.std()) if num_envs > 0 else 0.0,
        "episode_discounted_return_min": float(discounted_returns_np.min()) if num_envs > 0 else 0.0,
        "episode_discounted_return_max": float(discounted_returns_np.max()) if num_envs > 0 else 0.0,
        "episode_length_mean": float(lengths_np.mean()) if num_envs > 0 else 0.0,
        "episode_length_std": float(lengths_np.std()) if num_envs > 0 else 0.0,
        "success_count": success_count,
        "success_rate": float(success_count) / num_envs if num_envs > 0 else 0.0,
        "success_rate_attempted": float(success_count) / num_envs if num_envs > 0 else 0.0,
        "success_rate_finished": float(success_count) / envs_finished if envs_finished > 0 else 0.0,
        "success_std": float(success_np.astype(np.float64).std()) if num_envs > 0 else 0.0,
        "envs_finished": envs_finished,
        "envs_unfinished": envs_unfinished,
        "failure_count": envs_finished - success_count,
        "timeout_count": sum(1 for r in reasons if r == "timeout"),
        "bad_tracking_count": sum(1 for r in reasons if "bad_tracking" in r),
        "max_eval_steps_unfinished_count": sum(1 for r in reasons if r == "max_eval_steps_unfinished"),
        "failure_reason_counts": dict(__import__("collections").Counter(reasons)),
        "success_rate_denominator": "num_envs",
        "success_rate_finished_denominator": "envs_finished",
        "unfinished_envs_counted_as_failure": True,
        "return_gamma": return_gamma,
        # ── v1 diagnostics ────────────────────────────────────
        "grasp_rate_v1": grasp_rate_v1,
        "lift_rate_v1": lift_rate_v1,
        "grasp_rate": grasp_rate_v1,  # alias, backwards compat
        "lift_rate": lift_rate_v1,    # alias, backwards compat
        "mean_min_obj2goal_dist": _nan_mean(min_obj2goal),
        "mean_max_obj_height": _nan_mean(max_obj_height),
        "median_first_grasp_step": _pos_median(first_grasp_step),
        "median_first_lift_step": _pos_median(first_lift_step),
        "median_bad_tracking_step": _pos_median(bad_tracking_step),
        "mean_action_norm": float(action_norms_finite.mean()) if action_norms_finite.size else float("nan"),
        "max_action_norm_p95": float(np.percentile(action_norms_finite, 95.0)) if action_norms_finite.size else float("nan"),
        # ── v2 diagnostics ────────────────────────────────────
        "mean_initial_obj2goal_dist": _nan_mean(initial_obj2goal),
        "mean_goal_progress":         _nan_mean(goal_progress_list),
        "mean_goal_progress_frac":    _nan_mean(goal_progress_frac_list),
        "median_goal_progress_frac":  float(np.nanmedian(np.asarray(goal_progress_frac_list, dtype=np.float64))) if num_envs > 0 else float("nan"),
        "mean_object_xy_displacement": _nan_mean(object_xy_disp),
        "object_moved_rate":          float(np.mean(object_moved_list)) if num_envs > 0 else 0.0,
        "mean_min_hand_obj_dist":     _nan_mean(min_hand_obj_d),
        "first_contact_rate":         first_contact_rate,
        "median_first_contact_step":  _pos_median(first_contact_step),
        "median_first_approach_step": _pos_median(first_approach_step),
        "grasp_rate_v2":              float(np.mean(grasp_v2_list)) if num_envs > 0 else 0.0,
        "lift_after_contact_rate":    float(np.mean(lift_after_list)) if num_envs > 0 else 0.0,
        "bad_tracking_after_lift_rate": float(np.mean(bt_after_lift_list)) if num_envs > 0 else 0.0,
        "mean_reward_until_bad_tracking": _nan_mean(reward_until_bt),
        "mean_reward_per_alive_step":     _nan_mean(reward_per_alive_list),
        "mean_min_hand_obj_dist_at_lift": _nan_mean(hand_obj_d_at_lift),
        "csv_path": str(csv_path),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return csv_path, summary_path, summary


def run_eval_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    eval_cbs_cfg: EvalCallbacksConfig | None = None,
    eval_run_cfg: EvalRunConfig | None = None,
):
    # Use shared simulation environment setup
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    # Inject eval callbacks into algo config
    if eval_cbs_cfg is not None:
        cb_configs = eval_cbs_cfg.collect_active_callbacks()
        if cb_configs:
            object.__setattr__(tyro_config.algo.config, "eval_callbacks", cb_configs)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    algo.setup(eval_only=True, checkpoint_path=checkpoint_path)
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

    # Enable per-env single-episode tracking if requested
    if eval_run_cfg is not None and eval_run_cfg.single_episode_per_env:
        setattr(algo, "_single_episode_per_env", True)
        logger.info(
            f"[Eval] single_episode_per_env=True — will stop once all "
            f"{tyro_config.training.num_envs} envs finish one episode."
        )

    # Expose the diagnostics config to the agent's evaluate_policy loop.
    # ``evaluate_policy`` reads these via ``getattr(self, '_eval_diagnostics_cfg', None)``
    # with safe defaults, so agents that do not support diagnostics are
    # unaffected.
    if eval_run_cfg is not None:
        # Auto-derive checkpoint_step from filename if user left it at -1,
        # e.g. ``.../model_5000.pt`` → 5000, ``.../checkpoint_10000.pt`` → 10000.
        _resolved_step = int(eval_run_cfg.checkpoint_step)
        if _resolved_step < 0:
            import re as _re
            _m = _re.search(r"(\d+)(?=\.(?:pt|ckpt|safetensors))", checkpoint_path)
            if _m is not None:
                _resolved_step = int(_m.group(1))
        # Pack the resolved step back for downstream consumers.
        import dataclasses as _dc_eval
        _resolved_run_cfg = _dc_eval.replace(eval_run_cfg, checkpoint_step=_resolved_step)
        setattr(algo, "_eval_diagnostics_cfg", _resolved_run_cfg)
    else:
        _resolved_run_cfg = None

    algo.evaluate_policy(
        max_eval_steps=tyro_config.training.max_eval_steps,
    )

    # ── Per-env summary (single-episode-per-env mode) ──────────────
    # ── Per-env summary — always reported when stats are available ──────────
    # _last_per_env_stats is populated by run_evaluate_policy() for both
    # single_episode_per_env=True and single_episode_per_env=False modes.
    stats = getattr(algo, "_last_per_env_stats", None)
    if stats is None:
        if eval_run_cfg is not None and eval_run_cfg.single_episode_per_env:
            logger.warning(
                "[Eval] single_episode_per_env requested but agent did not "
                "produce per-env stats. The current algo may not support it."
            )
        else:
            logger.warning(
                "[Eval] No per-env stats available — the agent's evaluate_policy "
                "did not populate _last_per_env_stats."
            )
    else:
        import numpy as _np

        num_envs = int(stats["num_envs"])
        rewards_np = _np.asarray(stats["reward"], dtype=_np.float64)
        success_count = int(
            stats.get(
                "success_count",
                int(_np.asarray(stats["success"], dtype=_np.bool_).sum()),
            )
        )
        envs_finished = int(stats.get("envs_finished", sum(1 for d in stats["done"] if d)))
        envs_unfinished = int(stats.get("envs_unfinished", num_envs - envs_finished))
        success_rate = float(
            stats.get("success_rate", success_count / num_envs if num_envs else 0.0)
        )
        success_rate_finished = float(
            stats.get(
                "success_rate_finished",
                success_count / envs_finished if envs_finished else 0.0,
            )
        )
        bad_tracking_count = int(stats.get("bad_tracking_count", 0))
        timeout_count = int(stats.get("timeout_count", 0))
        max_eval_steps_unfinished_count = int(stats.get("max_eval_steps_unfinished_count", 0))
        mean_r = float(rewards_np.mean()) if num_envs else 0.0
        std_r = float(rewards_np.std()) if num_envs else 0.0
        summary_line = (
            f"[Eval Summary] "
            f"num_envs={num_envs}  "
            f"mean_reward={mean_r:.3f}  std_reward={std_r:.3f}  "
            f"success_count={success_count}  "
            f"success_rate={success_rate * 100.0:.2f}%  "
            f"[denominator=num_envs]  "
            f"envs_finished={envs_finished}  "
            f"envs_unfinished={envs_unfinished}  "
            f"success_rate_finished={success_rate_finished * 100.0:.2f}%  "
            f"bad_tracking={bad_tracking_count}  "
            f"timeout={timeout_count}  "
            f"max_eval_steps_unfinished={max_eval_steps_unfinished_count}"
        )
        logger.info(summary_line)
        print(summary_line)

        if eval_run_cfg is not None and eval_run_cfg.save_eval_results:
            out_dir = Path(
                eval_run_cfg.eval_results_dir
                if eval_run_cfg.eval_results_dir
                else eval_log_dir
            )
            csv_path, summary_path, _summary = _write_eval_results(
                stats,
                out_dir,
                checkpoint_path,
                eval_run_cfg=_resolved_run_cfg if _resolved_run_cfg is not None else eval_run_cfg,
            )
            logger.info(f"[Eval] Per-env CSV written to: {csv_path}")
            logger.info(f"[Eval] Summary JSON written to: {summary_path}")

    # Cleanup simulation app
    if simulation_app:
        close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    eval_cbs_cfg, remaining_args = tyro.cli(
        EvalCallbacksConfig, return_unknown_args=True, add_help=False, args=remaining_args
    )
    eval_run_cfg, remaining_args = tyro.cli(
        EvalRunConfig, return_unknown_args=True, add_help=False, args=remaining_args
    )
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )

    # Re-apply eval_overrides to training so that user-supplied
    # `--eval-overrides.num-envs` / `--eval-overrides.headless` on the CLI
    # actually propagate to `training.num_envs` / `training.headless`.
    # (get_eval_config() was already called above on `saved_cfg`, before
    # the user overrides were parsed, so training.* would otherwise keep
    # the values baked in from the saved eval_overrides defaults.)
    import dataclasses as _dc

    overwritten_tyro_config = _dc.replace(
        overwritten_tyro_config,
        training=_dc.replace(
            overwritten_tyro_config.training,
            num_envs=overwritten_tyro_config.eval_overrides.num_envs,
            headless=overwritten_tyro_config.eval_overrides.headless,
        ),
    )
    logger.info(
        f"[Eval] Effective training.num_envs={overwritten_tyro_config.training.num_envs}, "
        f"training.headless={overwritten_tyro_config.training.headless} "
        f"(from eval_overrides)."
    )

    run_eval_with_tyro(
        overwritten_tyro_config,
        checkpoint_cfg,
        saved_cfg,
        saved_wandb_path,
        eval_cbs_cfg=eval_cbs_cfg,
        eval_run_cfg=eval_run_cfg,
    )


if __name__ == "__main__":
    main()
