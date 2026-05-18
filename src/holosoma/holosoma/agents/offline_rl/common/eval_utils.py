"""Algorithm-neutral evaluation helpers for offline-RL agents.

The helpers in this module are the canonical evaluation surface used by
``CQLAgent``, ``SMQRAgent`` and ``SMQRSGAgent``.  They operate on the
shared ``OfflineRLAgentBase`` attribute contract and do not import or
call legacy ``OfflineCQLAgent`` descriptors.

Re-exports (free-function form)
-------------------------------
* ``run_evaluate_policy(agent, max_eval_steps)``
* ``run_eval_rollouts(agent, *args, **kwargs)``
* ``create_eval_callbacks(agent)``
* ``pre_evaluate_policy(agent)``
* ``post_evaluate_policy(agent)``
* ``pre_eval_env_step(agent, actor_state)``
* ``post_eval_env_step(agent, actor_state)``
"""

from __future__ import annotations

import itertools
from typing import Any

from loguru import logger

from holosoma.utils.helpers import instantiate
from holosoma.utils.safe_torch_import import torch


def run_evaluate_policy(
    agent: Any, max_eval_steps: int | None = None
) -> dict[str, float]:
    """Run the deterministic callback-compatible evaluation loop."""
    if not agent._eval_dims_match:
        raise RuntimeError(
            f"Cannot run evaluate_policy(): observation dimension mismatch "
            f"between the trained model and the current environment.\n"
            f"  model actor_obs_dim  = {agent.actor_obs_dim}\n"
            f"  env   actor_obs_dim  = {agent._env_actor_obs_dim}\n"
            f"  model critic_obs_dim = {agent.critic_obs_dim}\n"
            f"  env   critic_obs_dim = {agent._env_critic_obs_dim}\n"
            f"The checkpoint was trained with a different observation "
            f"config than the current env produces."
        )

    create_eval_callbacks(agent)
    pre_evaluate_policy(agent)

    was_training = agent.actor.training
    agent.actor.eval()
    if agent.obs_normalization:
        agent.obs_normalizer.eval()

    env = agent.env
    obs = env.reset()

    total_reward = torch.zeros(env.num_envs, device=agent.device)
    ep_reward_sums = torch.zeros(env.num_envs, device=agent.device)
    ep_lengths = torch.zeros(env.num_envs, device=agent.device)
    completed_ep_rewards: list[float] = []
    completed_ep_lengths: list[float] = []
    all_actions: list[Any] = []
    episode_signals: dict[str, list[float]] = {}

    single_episode_per_env = bool(getattr(agent, "_single_episode_per_env", False))
    per_env_done = torch.zeros(env.num_envs, dtype=torch.bool, device=agent.device)
    per_env_reward = torch.zeros(env.num_envs, device=agent.device)
    per_env_length = torch.zeros(env.num_envs, device=agent.device)
    per_env_success = torch.zeros(env.num_envs, dtype=torch.bool, device=agent.device)
    per_env_reason: list[str] = ["" for _ in range(env.num_envs)]

    _motion_cmd = getattr(
        getattr(agent.unwrapped_env, "command_manager", None),
        "get_state",
        lambda _: None,
    )("motion_command")

    step = -1
    for step in itertools.islice(itertools.count(), max_eval_steps):
        if agent.obs_normalization:
            normalized_obs = agent.obs_normalizer(obs, update=False)
        else:
            normalized_obs = obs

        actions, _pre_tanh_mean, _log_std = agent.actor(normalized_obs)
        actor_state = {"step": step, "actions": actions, "obs": obs}
        actor_state = pre_eval_env_step(agent, actor_state)

        _prev_motion_ts = (
            _motion_cmd.time_steps.clone() if _motion_cmd is not None else None
        )
        obs, rewards, dones, extras = env.step(actor_state["actions"])

        actor_state["obs"] = obs
        actor_state = post_eval_env_step(agent, actor_state)
        all_actions.append(actor_state["actions"])

        if single_episode_per_env:
            alive_f = (~per_env_done).float()
            total_reward += rewards * alive_f
            ep_reward_sums += rewards * alive_f
            ep_lengths += alive_f
        else:
            total_reward += rewards
            ep_reward_sums += rewards
            ep_lengths += 1

        def _finish_episodes(indices: Any, reason: str) -> None:
            if single_episode_per_env:
                keep = ~per_env_done[indices]
                indices = indices[keep]
                if indices.numel() == 0:
                    return
                per_env_reward[indices] = ep_reward_sums[indices]
                per_env_length[indices] = ep_lengths[indices]
                per_env_success[indices] = reason == "success"
                for idx in indices:
                    per_env_reason[int(idx.item())] = reason
                per_env_done[indices] = True
            for idx in indices:
                i = idx.item()
                completed_ep_rewards.append(ep_reward_sums[idx].item())
                completed_ep_lengths.append(ep_lengths[idx].item())
                ep_num = len(completed_ep_rewards)
                logger.info(
                    f"[Eval] Episode {ep_num} ended  "
                    f"(env={i}, steps={int(ep_lengths[idx].item())}, "
                    f"reward={ep_reward_sums[idx].item():.2f})  "
                    f"reason: {reason}"
                )
            ep_reward_sums[indices] = 0.0
            ep_lengths[indices] = 0.0

        done_mask = dones.bool()
        if done_mask.any():
            done_indices = done_mask.nonzero(as_tuple=False).squeeze(-1)
            term_mgr = getattr(getattr(env, "_env", None), "termination_manager", None)
            for idx in done_indices:
                i = idx.item()
                reason_str = "unknown"
                if term_mgr is not None and hasattr(term_mgr, "active_terms"):
                    reasons = [
                        name
                        for name, mask in term_mgr.active_terms.items()
                        if mask[i].item()
                    ]
                    if reasons:
                        reason_str = ", ".join(reasons)
                _finish_episodes(idx.unsqueeze(0), reason_str)

            ep_info = extras.get("episode", {})
            if isinstance(ep_info, dict):
                for k, v in ep_info.items():
                    if k not in episode_signals:
                        episode_signals[k] = []
                    if isinstance(v, torch.Tensor):
                        episode_signals[k].append(v.float().mean().item())
                    elif isinstance(v, (int, float)):
                        episode_signals[k].append(float(v))

        if _prev_motion_ts is not None:
            clip_ended = (
                (_prev_motion_ts > 1)
                & (_motion_cmd.time_steps < _prev_motion_ts)
                & ~done_mask
            )
            if clip_ended.any():
                success_indices = clip_ended.nonzero(as_tuple=False).squeeze(-1)
                if success_indices.dim() == 0:
                    success_indices = success_indices.unsqueeze(0)
                _finish_episodes(success_indices, "success")

        if single_episode_per_env and bool(per_env_done.all().item()):
            logger.info(
                f"[Eval] All {env.num_envs} envs completed their first episode "
                f"at step {step + 1}. Stopping evaluation loop."
            )
            break

    post_evaluate_policy(agent)

    if was_training:
        agent.actor.train()
        if agent.obs_normalization:
            agent.obs_normalizer.train()

    metrics: dict[str, float] = {}
    num_steps = step + 1 if max_eval_steps else 0
    if num_steps > 0:
        metrics["mean_reward"] = (total_reward / num_steps).mean().item()
        if completed_ep_rewards:
            metrics["mean_ep_reward"] = sum(completed_ep_rewards) / len(
                completed_ep_rewards
            )
            metrics["mean_ep_length"] = sum(completed_ep_lengths) / len(
                completed_ep_lengths
            )
        metrics["num_episodes"] = float(len(completed_ep_rewards))
        if all_actions:
            stacked = torch.cat(all_actions, dim=0)
            metrics["action_mean"] = stacked.abs().mean().item()
        for k, vals in episode_signals.items():
            if vals:
                metrics[f"ep_{k}"] = sum(vals) / len(vals)

    if single_episode_per_env:
        agent._last_per_env_stats = {
            "num_envs": int(env.num_envs),
            "reward": per_env_reward.detach().cpu().tolist(),
            "length": per_env_length.detach().cpu().tolist(),
            "success": per_env_success.detach().cpu().tolist(),
            "reason": list(per_env_reason),
            "done": per_env_done.detach().cpu().tolist(),
            "min_obj2goal_dist": [float("nan")] * int(env.num_envs),
            "max_obj_height": [float("nan")] * int(env.num_envs),
            "first_grasp_step": [-1] * int(env.num_envs),
            "first_lift_step": [-1] * int(env.num_envs),
            "bad_tracking_step": [-1] * int(env.num_envs),
            "action_norm_mean": [float("nan")] * int(env.num_envs),
            "action_abs_max": [float("nan")] * int(env.num_envs),
            "alive_steps": per_env_length.detach().long().cpu().tolist(),
            "initial_obj2goal_dist": [float("nan")] * int(env.num_envs),
            "object_xy_displacement": [float("nan")] * int(env.num_envs),
            "min_hand_obj_dist": [float("nan")] * int(env.num_envs),
            "first_contact_step": [-1] * int(env.num_envs),
            "first_approach_step": [-1] * int(env.num_envs),
            "reward_until_bad_tracking": per_env_reward.detach().cpu().tolist(),
            "min_hand_obj_dist_at_lift": [float("nan")] * int(env.num_envs),
        }

    return metrics


def run_eval_rollouts(agent: Any, *args: Any, **kwargs: Any) -> Any:
    """Run short deterministic rollouts and return structured metrics."""
    num_steps = int(args[0]) if args else int(kwargs.pop("num_steps", 200))
    was_training = agent.actor.training
    agent.actor.eval()
    if agent.obs_normalization:
        agent.obs_normalizer.eval()

    env = agent.env
    obs = env.reset()
    total_reward = torch.zeros(env.num_envs, device=agent.device)
    ep_reward_sums = torch.zeros(env.num_envs, device=agent.device)
    ep_lengths = torch.zeros(env.num_envs, device=agent.device)
    completed_ep_rewards: list[float] = []
    completed_ep_lengths: list[float] = []
    all_actions: list[Any] = []
    all_obs: list[Any] = []
    episode_signals: dict[str, list[float]] = {}
    total_alive_steps = 0

    _motion_cmd = getattr(
        getattr(agent.unwrapped_env, "command_manager", None),
        "get_state",
        lambda _: None,
    )("motion_command")

    for _step in range(num_steps):
        norm_obs = agent.obs_normalizer(obs, update=False) if agent.obs_normalization else obs
        actions, _pre_tanh_mean, _log_std = agent.actor(norm_obs)
        all_actions.append(actions)
        all_obs.append(obs)
        _prev_ts = _motion_cmd.time_steps.clone() if _motion_cmd is not None else None
        obs, rewards, dones, extras = env.step(actions)
        total_reward += rewards
        ep_reward_sums += rewards
        ep_lengths += 1
        total_alive_steps += int((~dones.bool()).sum().item())

        def _collect_episodes(indices: Any) -> None:
            for idx in indices:
                completed_ep_rewards.append(ep_reward_sums[idx].item())
                completed_ep_lengths.append(ep_lengths[idx].item())
            ep_reward_sums[indices] = 0.0
            ep_lengths[indices] = 0.0

        done_mask = dones.bool()
        if done_mask.any():
            done_indices = done_mask.nonzero(as_tuple=False).squeeze(-1)
            _collect_episodes(done_indices)
            ep_info = extras.get("episode", {})
            if isinstance(ep_info, dict):
                for k, v in ep_info.items():
                    episode_signals.setdefault(k, [])
                    if isinstance(v, torch.Tensor):
                        episode_signals[k].append(v.float().mean().item())
                    elif isinstance(v, (int, float)):
                        episode_signals[k].append(float(v))

        if _prev_ts is not None:
            clip_ended = ((_prev_ts > 1) & (_motion_cmd.time_steps < _prev_ts) & ~done_mask)
            if clip_ended.any():
                success_indices = clip_ended.nonzero(as_tuple=False).squeeze(-1)
                if success_indices.dim() == 0:
                    success_indices = success_indices.unsqueeze(0)
                _collect_episodes(success_indices)

    metrics: dict[str, float] = {}
    metrics["mean_reward"] = (total_reward / max(num_steps, 1)).mean().item()
    if completed_ep_rewards:
        metrics["mean_ep_reward"] = sum(completed_ep_rewards) / len(completed_ep_rewards)
        metrics["mean_ep_length"] = sum(completed_ep_lengths) / len(completed_ep_lengths)
    else:
        metrics["mean_ep_reward"] = ep_reward_sums.mean().item()
        metrics["mean_ep_length"] = ep_lengths.mean().item()
    metrics["num_episodes"] = float(len(completed_ep_rewards))
    if all_actions:
        stacked_actions = torch.cat(all_actions, dim=0)
        metrics["action_mean"] = stacked_actions.abs().mean().item()
        metrics["action_std"] = stacked_actions.std().item()
    if all_obs:
        stacked_obs = torch.cat(all_obs, dim=0)
        metrics["obs_mean"] = stacked_obs.abs().mean().item()
    for k, vals in episode_signals.items():
        if vals:
            metrics[f"ep_{k}"] = sum(vals) / len(vals)

    n_completed = len(completed_ep_rewards)
    metrics["fall_rate"] = n_completed / max(env.num_envs, 1)
    metrics["time_to_fall_mean"] = (
        sum(completed_ep_lengths) / len(completed_ep_lengths)
        if completed_ep_lengths
        else float(num_steps)
    )
    total_possible = num_steps * env.num_envs
    metrics["alive_steps_ratio"] = total_alive_steps / max(total_possible, 1)

    _task_key_map = {
        "grasp_success": "grasp_success_rate",
        "carry_success": "carry_success_rate",
        "place_success": "place_success_rate",
        "final_box_goal_dist": "final_box_goal_dist_mean",
        "box_height_max": "box_height_max_mean",
    }
    for src_key, dst_key in _task_key_map.items():
        if src_key in episode_signals and episode_signals[src_key]:
            metrics[dst_key] = sum(episode_signals[src_key]) / len(episode_signals[src_key])
        elif f"ep_{src_key}" in metrics:
            metrics[dst_key] = metrics[f"ep_{src_key}"]

    if was_training:
        agent.actor.train()
        if agent.obs_normalization:
            agent.obs_normalizer.train()

    return metrics


def create_eval_callbacks(agent: Any) -> None:
    """Create configured evaluation callbacks on *agent*."""
    if not hasattr(agent, "eval_callbacks"):
        agent.eval_callbacks = []
    if agent.config.eval_callbacks is not None:
        for cb_name in agent.config.eval_callbacks:
            agent.eval_callbacks.append(
                instantiate(agent.config.eval_callbacks[cb_name], training_loop=agent)
            )


def pre_evaluate_policy(agent: Any) -> None:
    agent.env.set_is_evaluating()
    for c in agent.eval_callbacks:
        c.on_pre_evaluate_policy()


def post_evaluate_policy(agent: Any) -> None:
    for c in agent.eval_callbacks:
        c.on_post_evaluate_policy()


def pre_eval_env_step(agent: Any, actor_state: dict) -> dict:
    for c in agent.eval_callbacks:
        actor_state = c.on_pre_eval_env_step(actor_state)
    return actor_state


def post_eval_env_step(agent: Any, actor_state: dict) -> dict:
    for c in agent.eval_callbacks:
        actor_state = c.on_post_eval_env_step(actor_state)
    return actor_state


__all__ = [
    "run_evaluate_policy",
    "run_eval_rollouts",
    "create_eval_callbacks",
    "pre_evaluate_policy",
    "post_evaluate_policy",
    "pre_eval_env_step",
    "post_eval_env_step",
]
