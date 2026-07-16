from __future__ import annotations

import math
from typing import Any


def _eval_env_candidates(algo: Any) -> list[Any]:
    env = getattr(algo, "env", None)
    return [
        getattr(algo, "unwrapped_env", None),
        env,
        getattr(env, "_env", None),
    ]


def set_defer_eval_resets(algo: Any, enabled: bool) -> bool:
    """Enable one-episode-per-environment evaluation when supported by the environment."""
    for env in _eval_env_candidates(algo):
        setter = getattr(env, "set_defer_resets", None)
        if callable(setter):
            setter(enabled)
            return True
    return False


def eval_terminal_motion_phases(algo: Any) -> list[float] | None:
    """Read per-environment motion phases captured at each terminal transition."""
    for env in _eval_env_candidates(algo):
        getter = getattr(env, "get_eval_terminal_motion_phases", None)
        if not callable(getter):
            continue
        phases = getter()
        if hasattr(phases, "detach"):
            phases = phases.detach()
        if hasattr(phases, "cpu"):
            phases = phases.cpu()
        if hasattr(phases, "tolist"):
            phases = phases.tolist()
        if isinstance(phases, (list, tuple)):
            return [float(value) for value in phases]
    return None


def attach_terminal_motion_phases(algo: Any, eval_results: list[dict[str, Any]]) -> None:
    """Attach the matching environment's terminal phase to vectorized evaluation results."""
    phases = eval_terminal_motion_phases(algo)
    if phases is None:
        return
    for env_idx, result in enumerate(eval_results):
        if env_idx >= len(phases):
            break
        phase = phases[env_idx]
        if math.isfinite(phase):
            result["terminal_motion_phase"] = min(max(phase, 0.0), 1.0)


def bad_tracking_phase_bin_counts(
    eval_results: list[dict[str, Any]],
    num_bins: int,
) -> tuple[list[int], int]:
    """Count bad-tracking terminal events by normalized motion-phase bin."""
    num_bins = max(1, int(num_bins))
    counts = [0] * num_bins
    unresolved = 0
    for result in eval_results:
        if result.get("stop_reason") != "bad_tracking":
            continue
        value = result.get("terminal_motion_phase")
        try:
            phase = float(value)
        except (TypeError, ValueError):
            unresolved += 1
            continue
        if not math.isfinite(phase):
            unresolved += 1
            continue
        phase = min(max(phase, 0.0), 1.0)
        bin_idx = min(int(phase * num_bins), num_bins - 1)
        counts[bin_idx] += 1
    return counts, unresolved


def phase_bin_label(bin_idx: int, num_bins: int) -> str:
    left = float(bin_idx) / float(num_bins)
    right = float(bin_idx + 1) / float(num_bins)
    closing = "]" if bin_idx == num_bins - 1 else ")"
    return f"bin{bin_idx:02d}[{left:.2f},{right:.2f}{closing}"


def phase_bin_metric_key(bin_idx: int, num_bins: int) -> str:
    left = float(bin_idx) / float(num_bins)
    right = float(bin_idx + 1) / float(num_bins)
    return f"Bad_tracking/phase_{left:.2f}_{right:.2f}"


def phase_bin_summary(
    counts: list[int],
    bad_tracking_total: int,
) -> tuple[dict[str, int], dict[str, float]]:
    num_bins = len(counts)
    count_summary = {
        phase_bin_label(bin_idx, num_bins): count
        for bin_idx, count in enumerate(counts)
        if count > 0
    }
    percentage_summary = {
        label: 100.0 * float(count) / float(max(bad_tracking_total, 1))
        for label, count in count_summary.items()
    }
    return count_summary, percentage_summary


def bad_tracking_phase_metrics(
    eval_results: list[dict[str, Any]],
    num_bins: int,
) -> tuple[dict[str, float], int]:
    """Return phase-bin percentages whose denominator is all bad-tracking eval episodes."""
    num_bins = max(1, int(num_bins))
    counts, unresolved = bad_tracking_phase_bin_counts(eval_results, num_bins)
    if not any("terminal_motion_phase" in result for result in eval_results):
        return {}, unresolved
    bad_tracking_total = sum(result.get("stop_reason") == "bad_tracking" for result in eval_results)
    denominator = max(1, bad_tracking_total)
    metrics = {
        phase_bin_metric_key(bin_idx, num_bins): 100.0 * float(count) / float(denominator)
        for bin_idx, count in enumerate(counts)
    }
    return metrics, unresolved
