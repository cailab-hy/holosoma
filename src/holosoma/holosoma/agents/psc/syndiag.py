"""Synergy-OOD diagnostic helpers for BF-CQL (logging only, no loss changes).

Everything in this module is pure, gradient-free math used by the
``_syndiag_*`` methods of :class:`BFCQLAgent`. Nothing here may touch training
state (losses, optimizers, normalizers, default RNG). See ``SYNDIAG_NOTES.md``
at the repo root for the logging contract and the raw-dump schema consumed by
``tools/eval_counterfactual_gap.py``.

Notation (all in the normalized [-1, 1] action space the critic consumes):
    a_D   dataset actions            [B, A]
    a_pi  deterministic actor action [B, A]
    d_g   per-group normalized-RMSE drift between a_pi and a_D   [B, G]
    v(M)  min(Q1,Q2)(s, a_cf(M)) - min(Q1,Q2)(s, a_D)            [B]
    Delta(M) = v(M) - sum_{g in M} v({g})                        [B]
where a_cf(M) replaces the dims of the groups in coalition M with a_pi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

DRIFT_EPS = 1e-6
RATIO_EPS = 1e-8

SIZE_PREFIX = {1: "sing", 2: "pair", 3: "tri", 4: "quad"}

# Named physical blocks added on top of singletons+pairs when the grouping
# uses the coarse left/right leg-arm-waist partition. Pairs among these are
# deduplicated against the all-pairs list.
NAMED_COALITION_MEMBERS: tuple[tuple[str, ...], ...] = (
    ("left_leg", "right_leg"),
    ("left_leg", "right_leg", "waist"),
    ("left_arm", "right_arm"),
    ("left_leg", "right_leg", "left_arm", "right_arm"),
)
NAMED_COALITION_REQUIRED = ("left_leg", "right_leg", "waist", "left_arm", "right_arm")


@dataclass(frozen=True)
class Coalition:
    name: str
    group_ids: tuple[int, ...]


def abbreviate_group_names(group_names: Sequence[str]) -> list[str]:
    """Deterministic, collision-free short names: first letter of each underscore token.

    e.g. left_leg -> LL, waist -> W, left_knee_ankle -> LKA. On collision the
    later name gets a numeric suffix (W, W2, W3, ...).
    """
    abbrevs: list[str] = []
    used: set[str] = set()
    for name in group_names:
        base = "".join(token[0] for token in name.split("_") if token).upper() or "G"
        abbr = base
        suffix = 2
        while abbr in used:
            abbr = f"{base}{suffix}"
            suffix += 1
        used.add(abbr)
        abbrevs.append(abbr)
    return abbrevs


def coalition_display_name(group_ids: tuple[int, ...], abbrevs: Sequence[str]) -> str:
    prefix = SIZE_PREFIX.get(len(group_ids), f"c{len(group_ids)}")
    return prefix + "_" + "_".join(abbrevs[g] for g in group_ids)


def build_coalitions(
    group_names: Sequence[str],
    max_coalitions: int,
    warn: Callable[[str], None] | None = None,
) -> list[Coalition]:
    """All singletons, all pairs, then named physical blocks; truncated at the cap.

    Singletons always come first in group order (Delta needs every singleton),
    then pairs in (i, j) i<j order, then named triples/quads. Truncation keeps
    at least the G singletons even if max_coalitions < G.
    """
    num_groups = len(group_names)
    abbrevs = abbreviate_group_names(group_names)
    coalitions: list[Coalition] = []
    seen: set[tuple[int, ...]] = set()

    def _add(ids: tuple[int, ...]) -> None:
        ids = tuple(sorted(ids))
        if ids in seen:
            return
        seen.add(ids)
        coalitions.append(Coalition(coalition_display_name(ids, abbrevs), ids))

    for g in range(num_groups):
        _add((g,))
    for i in range(num_groups):
        for j in range(i + 1, num_groups):
            _add((i, j))

    name_to_id = {name: idx for idx, name in enumerate(group_names)}
    if all(required in name_to_id for required in NAMED_COALITION_REQUIRED):
        for members in NAMED_COALITION_MEMBERS:
            _add(tuple(name_to_id[m] for m in members))
    elif warn is not None:
        warn(
            "syndiag: group names do not contain the coarse "
            f"{NAMED_COALITION_REQUIRED} partition; skipping named triple/quad coalitions."
        )

    if len(coalitions) > max_coalitions:
        keep = max(max_coalitions, num_groups)
        if warn is not None:
            warn(
                f"syndiag: coalition list has {len(coalitions)} entries; truncating to {keep} "
                f"(max_coalitions={max_coalitions}, singletons+pairs kept first)."
            )
        coalitions = coalitions[:keep]
    return coalitions


def group_dim_mask(
    group_indices: Sequence[Sequence[int]],
    n_act: int,
    device: torch.device | str,
) -> torch.Tensor:
    """[G, A] bool mask mapping each group to its action dims."""
    mask = torch.zeros(len(group_indices), n_act, dtype=torch.bool, device=device)
    for row, dims in enumerate(group_indices):
        mask[row, list(dims)] = True
    return mask


def coalition_group_mask(
    coalitions: Sequence[Coalition],
    num_groups: int,
    device: torch.device | str,
) -> torch.Tensor:
    """[C, G] bool membership matrix."""
    mask = torch.zeros(len(coalitions), num_groups, dtype=torch.bool, device=device)
    for row, coalition in enumerate(coalitions):
        mask[row, list(coalition.group_ids)] = True
    return mask


def singleton_columns(coalitions: Sequence[Coalition], num_groups: int) -> torch.Tensor:
    """[G] long tensor: column in the coalition list holding the singleton of each group."""
    cols = torch.full((num_groups,), -1, dtype=torch.long)
    for col, coalition in enumerate(coalitions):
        if len(coalition.group_ids) == 1:
            cols[coalition.group_ids[0]] = col
    if (cols < 0).any():
        missing = [g for g in range(num_groups) if cols[g] < 0]
        raise ValueError(f"syndiag coalition list is missing singletons for groups {missing}")
    return cols


def compute_group_drift(
    a_pi: torch.Tensor,
    a_data: torch.Tensor,
    sigma: torch.Tensor,
    group_dim_masks: torch.Tensor,
) -> torch.Tensor:
    """d_g(s, a_D) = sqrt(mean_{j in g} ((a_pi_j - a_D_j) / (sigma_j + eps))^2) -> [B, G]."""
    z2 = ((a_pi - a_data) / (sigma + DRIFT_EPS)).square()  # [B, A]
    mask = group_dim_masks.to(z2.dtype)  # [G, A]
    per_group_mean = (z2 @ mask.t()) / mask.sum(dim=1).clamp_min(1.0)
    return per_group_mean.sqrt()


def coalition_q_values(
    critic_fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    critic_obs: torch.Tensor,
    a_pi: torch.Tensor,
    a_data: torch.Tensor,
    coalition_dim_masks: torch.Tensor,
) -> torch.Tensor:
    """min(Q1,Q2)(s, a_cf(M)) for every coalition in ONE twin-critic forward -> [B, C].

    a_cf(M) = where(mask(M), a_pi, a_D), stacked along the batch dim as
    [C * B, ...]; no per-coalition loop over the batch.
    """
    num_coalitions = coalition_dim_masks.shape[0]
    batch_size, n_act = a_data.shape
    a_cf = torch.where(coalition_dim_masks[:, None, :], a_pi[None, :, :], a_data[None, :, :])
    flat_obs = (
        critic_obs[None, :, :]
        .expand(num_coalitions, batch_size, critic_obs.shape[-1])
        .reshape(num_coalitions * batch_size, -1)
    )
    q1, q2 = critic_fn(flat_obs, a_cf.reshape(num_coalitions * batch_size, n_act))
    return torch.minimum(q1, q2).view(num_coalitions, batch_size).transpose(0, 1).contiguous()


def synergy_residuals(
    v: torch.Tensor,
    coalition_group_masks: torch.Tensor,
    singleton_cols: torch.Tensor,
) -> torch.Tensor:
    """Delta(M) = v(M) - sum_{g in M} v({g}) -> [B, C]; exactly zero for singletons."""
    v_singletons = v.index_select(1, singleton_cols.to(v.device))  # [B, G]
    return v - v_singletons @ coalition_group_masks.to(v.dtype).t()


def quartile_delta_stats(
    block_drift: torch.Tensor,
    delta: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    """Split pooled (block_drift, Delta) points into drift quartiles; mean Delta per bin.

    Returns {q1..q4, q4_over_q1, q4_minus_q1} or None if there are too few points.
    q4_over_q1 keeps the sign of Q4 and divides by |Q1| clamped at eps.
    """
    block_drift = block_drift.reshape(-1).float()
    delta = delta.reshape(-1).float()
    if block_drift.numel() < 8:
        return None
    boundaries = torch.quantile(
        block_drift,
        torch.tensor([0.25, 0.5, 0.75], device=block_drift.device, dtype=block_drift.dtype),
    )
    bins = torch.bucketize(block_drift, boundaries)
    stats: dict[str, torch.Tensor] = {}
    bin_means: list[torch.Tensor] = []
    for b in range(4):
        selected = bins == b
        if selected.any():
            bin_means.append(delta[selected].mean())
        else:
            bin_means.append(torch.zeros((), device=delta.device))
        stats[f"q{b + 1}"] = bin_means[b]
    stats["q4_minus_q1"] = bin_means[3] - bin_means[0]
    stats["q4_over_q1"] = bin_means[3] / bin_means[0].abs().clamp_min(RATIO_EPS)
    return stats


def recall_top_pair(
    delta_pairs: torch.Tensor,
    pair_group_ids: torch.Tensor,
    drift: torch.Tensor,
    top_k: int,
    delta_min: float,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """recall@K for the per-sample top-Delta pair against the top-K drift groups.

    delta_pairs: [B, P], pair_group_ids: [P, 2], drift: [B, G].
    Returns (recall over active samples or None if no sample is active, active_frac).
    """
    top_delta, top_idx = delta_pairs.max(dim=1)
    active = top_delta > delta_min
    active_frac = active.float().mean()
    if not bool(active.any()):
        return None, active_frac

    chosen_pairs = pair_group_ids.to(drift.device)[top_idx]  # [B, 2]
    k = min(top_k, drift.shape[1])
    topk_groups = drift.topk(k, dim=1).indices  # [B, k]
    hit_first = (topk_groups == chosen_pairs[:, 0:1]).any(dim=1)
    hit_second = (topk_groups == chosen_pairs[:, 1:2]).any(dim=1)
    both_in_topk = hit_first & hit_second
    return both_in_topk[active].float().mean(), active_frac


def superadditivity_quad(
    delta: torch.Tensor,
    coalitions: Sequence[Coalition],
) -> torch.Tensor | None:
    """mean(Delta(4-limb quad) - sum of Delta over the 6 pairs inside it), or None.

    Positive values indicate higher-order synergy beyond pairwise interactions.
    """
    quad_col = None
    for col, coalition in enumerate(coalitions):
        if len(coalition.group_ids) == 4:
            quad_col = col
            quad_members = set(coalition.group_ids)
            break
    if quad_col is None:
        return None

    pair_cols = [
        col
        for col, coalition in enumerate(coalitions)
        if len(coalition.group_ids) == 2 and set(coalition.group_ids) <= quad_members
    ]
    if len(pair_cols) != 6:
        return None
    pair_sum = delta[:, pair_cols].sum(dim=1)
    return (delta[:, quad_col] - pair_sum).mean()


__all__ = [
    "Coalition",
    "abbreviate_group_names",
    "build_coalitions",
    "coalition_display_name",
    "coalition_group_mask",
    "coalition_q_values",
    "compute_group_drift",
    "group_dim_mask",
    "quartile_delta_stats",
    "recall_top_pair",
    "singleton_columns",
    "superadditivity_quad",
    "synergy_residuals",
]
