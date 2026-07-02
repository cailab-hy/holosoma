"""Utilities for interpreting vectorized evaluation stop reasons."""

from __future__ import annotations

from holosoma.utils.safe_torch_import import torch

BAD_TRACKING_DETAIL_KEYS: tuple[tuple[str, str], ...] = (
    ("bad_tracking_ref_pos", "ref_pos"),
    ("bad_tracking_ref_ori", "ref_ori"),
    ("bad_tracking_body_pos", "body_pos"),
    ("bad_tracking_body_pos_ankle", "body_pos_ankle"),
    ("bad_tracking_body_pos_wrist", "body_pos_wrist"),
    ("bad_tracking_object_pos", "object_pos"),
    ("bad_tracking_object_ori", "object_ori"),
)


def bad_tracking_detail_names(reason_flags: dict[str, torch.Tensor], env_idx: int) -> list[str]:
    """Return active bad-tracking subconditions for one evaluated env."""

    bad_tracking_flags = reason_flags.get("bad_tracking")
    if bad_tracking_flags is None or not bool(bad_tracking_flags[env_idx].item()):
        return []

    details: list[str] = []
    for key, name in BAD_TRACKING_DETAIL_KEYS:
        flags = reason_flags.get(key)
        if flags is not None and bool(flags[env_idx].item()):
            details.append(name)
    return details
