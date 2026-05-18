"""Optimiser / AMP helpers for canonical offline-RL agents.

``polyak_update`` is owned by :mod:`offline_rl.common.loss_utils` and is
re-exported here for callers that group optimiser-related helpers together.
The removed ``offline_cql`` package no longer provides a compatibility path.
"""

from contextlib import contextmanager

from holosoma.agents.offline_rl.common.loss_utils import polyak_update
from holosoma.utils.safe_torch_import import autocast, torch


@contextmanager
def amp_autocast(*, enabled: bool, amp_dtype: str):
    """Free-function AMP context — mirrors ``OfflineCQLAgent._maybe_amp``.

    Parameters
    ----------
    enabled:
        Forwarded to :func:`torch.amp.autocast` (the legacy ``args.amp``).
    amp_dtype:
        ``"bf16"`` → ``torch.bfloat16``; anything else → ``torch.float16``
        (matches the legacy two-branch check).
    """
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    with autocast(device_type="cuda", dtype=dtype, enabled=enabled):
        yield


__all__ = [
    "polyak_update",
    "amp_autocast",
]
