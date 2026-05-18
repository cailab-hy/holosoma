"""Loss / target-network utilities.

:func:`polyak_update` is the canonical offline-RL target-network update.
The legacy ``offline_cql`` re-export path has been removed; callers should
import this function from ``offline_rl.common.loss_utils`` or
``offline_rl.common.optim``.
"""

import torch
from torch import nn


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """In-place Polyak (exponential moving average) update of target params.

    ::

        θ_target ← τ · θ_source + (1 − τ) · θ_target

    Uses ``torch._foreach_*`` for fused element-wise operations, matching
    the same pattern as ``FastSACAgent``'s inline target update.

    Parameters
    ----------
    source:
        The online network whose parameters are being tracked.
    target:
        The target network to update in-place.
    tau:
        Interpolation coefficient in ``(0, 1]``.  Typical value: 0.005.
    """
    src_params = [p.data for p in source.parameters()]
    tgt_params = [p.data for p in target.parameters()]
    torch._foreach_mul_(tgt_params, 1.0 - tau)
    torch._foreach_add_(tgt_params, src_params, alpha=tau)

__all__ = [
    "polyak_update",
]
