from __future__ import annotations

"""Stage-1 CODAC network module.

This module intentionally reuses the stable distributional actor-critic
implementations from offline_sac.

Rationale:
- Keep the same action semantics (`env_scaled_action_training_v1`).
- Keep the same distributional critic implementation.
- Add conservative regularization only at the agent loss level.
"""

from holosoma.agents.offline_sac.offline_sac import Actor, CNNActor, CNNCritic, Critic

__all__ = [
    "Actor",
    "CNNActor",
    "Critic",
    "CNNCritic",
]
