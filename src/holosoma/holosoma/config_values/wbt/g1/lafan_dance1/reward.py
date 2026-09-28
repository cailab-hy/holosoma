"""Reward configurations isolated for the lafan-dance1 task."""

from copy import deepcopy

from holosoma.config_values.wbt.g1.reward import (
    g1_29dof_wbt_fast_sac_reward,
    g1_29dof_wbt_reward,
)


g1_29dof_wbt_lafan_dance1_reward = deepcopy(g1_29dof_wbt_reward)
g1_29dof_wbt_lafan_dance1_fast_sac_reward = deepcopy(g1_29dof_wbt_fast_sac_reward)


__all__ = [
    "g1_29dof_wbt_lafan_dance1_fast_sac_reward",
    "g1_29dof_wbt_lafan_dance1_reward",
]
