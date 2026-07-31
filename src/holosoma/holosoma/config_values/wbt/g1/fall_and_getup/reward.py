"""Reward configurations isolated for the fall-and-get-up task."""

from copy import deepcopy

from holosoma.config_values.wbt.g1.reward import (
    g1_29dof_wbt_fast_sac_reward,
    g1_29dof_wbt_reward,
)


g1_29dof_wbt_fall_and_getup_reward = deepcopy(g1_29dof_wbt_reward)
g1_29dof_wbt_fall_and_getup_fast_sac_reward = deepcopy(g1_29dof_wbt_fast_sac_reward)


__all__ = [
    "g1_29dof_wbt_fall_and_getup_fast_sac_reward",
    "g1_29dof_wbt_fall_and_getup_reward",
]
