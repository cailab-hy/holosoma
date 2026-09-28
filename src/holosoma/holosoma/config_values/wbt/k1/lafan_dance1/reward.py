"""Reward configurations isolated for the lafan-dance1 task."""

from copy import deepcopy

from holosoma.config_values.wbt.k1.reward import (
    k1_23dof_wbt_fast_sac_reward,
    k1_23dof_wbt_reward,
)

k1_23dof_wbt_lafan_dance1_reward = deepcopy(k1_23dof_wbt_reward)
k1_23dof_wbt_lafan_dance1_fast_sac_reward = deepcopy(k1_23dof_wbt_fast_sac_reward)


__all__ = [
    "k1_23dof_wbt_lafan_dance1_fast_sac_reward",
    "k1_23dof_wbt_lafan_dance1_reward",
]
