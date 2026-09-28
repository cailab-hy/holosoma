"""Observation configuration isolated for the lafan-dance1 task."""

from copy import deepcopy

from holosoma.config_values.wbt.k1.observation import k1_23dof_wbt_observation

k1_23dof_wbt_lafan_dance1_observation = deepcopy(k1_23dof_wbt_observation)


__all__ = ["k1_23dof_wbt_lafan_dance1_observation"]
