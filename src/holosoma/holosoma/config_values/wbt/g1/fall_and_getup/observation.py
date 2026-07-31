"""Observation configuration isolated for the fall-and-get-up task."""

from copy import deepcopy

from holosoma.config_values.wbt.g1.observation import g1_29dof_wbt_observation


g1_29dof_wbt_fall_and_getup_observation = deepcopy(g1_29dof_wbt_observation)


__all__ = ["g1_29dof_wbt_fall_and_getup_observation"]
