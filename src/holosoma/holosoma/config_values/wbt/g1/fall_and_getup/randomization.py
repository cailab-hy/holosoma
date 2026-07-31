"""Randomization configuration isolated for the fall-and-get-up task."""

from copy import deepcopy

from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization


g1_29dof_wbt_fall_and_getup_randomization = deepcopy(g1_29dof_wbt_randomization)


__all__ = ["g1_29dof_wbt_fall_and_getup_randomization"]
