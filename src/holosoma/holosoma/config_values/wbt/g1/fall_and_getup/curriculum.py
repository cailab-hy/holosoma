"""Curriculum configuration isolated for the fall-and-get-up task."""

from copy import deepcopy

from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_curriculum


g1_29dof_wbt_fall_and_getup_curriculum = deepcopy(g1_29dof_wbt_curriculum)


__all__ = ["g1_29dof_wbt_fall_and_getup_curriculum"]
