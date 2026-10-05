"""Curriculum configuration isolated for the lafan-kick task."""

from copy import deepcopy

from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_curriculum

g1_29dof_wbt_lafan_kick_curriculum = deepcopy(g1_29dof_wbt_curriculum)


__all__ = ["g1_29dof_wbt_lafan_kick_curriculum"]
