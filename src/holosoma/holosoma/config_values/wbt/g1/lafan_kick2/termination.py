"""Termination configuration isolated for the lafan-kick2 task."""

from copy import deepcopy

from holosoma.config_values.wbt.g1.termination import g1_29dof_wbt_termination

g1_29dof_wbt_lafan_kick2_termination = deepcopy(g1_29dof_wbt_termination)


__all__ = ["g1_29dof_wbt_lafan_kick2_termination"]
