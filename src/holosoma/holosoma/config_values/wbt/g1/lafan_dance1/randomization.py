"""Randomization configuration isolated for the lafan-dance1 task.

Kept as a distinct name so the task can diverge later, but the terms are currently
identical to motion_tracking's: pushes are enabled on motion_tracking's own
push_interval_s=[1.0, 3.0] schedule rather than disabled.
"""

from copy import deepcopy

from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization


g1_29dof_wbt_lafan_dance1_randomization = deepcopy(g1_29dof_wbt_randomization)


__all__ = ["g1_29dof_wbt_lafan_dance1_randomization"]
