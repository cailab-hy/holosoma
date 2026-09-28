"""Termination configuration isolated for the lafan-dance1 task."""

from copy import deepcopy

from holosoma.config_values.wbt.k1.termination import k1_23dof_wbt_termination

k1_23dof_wbt_lafan_dance1_termination = deepcopy(k1_23dof_wbt_termination)


__all__ = ["k1_23dof_wbt_lafan_dance1_termination"]
