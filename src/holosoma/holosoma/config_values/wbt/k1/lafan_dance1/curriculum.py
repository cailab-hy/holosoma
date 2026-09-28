"""Curriculum configuration isolated for the lafan-dance1 task."""

from copy import deepcopy

from holosoma.config_values.wbt.k1.curriculum import k1_23dof_wbt_curriculum

k1_23dof_wbt_lafan_dance1_curriculum = deepcopy(k1_23dof_wbt_curriculum)


__all__ = ["k1_23dof_wbt_lafan_dance1_curriculum"]
