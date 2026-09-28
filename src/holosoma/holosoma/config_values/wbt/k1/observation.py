"""Whole-body tracking observation presets for the AI Sapiens K1 Rev.1 (23-DoF) robot.

The WBT observation terms are robot-agnostic (dof/body dimensions are resolved from the
robot config at runtime), so K1 reuses the G1 layout.
"""

from copy import deepcopy

from holosoma.config_values.wbt.g1.observation import g1_29dof_wbt_observation

k1_23dof_wbt_observation = deepcopy(g1_29dof_wbt_observation)

__all__ = ["k1_23dof_wbt_observation"]
