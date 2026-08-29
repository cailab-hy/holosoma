"""Whole-body tracking observation presets for the T1 29-DoF robot."""

from dataclasses import replace

from holosoma.config_values.wbt.g1.observation import g1_29dof_wbt_observation

t1_29dof_wbt_observation = replace(g1_29dof_wbt_observation)

__all__ = ["t1_29dof_wbt_observation"]
