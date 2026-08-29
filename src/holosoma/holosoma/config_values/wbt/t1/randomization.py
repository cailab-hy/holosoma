"""Whole-body tracking randomization presets for the T1 29-DoF robot."""

from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_values.wbt.g1.randomization import base_reset_terms, base_setup_terms, base_step_terms

t1_29dof_wbt_randomization = RandomizationManagerCfg(
    setup_terms={**base_setup_terms},
    reset_terms={**base_reset_terms},
    step_terms={**base_step_terms},
)

__all__ = ["t1_29dof_wbt_randomization"]
