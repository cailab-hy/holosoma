"""Whole-body tracking randomization presets for the AI Sapiens K1 Rev.1 (23-DoF) robot.

Uses the shared WBT domain-randomization terms (pushes, PD gain scaling, friction,
base CoM and joint-bias offsets). Link names come from ``robot.k1_23dof``.
"""

from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_values.wbt.g1.randomization import base_reset_terms, base_setup_terms, base_step_terms

k1_23dof_wbt_randomization = RandomizationManagerCfg(
    setup_terms={**base_setup_terms},
    reset_terms={**base_reset_terms},
    step_terms={**base_step_terms},
)

__all__ = ["k1_23dof_wbt_randomization"]
