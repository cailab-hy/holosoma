"""Termination configuration isolated for the fall-and-get-up task."""

from copy import deepcopy

from holosoma.config_values.wbt.g1.termination import g1_29dof_wbt_termination


g1_29dof_wbt_fall_and_getup_termination = deepcopy(g1_29dof_wbt_termination)
g1_29dof_wbt_fall_and_getup_termination.terms["bad_tracking"].params.update(
    {
        "reset_grace_steps": 20,
        "reset_grace_motion_phase": 0.05,
    }
)


__all__ = ["g1_29dof_wbt_fall_and_getup_termination"]
