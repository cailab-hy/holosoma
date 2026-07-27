"""Default command manager configurations."""

from holosoma.config_values.loco.g1.command import g1_29dof_command
from holosoma.config_values.loco.t1.command import t1_29dof_command
from holosoma.config_values.wbt.g1.command import (
    g1_29dof_wbt_command,
    g1_29dof_wbt_command_d3_seg_a,
    g1_29dof_wbt_command_d3_seg_b,
    g1_29dof_wbt_command_d3_seg_c,
    g1_29dof_wbt_command_offline_collect,
    g1_29dof_wbt_command_w_object,
)
from holosoma.config_values.wbt.g1.fall_and_getup_command import (
    g1_29dof_wbt_fall_and_getup_command,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_command,
    "g1_29dof": g1_29dof_command,
    "g1_29dof_wbt": g1_29dof_wbt_command,
    "g1_29dof_wbt_offline_collect" : g1_29dof_wbt_command_offline_collect,
    "g1_29dof_wbt_offline_eval" : g1_29dof_wbt_command_offline_collect,
    "g1_29dof_wbt_d3_seg_a": g1_29dof_wbt_command_d3_seg_a,
    "g1_29dof_wbt_d3_seg_b": g1_29dof_wbt_command_d3_seg_b,
    "g1_29dof_wbt_d3_seg_c": g1_29dof_wbt_command_d3_seg_c,
    "g1_29dof_wbt_fall_and_getup": g1_29dof_wbt_fall_and_getup_command,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_command_w_object,
}
