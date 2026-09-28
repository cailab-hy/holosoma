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
from holosoma.config_values.wbt.g1.lafan_dance1.command import (
    g1_29dof_wbt_lafan_dance1_command,
)
from holosoma.config_values.wbt.t1.command import t1_29dof_wbt_command
from holosoma.config_values.wbt.k1.command import k1_23dof_wbt_command, k1_23dof_wbt_command_offline_collect
from holosoma.config_values.wbt.k1.lafan_dance1.command import k1_23dof_wbt_lafan_dance1_command

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_command,
    "t1_29dof_wbt": t1_29dof_wbt_command,
    "g1_29dof": g1_29dof_command,
    "g1_29dof_wbt": g1_29dof_wbt_command,
    "g1_29dof_wbt_offline_collect" : g1_29dof_wbt_command_offline_collect,
    "g1_29dof_wbt_offline_eval" : g1_29dof_wbt_command_offline_collect,
    "g1_29dof_wbt_d3_seg_a": g1_29dof_wbt_command_d3_seg_a,
    "g1_29dof_wbt_d3_seg_b": g1_29dof_wbt_command_d3_seg_b,
    "g1_29dof_wbt_d3_seg_c": g1_29dof_wbt_command_d3_seg_c,
    "g1_29dof_wbt_lafan_dance1": g1_29dof_wbt_lafan_dance1_command,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_command_w_object,
    "k1_23dof_wbt": k1_23dof_wbt_command,
    "k1_23dof_wbt_offline_collect": k1_23dof_wbt_command_offline_collect,
    "k1_23dof_wbt_lafan_dance1": k1_23dof_wbt_lafan_dance1_command,
}
