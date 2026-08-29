"""Default termination manager configurations."""

from holosoma.config_values.loco.g1.termination import g1_29dof_termination
from holosoma.config_values.loco.t1.termination import t1_29dof_termination
from holosoma.config_values.wbt.g1.termination import (
    g1_29dof_wbt_offline_termination,
    g1_29dof_wbt_termination,
    g1_29dof_wbt_termination_collect,
    g1_29dof_wbt_termination_d3_segment,
    g1_29dof_wbt_termination_offline_collect,
)
from holosoma.config_values.wbt.g1.fall_and_getup.termination import (
    g1_29dof_wbt_fall_and_getup_termination,
)
from holosoma.config_values.wbt.t1.termination import t1_29dof_wbt_termination

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_termination,
    "t1_29dof_wbt": t1_29dof_wbt_termination,
    "g1_29dof": g1_29dof_termination,
    "g1_29dof_wbt": g1_29dof_wbt_termination,
    "g1_29dof_wbt_fall_and_getup": g1_29dof_wbt_fall_and_getup_termination,
    "g1_29dof_wbt_termination_offline_collect" : g1_29dof_wbt_termination_offline_collect,
    "g1_29dof_wbt_offline_termination" : g1_29dof_wbt_offline_termination,
    "g1_29dof_wbt_termination_collect" : g1_29dof_wbt_termination_collect,
    "g1_29dof_wbt_termination_d3_segment": g1_29dof_wbt_termination_d3_segment,
}
