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
from holosoma.config_values.wbt.g1.lafan_dance1.termination import (
    g1_29dof_wbt_lafan_dance1_termination,
)
from holosoma.config_values.wbt.t1.termination import t1_29dof_wbt_termination
from holosoma.config_values.wbt.k1.lafan_dance1.termination import k1_23dof_wbt_lafan_dance1_termination
from holosoma.config_values.wbt.k1.termination import (
    k1_23dof_wbt_termination,
    k1_23dof_wbt_termination_offline_collect,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_termination,
    "t1_29dof_wbt": t1_29dof_wbt_termination,
    "g1_29dof": g1_29dof_termination,
    "g1_29dof_wbt": g1_29dof_wbt_termination,
    "g1_29dof_wbt_lafan_dance1": g1_29dof_wbt_lafan_dance1_termination,
    "g1_29dof_wbt_termination_offline_collect" : g1_29dof_wbt_termination_offline_collect,
    "g1_29dof_wbt_offline_termination" : g1_29dof_wbt_offline_termination,
    "g1_29dof_wbt_termination_collect" : g1_29dof_wbt_termination_collect,
    "g1_29dof_wbt_termination_d3_segment": g1_29dof_wbt_termination_d3_segment,
    "k1_23dof_wbt": k1_23dof_wbt_termination,
    "k1_23dof_wbt_offline_collect": k1_23dof_wbt_termination_offline_collect,
    "k1_23dof_wbt_lafan_dance1": k1_23dof_wbt_lafan_dance1_termination,
}
