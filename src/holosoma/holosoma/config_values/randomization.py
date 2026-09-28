"""Default randomization manager configurations."""
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_values.loco.g1.randomization import g1_29dof_randomization
from holosoma.config_values.loco.t1.randomization import t1_29dof_randomization
from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization, g1_29dof_wbt_randomization_w_object
from holosoma.config_values.wbt.g1.lafan_dance1.randomization import (
    g1_29dof_wbt_lafan_dance1_randomization,
)
from holosoma.config_values.wbt.t1.randomization import t1_29dof_wbt_randomization
from holosoma.config_values.wbt.k1.lafan_dance1.randomization import k1_23dof_wbt_lafan_dance1_randomization
from holosoma.config_values.wbt.k1.randomization import k1_23dof_wbt_randomization

none = none = RandomizationManagerCfg()

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_randomization,
    "t1_29dof_wbt": t1_29dof_wbt_randomization,
    "g1_29dof": g1_29dof_randomization,
    "g1_29dof_wbt": g1_29dof_wbt_randomization,
    "g1_29dof_wbt_lafan_dance1": g1_29dof_wbt_lafan_dance1_randomization,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_randomization_w_object,
    "k1_23dof_wbt": k1_23dof_wbt_randomization,
    "k1_23dof_wbt_lafan_dance1": k1_23dof_wbt_lafan_dance1_randomization,
}
