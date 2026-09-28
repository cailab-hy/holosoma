"""Default observation manager configurations."""

from holosoma.config_values.loco.g1.observation import g1_29dof_loco_single_wolinvel
from holosoma.config_values.loco.t1.observation import t1_29dof_loco_single_wolinvel
from holosoma.config_values.wbt.g1.observation import g1_29dof_wbt_observation, g1_29dof_wbt_observation_w_object
from holosoma.config_values.wbt.g1.lafan_dance1.observation import (
    g1_29dof_wbt_lafan_dance1_observation,
)
from holosoma.config_values.wbt.t1.observation import t1_29dof_wbt_observation
from holosoma.config_values.wbt.k1.lafan_dance1.observation import k1_23dof_wbt_lafan_dance1_observation
from holosoma.config_values.wbt.k1.observation import k1_23dof_wbt_observation

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof_loco_single_wolinvel": t1_29dof_loco_single_wolinvel,
    "t1_29dof_wbt": t1_29dof_wbt_observation,
    "g1_29dof_loco_single_wolinvel": g1_29dof_loco_single_wolinvel,
    "g1_29dof_wbt": g1_29dof_wbt_observation,
    "g1_29dof_wbt_lafan_dance1": g1_29dof_wbt_lafan_dance1_observation,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_observation_w_object,
    "k1_23dof_wbt": k1_23dof_wbt_observation,
    "k1_23dof_wbt_lafan_dance1": k1_23dof_wbt_lafan_dance1_observation,
}
