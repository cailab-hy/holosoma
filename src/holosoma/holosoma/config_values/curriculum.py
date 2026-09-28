"""Default curriculum manager configurations."""

from holosoma.config_values.loco.g1.curriculum import (
    g1_29dof_curriculum,
    g1_29dof_curriculum_fast_sac,
    g1_29dof_curriculum_fast_sac_data,
)
from holosoma.config_values.loco.t1.curriculum import t1_29dof_curriculum, t1_29dof_curriculum_fast_sac
from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_curriculum
from holosoma.config_values.wbt.g1.lafan_dance1.curriculum import (
    g1_29dof_wbt_lafan_dance1_curriculum,
)
from holosoma.config_values.wbt.t1.curriculum import t1_29dof_wbt_curriculum
from holosoma.config_values.wbt.k1.lafan_dance1.curriculum import k1_23dof_wbt_lafan_dance1_curriculum
from holosoma.config_values.wbt.k1.curriculum import k1_23dof_wbt_curriculum

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_curriculum,
    "g1_29dof": g1_29dof_curriculum,
    "t1_29dof_fast_sac": t1_29dof_curriculum_fast_sac,
    "t1_29dof_wbt": t1_29dof_wbt_curriculum,
    "g1_29dof_fast_sac": g1_29dof_curriculum_fast_sac,
    "g1_29dof_fast_sac_data": g1_29dof_curriculum_fast_sac_data,
    "g1_29dof_wbt_curriculum": g1_29dof_wbt_curriculum,
    "g1_29dof_wbt_lafan_dance1": g1_29dof_wbt_lafan_dance1_curriculum,
    "k1_23dof_wbt": k1_23dof_wbt_curriculum,
    "k1_23dof_wbt_lafan_dance1": k1_23dof_wbt_lafan_dance1_curriculum,
}
