"""Default reward manager configurations."""

from holosoma.config_values.loco.g1.reward import g1_29dof_loco, g1_29dof_loco_fast_sac
from holosoma.config_values.loco.t1.reward import t1_29dof_loco, t1_29dof_loco_fast_sac
from holosoma.config_values.wbt.g1.reward import (
    g1_29dof_wbt_fast_sac_reward,
    g1_29dof_wbt_reward,
    g1_29dof_wbt_reward_w_object,
    g1_29dof_wbt_fast_sac_reward_collect,
    g1_29dof_wbt_fast_sac_reward_offline_collect,
)
from holosoma.config_values.wbt.g1.fall_and_getup.reward import (
    g1_29dof_wbt_fall_and_getup_fast_sac_reward,
    g1_29dof_wbt_fall_and_getup_reward,
)
from holosoma.config_values.wbt.t1.reward import t1_29dof_wbt_fast_sac_reward, t1_29dof_wbt_reward

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof_loco": t1_29dof_loco,
    "t1_29dof_loco_fast_sac": t1_29dof_loco_fast_sac,
    "t1_29dof_wbt": t1_29dof_wbt_reward,
    "t1_29dof_wbt_fast_sac": t1_29dof_wbt_fast_sac_reward,
    "g1_29dof_loco": g1_29dof_loco,
    "g1_29dof_loco_fast_sac": g1_29dof_loco_fast_sac,
    "g1_29dof_wbt": g1_29dof_wbt_reward,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_reward_w_object,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac_reward,
    "g1_29dof_wbt_fall_and_getup": g1_29dof_wbt_fall_and_getup_reward,
    "g1_29dof_wbt_fall_and_getup_fast_sac": g1_29dof_wbt_fall_and_getup_fast_sac_reward,
    # "g1_29dof_wbt_fast_sac_collect": g1_29dof_wbt_fast_sac_reward_collect,
    # "g1_29dof_wbt_fast_sac_reward_offline_collect": g1_29dof_wbt_fast_sac_reward_offline_collect,
}
