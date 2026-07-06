"""Whole Body Tracking reward presets for the G1 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

g1_29dof_wbt_reward = RewardManagerCfg(
    terms={
        # Motion tracking rewards - global reference frame
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=0.5,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        # Motion tracking rewards - body velocities
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14},
            weight=1.0,
        ),
        # Regularization rewards
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.1,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-100.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
                    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
                ),
            },
            weight=-0.5,
        ),
    }
)

g1_29dof_wbt_fast_sac_reward = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-1.0,
        ),
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=2.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

g1_29dof_wbt_fast_sac_reward_collect = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_fast_sac_reward.terms,
        "motion_global_ref_position_error_hinge": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_hinge",
            params={"threshold": 0.5},
            weight=-2.0,
        ),
        "motion_relative_body_position_error_hinge": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_hinge",
            params={
                "threshold": 0.25,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                "aggregation": "max",
            },
            weight=-5.0,
        ),
    }
)

g1_29dof_wbt_fast_sac_reward_offline_collect = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_fast_sac_reward.terms,
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.5,
        ),
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=2.5,
        ),
        "motion_global_ref_position_error_soft_margin": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_hinge",
            params={"threshold": 0.4, "power": 2.0},
            weight=-8.0,
        ),
        "motion_global_ref_position_error_hinge": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_hinge",
            params={"threshold": 0.5},
            weight=-2.0,
        ),
        "motion_global_ref_position_error_hard_hinge": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_hinge",
            params={"threshold": 1.0},
            weight=-6.0,
        ),
        "motion_global_ref_orientation_error_soft_margin": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_hinge",
            params={"threshold": 0.64, "power": 2.0},
            weight=-3.0,
        ),
        "motion_global_ref_orientation_error_hinge": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_hinge",
            params={"threshold": 0.8},
            weight=-1.0,
        ),
        "motion_global_ref_orientation_error_hard_hinge": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_hinge",
            params={"threshold": 1.6},
            weight=-3.0,
        ),
        "motion_relative_body_position_error_ankle_soft_margin": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_hinge",
            params={
                "threshold": 0.2,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
                "aggregation": "max",
                "power": 2.0,
            },
            weight=-20.0,
        ),
        "motion_relative_body_position_error_wrist_soft_margin": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_hinge",
            params={
                "threshold": 0.14,
                "body_names": [
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                "aggregation": "max",
                "power": 2.0,
            },
            weight=-40.0,
        ),
        "motion_relative_body_position_error_hinge": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_hinge",
            params={
                "threshold": 0.25,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                "aggregation": "max",
            },
            weight=-5.0,
        ),
        "motion_relative_body_position_error_hard_hinge": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_hinge",
            params={
                "threshold": 0.5,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                "aggregation": "max",
            },
            weight=-10.0,
        ),
        "motion_relative_body_orientation_error_soft_margin": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_hinge",
            params={
                "threshold": 0.64,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                "aggregation": "mean",
                "power": 2.0,
            },
            weight=-3.0,
        ),
        "motion_relative_body_orientation_error_hinge": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_hinge",
            params={
                "threshold": 0.8,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                "aggregation": "mean",
            },
            weight=-1.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        # Motion tracking rewards - global reference frame
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

__all__ = [
    "g1_29dof_wbt_fast_sac_reward",
    "g1_29dof_wbt_fast_sac_reward_collect",
    "g1_29dof_wbt_fast_sac_reward_offline_collect",
    "g1_29dof_wbt_reward",
    "g1_29dof_wbt_reward_w_object",
]
