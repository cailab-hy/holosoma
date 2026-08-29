"""Whole-body tracking command presets for the T1 29-DoF robot."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

T1_WBT_BODY_NAMES = [
    "Trunk",
    "Hip_Roll_Left",
    "Shank_Left",
    "left_foot_link",
    "Hip_Roll_Right",
    "Shank_Right",
    "right_foot_link",
    "Waist",
    "AL2",
    "AL4",
    "left_hand_link",
    "AR2",
    "AR4",
    "right_hand_link",
]

T1_WBT_END_EFFECTOR_NAMES = [
    "left_foot_link",
    "right_foot_link",
    "left_hand_link",
    "right_hand_link",
]

motion_config = MotionConfig(
    motion_file="holosoma/data/motions/t1_29dof/whole_body_tracking/t1_29dof_standing_mj.npz",
    body_names_to_track=T1_WBT_BODY_NAMES,
    body_name_ref=["Trunk"],
    ankle_body_names=["left_foot_link", "right_foot_link"],
    wrist_body_names=["left_hand_link", "right_hand_link"],
    use_adaptive_timesteps_sampler=False,
    start_at_timestep_zero_prob=1.0,
    freeze_at_timestep_zero_prob=0.0,
    enable_default_pose_prepend=False,
    default_pose_prepend_duration_s=0.0,
    enable_default_pose_append=False,
    default_pose_append_duration_s=0.0,
    noise_to_initial_pose=NoiseToInitialPoseConfig(
        overall_noise_scale=1.0,
        dof_pos=0.05,
        root_pos=[0.02, 0.02, 0.01],
        root_rot=[0.05, 0.05, 0.1],
        root_lin_vel=[0.05, 0.05, 0.025],
        root_ang_vel=[0.05, 0.05, 0.05],
    ),
)

t1_29dof_wbt_command = CommandManagerCfg(
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": motion_config},
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand"),
    },
    step_terms={
        "motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand"),
    },
)

__all__ = [
    "T1_WBT_BODY_NAMES",
    "T1_WBT_END_EFFECTOR_NAMES",
    "motion_config",
    "t1_29dof_wbt_command",
]
