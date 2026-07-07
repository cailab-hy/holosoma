"""Whole Body Tracking command presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.1, 0.1, 0.05],
    root_ang_vel=[0.1, 0.1, 0.1],
    object_pos=[0.05, 0.05, 0.0],
)

motion_config = MotionConfig(
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz",
    body_names_to_track=[
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ],
    body_name_ref=["torso_link"],
    use_adaptive_timesteps_sampler=False,
    noise_to_initial_pose=init_pose_config,
)

motion_config_offline_collect = replace(
    motion_config,
    use_adaptive_timesteps_sampler=False,
    start_at_timestep_zero_prob=0.2,
    freeze_at_timestep_zero_prob=0.0,
    enable_default_pose_prepend=True,
    default_pose_prepend_duration_s=1.0,
    enable_default_pose_append=False,
    default_pose_append_duration_s=0.0,
)

motion_config_d3_base = replace(
    motion_config,
    use_adaptive_timesteps_sampler=False,
    start_at_timestep_zero_prob=0.2,
    freeze_at_timestep_zero_prob=0.0,
    enable_default_pose_prepend=False,
    default_pose_prepend_duration_s=0.0,
    enable_default_pose_append=False,
    default_pose_append_duration_s=0.0,
)

motion_config_d3_seg_a = replace(
    motion_config_d3_base,
    d3_segment_id=0,
    d3_segment_start_phase=0.00,
    d3_segment_end_phase=0.39,
)

motion_config_d3_seg_b = replace(
    motion_config_d3_base,
    d3_segment_id=1,
    d3_segment_start_phase=0.29,
    d3_segment_end_phase=0.69,
)

motion_config_d3_seg_c = replace(
    motion_config_d3_base,
    d3_segment_id=2,
    d3_segment_start_phase=0.60,
    d3_segment_end_phase=1.00,
)

motion_config_w_object = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz",
)

g1_29dof_wbt_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config,
            },
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
)

g1_29dof_wbt_command_w_object = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object,
            },
        )
    },
)

g1_29dof_wbt_command_offline_collect = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_offline_collect,
            },
        )
    },
)

g1_29dof_wbt_command_d3_seg_a = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_d3_seg_a,
            },
        )
    },
)

g1_29dof_wbt_command_d3_seg_b = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_d3_seg_b,
            },
        )
    },
)

g1_29dof_wbt_command_d3_seg_c = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_d3_seg_c,
            },
        )
    },
)

__all__ = [
    "g1_29dof_wbt_command",
    "g1_29dof_wbt_command_offline_collect",
    "g1_29dof_wbt_command_d3_seg_a",
    "g1_29dof_wbt_command_d3_seg_b",
    "g1_29dof_wbt_command_d3_seg_c",
    "g1_29dof_wbt_command_w_object",
]
