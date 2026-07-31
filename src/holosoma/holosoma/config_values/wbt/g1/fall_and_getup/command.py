"""Whole-body tracking command for fallAndGetUp3 frames [2460, 2580)."""

from holosoma.config_types.command import (
    CommandManagerCfg,
    CommandTermCfg,
    MotionConfig,
    NoiseToInitialPoseConfig,
)


g1_29dof_wbt_fall_and_getup_motion = MotionConfig(
    motion_file=(
        "holosoma/data/motions/g1_29dof/whole_body_tracking/"
        "g1_29dof_wbt_fall_and_getup_mj.npz"
    ),
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
    enable_default_pose_prepend=False,
    default_pose_prepend_duration_s=0.0,
    enable_default_pose_append=True,
    default_pose_append_duration_s=2.0,
    noise_to_initial_pose=NoiseToInitialPoseConfig(
        overall_noise_scale=1.0,
        dof_pos=0.1,
        root_pos=[0.05, 0.05, 0.01],
        root_rot=[0.1, 0.1, 0.2],
        root_lin_vel=[0.1, 0.1, 0.05],
        root_ang_vel=[0.1, 0.1, 0.1],
        object_pos=[0.05, 0.05, 0.0],
    ),
)

g1_29dof_wbt_fall_and_getup_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": g1_29dof_wbt_fall_and_getup_motion},
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


__all__ = [
    "g1_29dof_wbt_fall_and_getup_command",
    "g1_29dof_wbt_fall_and_getup_motion",
]
