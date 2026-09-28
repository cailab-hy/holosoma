"""Whole-body tracking command for LAFAN dance1 retargeted to K1 in cyclo_lab."""

from holosoma.config_types.command import (
    CommandManagerCfg,
    CommandTermCfg,
    MotionConfig,
    NoiseToInitialPoseConfig,
)

k1_23dof_wbt_lafan_dance1_motion = MotionConfig(
    motion_file="holosoma/data/motions/k1_23dof/whole_body_tracking/dance1_mj.npz",
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
        "left_wrist_roll_rubber_hand",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_roll_rubber_hand",
    ],
    body_name_ref=["torso_link"],
    ankle_body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
    wrist_body_names=["left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand"],
    use_adaptive_timesteps_sampler=False,
    # Ease in from the default pose, but do NOT ease back out (the prepend is a lerp in
    # body-position space and is only faithful near the default standing pose).
    enable_default_pose_prepend=True,
    default_pose_prepend_duration_s=2.0,
    enable_default_pose_append=False,
    default_pose_append_duration_s=0.0,
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

k1_23dof_wbt_lafan_dance1_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": k1_23dof_wbt_lafan_dance1_motion},
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
    "k1_23dof_wbt_lafan_dance1_command",
    "k1_23dof_wbt_lafan_dance1_motion",
]
