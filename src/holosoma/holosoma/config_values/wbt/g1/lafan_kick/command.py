"""Whole-body tracking command for LAFAN fight1_subject3 frames [4999, 5116] (30 fps, original speed).

One left-leg front kick (foot 0.53 m, torso tilt <= 28 deg) between calm standing: 195 frames @ 50 Hz.
"""

from holosoma.config_types.command import (
    CommandManagerCfg,
    CommandTermCfg,
    MotionConfig,
    NoiseToInitialPoseConfig,
)

g1_29dof_wbt_lafan_kick_motion = MotionConfig(
    motion_file=("holosoma/data/motions/g1_29dof/whole_body_tracking/fight1_subject3_frames_4999_5116_mj.npz"),
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
    # Chosen by scripts/scan_lafan_one_event.py (original speed, standing ends, exactly one high-risk event,
    # torso tilt <= 50 deg, travel <= 2 m). The window is bounded by walking-in before frame 4999 and a second,
    # much larger kick (foot 0.84 m, tilt 65 deg) starting at frame ~5120.
    # Ease in from and back out to the default pose (2 s each), like the largebox task.
    # Both clip ends are upright standing poses (first frame 1.59 rad, last frame 1.44 rad
    # from the default pose in joint space, the same range as largebox's 1.44 / 1.12), so
    # the body-position lerp of the append is faithful here, unlike dance1 whose last
    # frame is mid-motion.
    enable_default_pose_prepend=True,
    default_pose_prepend_duration_s=2.0,
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

g1_29dof_wbt_lafan_kick_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": g1_29dof_wbt_lafan_kick_motion},
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
    "g1_29dof_wbt_lafan_kick_command",
    "g1_29dof_wbt_lafan_kick_motion",
]
