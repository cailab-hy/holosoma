"""T1 dance command using dance2_subject1 source frames [0, 120)."""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg
from holosoma.config_values.wbt.t1.command import motion_config


t1_29dof_wbt_dance_motion = replace(
    motion_config,
    motion_file="holosoma/data/motions/t1_29dof/whole_body_tracking/dance2_short_mj.npz",
    use_adaptive_timesteps_sampler=False,
    start_at_timestep_zero_prob=1.0,
    freeze_at_timestep_zero_prob=0.0,
    enable_default_pose_prepend=False,
    default_pose_prepend_duration_s=0.0,
    enable_default_pose_append=False,
    default_pose_append_duration_s=0.0,
)

t1_29dof_wbt_dance_command = CommandManagerCfg(
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": t1_29dof_wbt_dance_motion},
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand"),
    },
    step_terms={
        "motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand"),
    },
)

__all__ = ["t1_29dof_wbt_dance_command", "t1_29dof_wbt_dance_motion"]
