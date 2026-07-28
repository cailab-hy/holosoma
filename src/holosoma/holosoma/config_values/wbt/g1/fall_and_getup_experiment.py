"""Standalone experiments for fallAndGetUp3 frames [2460, 2580)."""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)
from holosoma.config_values.wbt.g1.fall_and_getup_command import (
    g1_29dof_wbt_fall_and_getup_command,
)


_FALL_AND_GETUP_NIGHTLY = NightlyConfig(
    iterations=200000,
    metrics={
        "Episode/rew_motion_global_ref_position_error_exp": [0.40, "inf"],
        "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
        "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
        "Episode/rew_motion_relative_body_orientation_error_exp": [0.35, "inf"],
        "Episode/rew_motion_global_body_lin_vel": [0.45, "inf"],
        "Episode/rew_motion_global_body_ang_vel": [0.15, "inf"],
    },
)


def _fall_and_getup_simulator():
    return replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            scene=replace(
                simulator.isaacsim.config.scene,
                env_spacing=2.5,
            ),
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    )


def _fall_and_getup_robot():
    return replace(
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_scale=1.0),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    )


g1_29dof_wbt_fall_and_getup = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_fall_and_getup_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=30000,
            save_interval=4000,
            entropy_coef=0.005,
            init_noise_std=1.0,
            init_at_random_ep_len=False,
            use_symmetry=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.0),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.0),
        ),
    ),
    simulator=_fall_and_getup_simulator(),
    robot=_fall_and_getup_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=g1_29dof_wbt_fall_and_getup_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_reward,
    nightly=_FALL_AND_GETUP_NIGHTLY,
)

g1_29dof_wbt_fall_and_getup_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_fall_and_getup_fast_sac_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=200000,
            v_max=20.0,
            v_min=-20.0,
            gamma=0.99,
            num_steps=1,
            num_updates=4,
            num_atoms=501,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
            offline_dataset_path=(
                "offline_data/g1_29dof_wbt_fall_and_getup_fastsac_dataset.h5"
            ),
        ),
    ),
    simulator=_fall_and_getup_simulator(),
    robot=_fall_and_getup_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=g1_29dof_wbt_fall_and_getup_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
    nightly=_FALL_AND_GETUP_NIGHTLY,
)

g1_29dof_wbt_fall_and_getup_fast_sac_episode_data = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_fall_and_getup_fast_sac_episode_data_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fast_sac_episode_data,
        config=replace(
            algo.fast_sac_episode_data.config,
            num_learning_iterations=40000,
            v_max=20.0,
            v_min=-20.0,
            gamma=0.99,
            num_steps=1,
            num_updates=4,
            num_atoms=501,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
            offline_dataset_path=(
                "offline_data/"
                "g1_29dof_wbt_fall_and_getup_fastsac_4m_episode_env64_dataset.h5"
            ),
            episode_data_active_envs=64,
        ),
    ),
    simulator=_fall_and_getup_simulator(),
    robot=_fall_and_getup_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=g1_29dof_wbt_fall_and_getup_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
    nightly=_FALL_AND_GETUP_NIGHTLY,
)


__all__ = [
    "g1_29dof_wbt_fall_and_getup",
    "g1_29dof_wbt_fall_and_getup_fast_sac",
    "g1_29dof_wbt_fall_and_getup_fast_sac_episode_data",
]
