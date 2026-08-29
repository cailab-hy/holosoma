"""Whole-body tracking experiments for the T1 29-DoF robot."""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)


def _t1_wbt_simulator():
    return replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            scene=replace(simulator.isaacsim.config.scene, env_spacing=2.5),
            sim=replace(simulator.isaacsim.config.sim, max_episode_length_s=4.0),
        ),
    )


def _t1_wbt_robot():
    return replace(
        robot.t1_29dof_waist_wrist,
        control=replace(robot.t1_29dof_waist_wrist.control, action_scale=1.0),
        asset=replace(robot.t1_29dof_waist_wrist.asset, enable_self_collisions=True),
        init_state=replace(robot.t1_29dof_waist_wrist.init_state, pos=[0.0, 0.0, 0.68]),
    )


t1_29dof_wbt = ExperimentConfig(
    training=TrainingConfig(project="WholeBodyTracking", name="t1_29dof_wbt_manager", num_envs=4096),
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
        ),
    ),
    simulator=_t1_wbt_simulator(),
    robot=_t1_wbt_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.t1_29dof_wbt_observation,
    action=action.t1_29dof_joint_pos,
    termination=termination.t1_29dof_wbt_termination,
    randomization=randomization.t1_29dof_wbt_randomization,
    command=command.t1_29dof_wbt_command,
    curriculum=curriculum.t1_29dof_wbt_curriculum,
    reward=reward.t1_29dof_wbt_reward,
    nightly=NightlyConfig(iterations=8000, metrics={}),
)

t1_29dof_wbt_fast_sac = replace(
    t1_29dof_wbt,
    training=replace(t1_29dof_wbt.training, name="t1_29dof_wbt_fast_sac_manager"),
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=50000,
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
            offline_dataset_path="offline_data/t1_29dof_wbt_fastsac_dataset.h5",
        ),
    ),
    reward=reward.t1_29dof_wbt_fast_sac_reward,
)

t1_29dof_wbt_fast_sac_episode_data = replace(
    t1_29dof_wbt_fast_sac,
    training=replace(
        t1_29dof_wbt_fast_sac.training,
        name="t1_29dof_wbt_fast_sac_episode_data_manager",
    ),
    algo=replace(
        algo.fast_sac_episode_data,
        config=replace(
            algo.fast_sac_episode_data.config,
            num_learning_iterations=50000,
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
            offline_dataset_path="offline_data/t1_29dof_wbt_fastsac_episode_dataset.h5",
            episode_data_active_envs=64,
        ),
    ),
)

__all__ = ["t1_29dof_wbt", "t1_29dof_wbt_fast_sac", "t1_29dof_wbt_fast_sac_episode_data"]
