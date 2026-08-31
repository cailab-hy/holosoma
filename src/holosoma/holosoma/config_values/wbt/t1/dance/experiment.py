"""Training and episode-data experiments for the cropped T1 dance task."""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import action, algo, curriculum, observation, randomization, reward, termination, terrain
from holosoma.config_values.wbt.t1.dance.command import t1_29dof_wbt_dance_command
from holosoma.config_values.wbt.t1.experiment import _t1_wbt_robot, _t1_wbt_simulator


_FAST_SAC_DATASET = "offline_data/t1_29dof_wbt_dance_fastsac_dataset.h5"
_EPISODE_DATASET = "offline_data/t1_29dof_wbt_dance_fastsac_episode_dataset.h5"


def _dance_simulator():
    base = _t1_wbt_simulator()
    return replace(
        base,
        config=replace(
            base.config,
            sim=replace(base.config.sim, max_episode_length_s=5.0),
        ),
    )


t1_29dof_wbt_dance = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="t1_29dof_wbt_dance_manager",
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
        ),
    ),
    simulator=_dance_simulator(),
    robot=_t1_wbt_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.t1_29dof_wbt_observation,
    action=action.t1_29dof_joint_pos,
    termination=termination.t1_29dof_wbt_termination,
    randomization=randomization.t1_29dof_wbt_randomization,
    command=t1_29dof_wbt_dance_command,
    curriculum=curriculum.t1_29dof_wbt_curriculum,
    reward=reward.t1_29dof_wbt_reward,
    nightly=NightlyConfig(iterations=8000, metrics={}),
)

t1_29dof_wbt_dance_fast_sac = replace(
    t1_29dof_wbt_dance,
    training=replace(
        t1_29dof_wbt_dance.training,
        name="t1_29dof_wbt_dance_fast_sac_manager",
    ),
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
            offline_dataset_path=_FAST_SAC_DATASET,
        ),
    ),
    reward=reward.t1_29dof_wbt_fast_sac_reward,
)

t1_29dof_wbt_dance_fast_sac_episode_data = replace(
    t1_29dof_wbt_dance_fast_sac,
    training=replace(
        t1_29dof_wbt_dance_fast_sac.training,
        name="t1_29dof_wbt_dance_fast_sac_episode_data_manager",
    ),
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
            offline_dataset_path=_EPISODE_DATASET,
            episode_data_active_envs=64,
        ),
    ),
)

__all__ = [
    "t1_29dof_wbt_dance",
    "t1_29dof_wbt_dance_fast_sac",
    "t1_29dof_wbt_dance_fast_sac_episode_data",
]
