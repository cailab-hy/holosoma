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

g1_29dof = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_manager"),
    algo=replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=25000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum,
    reward=reward.g1_29dof_loco,
    nightly=NightlyConfig(
        iterations=5000,
        metrics={"Episode/rew_tracking_ang_vel": [0.7, "inf"], "Episode/rew_tracking_lin_vel": [0.55, "inf"]},
    ),
)

g1_29dof_fast_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fast_sac_manager"),
    algo=replace(algo.fast_sac, config=replace(algo.fast_sac.config, num_learning_iterations=50000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac,
    reward=reward.g1_29dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)

g1_29dof_fast_sac_data = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fast_sac_data_collect_manager"),
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=50000,
            use_symmetry=True,
            offline_dataset_path="offline_data/g1_29dof_loco_fastsac_dataset.h5",
        ),
    ),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac_data,
    reward=reward.g1_29dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)

g1_29dof_fast_sac_episode_data = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fast_sac_episode_data_collect_manager"),
    algo=replace(
        algo.fast_sac_episode_data,
        config=replace(
            algo.fast_sac_episode_data.config,
            num_learning_iterations=50000,
            use_symmetry=True,
            offline_dataset_path="offline_data/g1_29dof_loco_fastsac_episode_dataset.h5",
            episode_data_active_envs=64,
            episode_data_mc_gamma=None,
        ),
    ),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac_data,
    reward=reward.g1_29dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)


g1_29dof_cql = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(
        project="hv-g1-manager",
        name="g1_29dof_cql_weight_5_manager",
        num_envs=512,
        eval_num_episodes=1,
    ),
    algo=replace(
        algo.cql,
        config=replace(
            algo.cql.config,
            num_learning_iterations=50000,
            use_symmetry=True,
            cql_weight=5.0,
            cql_num_action_samples=10,
            offline_dataset_path="offline_data/g1_29dof_loco_fastsac_dataset_Replay64.h5",
        ),
    ),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac_data,
    reward=reward.g1_29dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)

g1_29dof_os_cql = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(
        project="hv-g1-manager",
        name="g1_29dof_os_cql_weight_5_manager",
        num_envs=512,
        eval_num_episodes=1,
    ),
    algo=replace(
        algo.os_cql,
        config=replace(
            algo.os_cql.config,
            num_learning_iterations=50000,
            use_symmetry=True,
            cql_weight=5.0,
            cql_num_action_samples=10,
            offline_dataset_path="offline_data/g1_29dof_loco_fastsac_dataset_Replay64.h5",
        ),
    ),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac_data,
    reward=reward.g1_29dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)

g1_29dof_cal_ql = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(
        project="hv-g1-manager",
        name="g1_29dof_cal_ql_o2o_manager",
        num_envs=512,
        eval_num_episodes=1,
    ),
    algo=replace(
        algo.cal_ql,
        config=replace(
            algo.cal_ql.config,
            num_learning_iterations=50000,
            offline_pretrain_steps=10000,
            online_total_steps=40000,
            online_warmup_steps=1000,
            online_collect_steps=1,
            updates_per_collect=1,
            mixing_ratio_schedule="fixed",
            offline_mixing_ratio=0.5,
            use_symmetry=True,
            cql_weight=5.0,
            cql_num_action_samples=10,
            offline_dataset_path="offline_data/g1_29dof_loco_fastsac_dataset_Replay64.h5",
        ),
    ),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac_data,
    reward=reward.g1_29dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)

g1_29dof_os_cal_ql = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(
        project="hv-g1-manager",
        name="g1_29dof_os_cal_ql_lagrange5_manager",
        num_envs=512,
        eval_num_episodes=1,
    ),
    algo=replace(
        algo.cal_ql,
        config=replace(
            algo.cal_ql.config,
            num_learning_iterations=50000,
            offline_pretrain_steps=10000,
            online_total_steps=40000,
            online_warmup_steps=1000,
            online_collect_steps=1,
            updates_per_collect=1,
            mixing_ratio_schedule="fixed",
            offline_mixing_ratio=0.5,
            use_symmetry=True,
            use_lagrange = True,
            cql_weight=5.0,
            cql_num_action_samples=10,
            offline_dataset_path="offline_data/g1_29dof_loco_fastsac_dataset_Replay64.h5",
        ),
    ),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac_data,
    reward=reward.g1_29dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)



__all__ = [
    "g1_29dof",
    "g1_29dof_fast_sac",
    "g1_29dof_fast_sac_data",
    "g1_29dof_fast_sac_episode_data",
    "g1_29dof_cql",
    "g1_29dof_os_cql",
    "g1_29dof_cal_ql",
    "g1_29dof_os_cal_ql",
]
