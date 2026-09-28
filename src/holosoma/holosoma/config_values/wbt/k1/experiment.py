from dataclasses import asdict, replace

from holosoma.config_types.algo import (
    DWCQLAlgoConfig,
    DWCQLConfig,
    WBCAlgoConfig,
    WBCConfig,
)
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

k1_23dof_wbt = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_manager",
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
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
        ),
    ),
    simulator=replace(
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
    ),
    # NOTE: unlike G1 (action_scale=1.0), K1 keeps its per-joint action scale
    # (0.25 * effort_limit / kp) so exported policies match cyclo_lab's deployment stack.
    robot=replace(
        robot.k1_23dof,
        asset=replace(robot.k1_23dof.asset, enable_self_collisions=True),
        init_state=replace(robot.k1_23dof.init_state, pos=[0.0, 0.0, 0.764]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.k1_23dof_wbt_observation,
    action=action.k1_23dof_joint_pos,
    termination=termination.k1_23dof_wbt_termination,
    randomization=randomization.k1_23dof_wbt_randomization,
    command=command.k1_23dof_wbt_command,
    curriculum=curriculum.k1_23dof_wbt_curriculum,
    reward=reward.k1_23dof_wbt_reward,
    nightly=NightlyConfig(
        iterations=8000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.16, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.20, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [0.45, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.30, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.30, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.02, "inf"],
        },
    ),
)

k1_23dof_wbt_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_fast_sac_reward_terminate_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=50000,
            v_max=20.0,
            v_min=-20.0,
            gamma=0.99,  # For motion tracking, high gamma + high num_steps is better
            num_steps=1,
            num_updates=4,
            num_atoms=501,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
            offline_dataset_path="offline_data/k1_23dof_wbt_fastsac_env4096_128_rew_dataset.h5",
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            scene=replace(
                simulator.isaacsim.config.scene,
                env_spacing=2.5,
            ),
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=12.0,
            ),
        ),
    ),
    robot=replace(
        robot.k1_23dof,
        asset=replace(robot.k1_23dof.asset, enable_self_collisions=True),
        init_state=replace(robot.k1_23dof.init_state, pos=[0.0, 0.0, 0.764]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.k1_23dof_wbt_observation,
    action=action.k1_23dof_joint_pos,
    termination=termination.k1_23dof_wbt_termination_offline_collect,
    randomization=randomization.k1_23dof_wbt_randomization,
    command=command.k1_23dof_wbt_command_offline_collect,
    curriculum=curriculum.k1_23dof_wbt_curriculum,
    reward=reward.k1_23dof_wbt_fast_sac_reward_offline_collect,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.40, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.35, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.45, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.15, "inf"],
        },
    ),
)

k1_23dof_wbt_fast_sac_episode_data = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_fast_sac_episode_data_1m_collect_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fast_sac_episode_data,
        config=replace(
            algo.fast_sac_episode_data.config,
            num_learning_iterations=20000,
            v_max=20.0,
            v_min=-20.0,
            gamma=0.99,  # For motion tracking, high gamma + high num_steps is better
            num_steps=1,
            num_updates=4,
            num_atoms=501,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
            offline_dataset_path="offline_data/k1_23dof_wbt_fastsac_episode1m_env256_dataset.h5",
            episode_data_active_envs=256,
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            scene=replace(
                simulator.isaacsim.config.scene,
                env_spacing=2.5,
            ),
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=12.0,
            ),
        ),
    ),
    robot=replace(
        robot.k1_23dof,
        asset=replace(robot.k1_23dof.asset, enable_self_collisions=True),
        init_state=replace(robot.k1_23dof.init_state, pos=[0.0, 0.0, 0.764]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.k1_23dof_wbt_observation,
    action=action.k1_23dof_joint_pos,
    termination=termination.k1_23dof_wbt_termination,
    randomization=randomization.k1_23dof_wbt_randomization,
    command=command.k1_23dof_wbt_command,
    curriculum=curriculum.k1_23dof_wbt_curriculum,
    reward=reward.k1_23dof_wbt_fast_sac_reward,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.40, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.35, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.45, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.15, "inf"],
        },
    ),
)

k1_23dof_wbt_cql = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_cql_manager",
        num_envs=4096,
        eval_num_episodes=1,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.cql,
        config=replace(
            algo.cql.config,
            num_learning_iterations=100000,
            gamma=0.99,  # For motion tracking, high gamma + high num_steps is better
            num_updates=4,
            policy_frequency=1,
            target_entropy_ratio=0.5,
            tau=0.05,
            cql_weight=5.0,
            cql_num_action_samples=10,
            use_symmetry=False,
            use_lagrange=False,
            batch_size=1024,
            cql_target_action_gap=0.0,
            offline_dataset_path="offline_data/k1_23dof_wbt_fastsac_episode1m_env256_dataset.h5",
            use_gpu_cache=True,
            reward_scale=5.0,
            bellman_loss_type="mse",
            huber_beta=5.0,
            cql_max_target_backup=False,
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=12.0,
            ),
        ),
    ),
    robot=replace(
        robot.k1_23dof,
        asset=replace(robot.k1_23dof.asset, enable_self_collisions=True),
        init_state=replace(robot.k1_23dof.init_state, pos=[0.0, 0.0, 0.764]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.k1_23dof_wbt_observation,
    action=action.k1_23dof_joint_pos,
    # Strict (original) termination for eval: deployment-grade success criterion,
    # deliberately tighter than the loose collection thresholds.
    termination=termination.k1_23dof_wbt_termination,
    randomization=randomization.k1_23dof_wbt_randomization,
    # Eval must replay the dataset's motion timeline (1s default-pose prepend):
    # raw frame 0 carries a retargeting velocity spike and never appears as an
    # episode-start state in the collected data.
    command=command.k1_23dof_wbt_command,
    curriculum=curriculum.k1_23dof_wbt_curriculum,
    reward=reward.k1_23dof_wbt_fast_sac_reward,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.40, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.35, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.45, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.15, "inf"],
        },
    ),
)

k1_23dof_wbt_aw_cql = replace(
    k1_23dof_wbt_cql,
    training=replace(k1_23dof_wbt_cql.training, name="k1_23dof_wbt_aw_cql_manager"),
    algo=replace(
        algo.aw_cql,
        config=replace(
            algo.aw_cql.config,
            **{
                **asdict(k1_23dof_wbt_cql.algo.config),
                "aw_weights_path": "offline_data/k1_23dof_wbt_fastsac_episode1m_env256_dataset.h5.aw_weights.npz",
            },
        ),
    ),
)

k1_23dof_wbt_b_arm = replace(
    k1_23dof_wbt_cql,
    training=replace(k1_23dof_wbt_cql.training, name="k1_23dof_wbt_b_arm_manager"),
    algo=replace(
        algo.b_arm,
        config=replace(
            algo.b_arm.config,
            **{
                **asdict(k1_23dof_wbt_cql.algo.config),
                "aw_weights_path": "offline_data/k1_23dof_wbt_fastsac_episode1m_env256_dataset.h5.aw_weights.npz",
            },
        ),
    ),
)

k1_23dof_wbt_c_arm = replace(
    k1_23dof_wbt_cql,
    training=replace(k1_23dof_wbt_cql.training, name="k1_23dof_wbt_c_arm_manager"),
    algo=replace(
        algo.c_arm,
        config=replace(
            algo.c_arm.config,
            **{
                **asdict(k1_23dof_wbt_cql.algo.config),
                "aw_weights_path": "offline_data/k1_23dof_wbt_fastsac_episode1m_env256_dataset.h5.aw_weights.npz",
            },
        ),
    ),
)


k1_23dof_wbt_dw_cql = replace(
    k1_23dof_wbt_cql,
    training=replace(k1_23dof_wbt_cql.training, name="k1_23dof_wbt_dw_cql_manager"),
    algo=DWCQLAlgoConfig(
        _target_="holosoma.agents.dw_cql.dw_cql_agent.DWCQLAgent",
        _recursive_=False,
        config=DWCQLConfig(
            **{
                **asdict(algo.aw_cql.config),
                **asdict(k1_23dof_wbt_cql.algo.config),
                "aw_weights_path": "offline_data/k1_23dof_wbt_fastsac_episode1m_env256_dataset.h5.aw_weights.npz",
            }
        ),
    ),
)

k1_23dof_wbt_iql = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_iql_manager",
        num_envs=4096,
        eval_num_episodes=1,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.iql,
        config=replace(
            algo.iql.config,
            num_learning_iterations=100000,
            discount=0.99,  # For motion tracking, high discount is typically beneficial
            reward_scale=5.0,
            num_updates=4,
            tau=0.05,
            expectile=0.7,
            beta=3.0,
            max_weight=100.0,
            use_symmetry=False,
            offline_dataset_path="offline_data/k1_23dof_wbt_fastsac_episode1m_env256_dataset.h5",
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=12.0,
            ),
        ),
    ),
    robot=replace(
        robot.k1_23dof,
        asset=replace(robot.k1_23dof.asset, enable_self_collisions=True),
        init_state=replace(robot.k1_23dof.init_state, pos=[0.0, 0.0, 0.764]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.k1_23dof_wbt_observation,
    action=action.k1_23dof_joint_pos,
    termination=termination.k1_23dof_wbt_termination,
    randomization=randomization.k1_23dof_wbt_randomization,
    command=command.k1_23dof_wbt_command,
    curriculum=curriculum.k1_23dof_wbt_curriculum,
    reward=reward.k1_23dof_wbt_fast_sac_reward,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.40, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.35, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.45, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.15, "inf"],
        },
    ),
)

k1_23dof_wbt_bc = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_bc_manager",
        num_envs=512,
        eval_num_episodes=1,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.bc,
        config=replace(
            algo.bc.config,
            num_learning_iterations=100000,
            num_updates=4,
            use_symmetry=False,
            offline_dataset_path="offline_data/k1_23dof_wbt_fastsac_episode1m_env256_dataset.h5",
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=12.0,
            ),
        ),
    ),
    robot=replace(
        robot.k1_23dof,
        asset=replace(robot.k1_23dof.asset, enable_self_collisions=True),
        init_state=replace(robot.k1_23dof.init_state, pos=[0.0, 0.0, 0.764]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.k1_23dof_wbt_observation,
    action=action.k1_23dof_joint_pos,
    termination=termination.k1_23dof_wbt_termination,
    randomization=randomization.k1_23dof_wbt_randomization,
    command=command.k1_23dof_wbt_command,
    curriculum=curriculum.k1_23dof_wbt_curriculum,
    reward=reward.k1_23dof_wbt_fast_sac_reward,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.40, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.35, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.45, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.15, "inf"],
        },
    ),
)

k1_23dof_wbt_w_bc = replace(
    k1_23dof_wbt_bc,
    training=replace(k1_23dof_wbt_bc.training, name="k1_23dof_wbt_w_bc_manager"),
    algo=WBCAlgoConfig(
        _target_="holosoma.agents.w_bc.w_bc_agent.WBCAgent",
        _recursive_=False,
        config=replace(
            WBCConfig(**asdict(k1_23dof_wbt_bc.algo.config)),
            aw_weights_path="offline_data/k1_23dof_wbt_fastsac_episode1m_env256_dataset.h5.aw_weights.npz",
        ),
    ),
)

k1_23dof_wbt_td3_bc = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_td3_bc_manager",
        num_envs=4096,
        eval_num_episodes=1,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.td3_bc,
        config=replace(
            algo.td3_bc.config,
            num_learning_iterations=100000,
            critic_learning_rate=3e-4,
            actor_learning_rate=3e-4,
            batch_size=1024,
            num_updates=4,
            discount=0.99,
            reward_scale=5.0,
            bootstrap_truncations=True,
            tau=0.05,
            policy_delay=2,
            use_symmetry=False,
            td3bc_alpha=2.5,
            offline_dataset_path="offline_data/k1_23dof_wbt_fastsac_episode1m_env256_dataset.h5",
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=12.0,
            ),
        ),
    ),
    robot=replace(
        robot.k1_23dof,
        asset=replace(robot.k1_23dof.asset, enable_self_collisions=True),
        init_state=replace(robot.k1_23dof.init_state, pos=[0.0, 0.0, 0.764]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.k1_23dof_wbt_observation,
    action=action.k1_23dof_joint_pos,
    termination=termination.k1_23dof_wbt_termination,
    randomization=randomization.k1_23dof_wbt_randomization,
    command=command.k1_23dof_wbt_command,
    curriculum=curriculum.k1_23dof_wbt_curriculum,
    reward=reward.k1_23dof_wbt_fast_sac_reward,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.40, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.35, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.45, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.15, "inf"],
        },
    ),
)

__all__ = [
    "k1_23dof_wbt",
    "k1_23dof_wbt_aw_cql",
    "k1_23dof_wbt_b_arm",
    "k1_23dof_wbt_bc",
    "k1_23dof_wbt_c_arm",
    "k1_23dof_wbt_cql",
    "k1_23dof_wbt_dw_cql",
    "k1_23dof_wbt_fast_sac",
    "k1_23dof_wbt_fast_sac_episode_data",
    "k1_23dof_wbt_iql",
    "k1_23dof_wbt_td3_bc",
    "k1_23dof_wbt_w_bc",
]

"""
Example 1: PPO (online):
python src/holosoma/holosoma/train_agent.py \
    exp:k1-23dof-wbt

Example 2: FastSAC episode-data collection, then offline CQL:
python src/holosoma/holosoma/train_agent.py \
    exp:k1-23dof-wbt-fast-sac-episode-data
python src/holosoma/holosoma/offline_train_agent.py \
    exp:k1-23dof-wbt-cql
"""
