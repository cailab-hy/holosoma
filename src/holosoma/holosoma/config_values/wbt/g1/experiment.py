from dataclasses import asdict, replace

from holosoma.config_types.algo import (
    AWCQLAlgoConfig,
    AWCQLConfig,
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

g1_29dof_wbt = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_manager",
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
    robot=replace(
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_scale=1.0),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_reward,
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

g1_29dof_wbt_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_fast_sac_reward_terminate_manager",
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
            offline_dataset_path="offline_data/g1_29dof_wbt_fastsac_env4096_128_rew_dataset.h5",
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
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_scale=1.0),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination_offline_collect,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command_offline_collect,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward_offline_collect,
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

g1_29dof_wbt_fast_sac_data = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_fast_sac_data_collect_manager",
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
            offline_dataset_path="offline_data/g1_29dof_wbt_5m_step_32_env_dataset.h5",
            save_env_num = 32,
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
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_scale=1.0),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination_offline_collect,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command_offline_collect,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward_offline_collect,
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

g1_29dof_wbt_fast_sac_d3_seg_a_data = replace(
    g1_29dof_wbt_fast_sac_data,
    training=replace(
        g1_29dof_wbt_fast_sac_data.training,
        name="g1_29dof_wbt_fast_sac_d3_seg_a_data_collect_manager",
    ),
    algo=replace(
        g1_29dof_wbt_fast_sac_data.algo,
        config=replace(
            g1_29dof_wbt_fast_sac_data.algo.config,
            num_learning_iterations=20000,
            offline_dataset_path="offline_data/g1_29dof_wbt_d3_seg_a_dataset.h5",
            save_env_num = 64,
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
                max_episode_length_s=5.0,
            ),
        ),
    ),
    termination=termination.g1_29dof_wbt_termination_d3_segment,
    command=command.g1_29dof_wbt_command_d3_seg_a,
)

g1_29dof_wbt_fast_sac_d3_seg_b_data = replace(
    g1_29dof_wbt_fast_sac_data,
    training=replace(
        g1_29dof_wbt_fast_sac_data.training,
        name="g1_29dof_wbt_fast_sac_d3_seg_b_data_collect_manager",
    ),
    algo=replace(
        g1_29dof_wbt_fast_sac_data.algo,
        config=replace(
            g1_29dof_wbt_fast_sac_data.algo.config,
            num_learning_iterations=20000,
            offline_dataset_path="offline_data/g1_29dof_wbt_d3_seg_b_dataset.h5",
            save_env_num = 64,
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
                max_episode_length_s=5.0,
            ),
        ),
    ),    
    termination=termination.g1_29dof_wbt_termination_d3_segment,
    command=command.g1_29dof_wbt_command_d3_seg_b,
)

g1_29dof_wbt_fast_sac_d3_seg_c_data = replace(
    g1_29dof_wbt_fast_sac_data,
    training=replace(
        g1_29dof_wbt_fast_sac_data.training,
        name="g1_29dof_wbt_fast_sac_d3_seg_c_data_collect_manager",
    ),
    algo=replace(
        g1_29dof_wbt_fast_sac_data.algo,
        config=replace(
            g1_29dof_wbt_fast_sac_data.algo.config,
            num_learning_iterations=20000,
            offline_dataset_path="offline_data/g1_29dof_wbt_d3_seg_c_dataset.h5",
            save_env_num = 64,
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
                max_episode_length_s=5.0,
            ),
        ),
    ),
    termination=termination.g1_29dof_wbt_termination_d3_segment,
    command=command.g1_29dof_wbt_command_d3_seg_c,
)

g1_29dof_wbt_fast_sac_episode_data = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_fast_sac_episode_data_2m_collect_manager",
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
            offline_dataset_path="offline_data/g1_29dof_wbt_fastsac_episode3m_env64_dataset.h5",
            episode_data_active_envs=64,
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
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_scale=1.0),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
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


g1_29dof_wbt_fixed_fast_sac_data = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_fixed_fast_sac_data_collect_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fixed_fast_sac,
        config=replace(
            algo.fixed_fast_sac.config,
            num_learning_iterations=1000,
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
            offline_dataset_path="offline_data/g1_29dof_wbt_1m_step_512_env_dataset.h5",
            episode_data_active_envs=4096,
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
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_scale=1.0),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
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


##for offline rL
g1_29dof_wbt_cql = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_cql_manager_pu1_4096_seed2",
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
            cql_weight = 5.0,
            cql_num_action_samples=10,
            use_symmetry=False,
            use_lagrange=False,
            batch_size=1024,
            cql_target_action_gap=0.0,
            offline_dataset_path="offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5",
            use_gpu_cache=True,
            reward_scale = 5.0,
            bellman_loss_type="mse",
            huber_beta=5.0,
            cql_max_target_backup = False,
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
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_scale=1.0),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    # Strict (original) termination for eval: deployment-grade success criterion,
    # deliberately tighter than the loose collection thresholds.
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    # Eval must replay the dataset's motion timeline (1s default-pose prepend):
    # raw frame 0 carries a retargeting velocity spike and never appears as an
    # episode-start state in the collected data.
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
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

g1_29dof_wbt_vc_cql = replace(
    g1_29dof_wbt_cql,
    training=replace(g1_29dof_wbt_cql.training, name="g1_29dof_wbt_vc_cql_manager"),
    algo=replace(
        algo.vc_cql,
        config=replace(
            algo.vc_cql.config,
            num_learning_iterations=100000,
            gamma=0.99,
            num_updates=4,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            cql_weight=5.0,
            cql_num_action_samples=10,
            use_symmetry=False,
            use_lagrange=False,
            batch_size=1024,
            cql_target_action_gap=0.0,
            offline_dataset_path="offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5",
            use_gpu_cache=True,
            reward_scale=5.0,
            bellman_loss_type="mse",
            huber_beta=5.0,
            cql_max_target_backup=False,
            compile=False,
        ),
    ),
)

g1_29dof_wbt_lr_cql = replace(
    g1_29dof_wbt_vc_cql,
    training=replace(g1_29dof_wbt_vc_cql.training, name="g1_29dof_wbt_lr_cql_manager"),
    algo=replace(
        g1_29dof_wbt_vc_cql.algo,
        config=replace(g1_29dof_wbt_vc_cql.algo.config, vc_contour_weight=0.0),
    ),
)


# AW-CQL v0: exact hyperparameter clone of g1_29dof_wbt_cql (paired one-variable
# comparison); the only change is the agent class, which multiplies the CQL
# conservative bracket by precomputed <h5>.aw_weights.npz sidecar weights.
# Run scripts/aw_precompute_weights.py on the dataset first (Stage 0 gate).
g1_29dof_wbt_aw_cql = replace(
    g1_29dof_wbt_cql,
    training=replace(g1_29dof_wbt_cql.training, name="g1_29dof_wbt_aw_cql_manager_pu1_4096_seed2"),
    algo=AWCQLAlgoConfig(
        _target_="holosoma.agents.aw_cql.aw_cql_agent.AWCQLAgent",
        _recursive_=False,
        config=AWCQLConfig(**asdict(g1_29dof_wbt_cql.algo.config)),
    ),
)


# OS-AW-CQL: paired AW-CQL ablation using the same sidecar weights and all
# hyperparameters, with w applied only to the -Q(s, a_data) anchor term.
g1_29dof_wbt_os_aw_cql = replace(
    g1_29dof_wbt_aw_cql,
    training=replace(g1_29dof_wbt_aw_cql.training, name="g1_29dof_wbt_os_aw_cql_manager_pu1_4096_seed2"),
    algo=AWCQLAlgoConfig(
        _target_="holosoma.agents.os_aw_cql.os_aw_cql_agent.OSAWCQLAgent",
        _recursive_=False,
        config=AWCQLConfig(**asdict(g1_29dof_wbt_aw_cql.algo.config)),
    ),
)


# LSE-AW-CQL: paired AW-CQL ablation using the same sidecar weights and all
# hyperparameters, with w applied only to the logsumexp OOD term.
g1_29dof_wbt_lse_aw_cql = replace(
    g1_29dof_wbt_aw_cql,
    training=replace(g1_29dof_wbt_aw_cql.training, name="g1_29dof_wbt_lse_aw_cql_manager_pu1_4096_seed2"),
    algo=AWCQLAlgoConfig(
        _target_="holosoma.agents.lse_aw_cql.lse_aw_cql_agent.LSEAWCQLAgent",
        _recursive_=False,
        config=AWCQLConfig(**asdict(g1_29dof_wbt_aw_cql.algo.config)),
    ),
)


# DW-CQL: same sidecar, data, and hyperparameters as AW-CQL. The separate
# agent propagates the same fixed w to TD and actor losses as a placement-only
# ablation; target construction and alpha/Lagrange paths remain unchanged.
g1_29dof_wbt_dw_cql = replace(
    g1_29dof_wbt_aw_cql,
    training=replace(g1_29dof_wbt_aw_cql.training, name="g1_29dof_wbt_dw_cql_manager_pu1_4096_seed2"),
    algo=DWCQLAlgoConfig(
        _target_="holosoma.agents.dw_cql.dw_cql_agent.DWCQLAgent",
        _recursive_=False,
        config=DWCQLConfig(**asdict(g1_29dof_wbt_aw_cql.algo.config)),
    ),
)


# MCQ (Lyu et al., NeurIPS 2022): same task/sim/data stack as g1_29dof_wbt_cql,
# with the CQL push-down replaced by MCB in-support anchoring (see MCQConfig).
g1_29dof_wbt_mcq = replace(
    g1_29dof_wbt_cql,
    training=replace(
        g1_29dof_wbt_cql.training,
        name="g1_29dof_wbt_mcq_manager",
    ),
    algo=replace(
        algo.mcq,
        config=replace(
            algo.mcq.config,
            num_learning_iterations=400000,
            gamma=0.99,  # For motion tracking, high gamma + high num_steps is better
            num_updates=4,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
            use_lagrange=False,
            batch_size=1024,
            cql_weight = 5.0,
            offline_dataset_path="offline_data/g1_29dof_wbt_1m_step_512_env_dataset.h5",
            use_gpu_cache=True,
            reward_scale=5.0,
            bellman_loss_type="mse",
            huber_beta=5.0,
            cql_max_target_backup=False,
        ),
    ),
)


g1_29dof_wbt_bf_cql = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_bf_cql_manager",
        num_envs=512,
        eval_num_episodes=1,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.bf_cql,
        config=replace(
            algo.bf_cql.config,
            bf_cql_action_grouping="coarse_5",
            ood_actor_num=1,
            num_learning_iterations=200000,
            gamma=0.99,
            num_updates=4,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            cql_weight=5.0,
            cql_num_action_samples=2,
            use_symmetry=False,
            use_lagrange=False,
            batch_size=1024,
            cql_target_action_gap=0.0,
            offline_dataset_path="offline_data/g1_29dof_wbt_1m_step_512_env_dataset.h5",
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
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_scale=1.0),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
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

##for offline rL
g1_29dof_wbt_iql = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_iql_manager",
        num_envs=4096,
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
            offline_dataset_path="offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5",
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
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_scale=1.0),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
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





##for offline rL
g1_29dof_wbt_bc = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_bc_manager",
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
            offline_dataset_path="offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5",
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
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_scale=1.0),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
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

g1_29dof_wbt_w_bc = replace(
    g1_29dof_wbt_bc,
    training=replace(g1_29dof_wbt_bc.training, name="g1_29dof_wbt_w_bc_manager"),
    algo=WBCAlgoConfig(
        _target_="holosoma.agents.w_bc.w_bc_agent.WBCAgent",
        _recursive_=False,
        config=WBCConfig(**asdict(g1_29dof_wbt_bc.algo.config)),
    ),
)

g1_29dof_wbt_td3_bc = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_td3_bc_manager_seed1",
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
            td3bc_alpha =2.5,
            offline_dataset_path="offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5",
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
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_scale=1.0),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
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



g1_29dof_wbt_w_object = replace(
    g1_29dof_wbt,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(
            robot.g1_29dof_w_object.asset,
            enable_self_collisions=True,
        ),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_29dof_wbt_fast_sac_w_object = replace(
    g1_29dof_wbt_fast_sac,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_29dof_wbt_fast_sac_w_object_data = replace(
    g1_29dof_wbt_fast_sac_data,
    algo=replace(
        g1_29dof_wbt_fast_sac_data.algo,
        config=replace(
            g1_29dof_wbt_fast_sac_data.algo.config,
            offline_dataset_path="offline_data/g1_29dof_wbt_fastsac_w_object_offline_collect_5m_dataset.h5",
        ),
    ),
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_29dof_wbt_fast_sac_w_object_episode_data = replace(
    g1_29dof_wbt_fast_sac_episode_data,
    algo=replace(
        g1_29dof_wbt_fast_sac_episode_data.algo,
        config=replace(
            g1_29dof_wbt_fast_sac_episode_data.algo.config,
            offline_dataset_path="offline_data/g1_29dof_wbt_fastsac_w_object_episode_2m_env128.h5",
            episode_data_active_envs=128,
        ),
    ),
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)


g1_29dof_wbt_cql_w_object = replace(
    g1_29dof_wbt_cql,
    algo=replace(
        g1_29dof_wbt_cql.algo,
        config=replace(
            g1_29dof_wbt_cql.algo.config,
            num_learning_iterations= 100000,
            offline_dataset_path=(
                "offline_data/g1_29dof_wbt_fastsac_w_object_episode_4m_env64.h5"
            ),
        ),
    ),
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_29dof_wbt_aw_cql_w_object = replace(
    g1_29dof_wbt_cql_w_object,
    training=replace(
        g1_29dof_wbt_cql_w_object.training,
        name="g1_29dof_wbt_aw_cql_w_object_manager",
    ),
    algo=AWCQLAlgoConfig(
        _target_="holosoma.agents.aw_cql.aw_cql_agent.AWCQLAgent",
        _recursive_=False,
        config=AWCQLConfig(**asdict(g1_29dof_wbt_cql_w_object.algo.config)),
    ),
)

g1_29dof_wbt_lse_aw_cql_w_object = replace(
    g1_29dof_wbt_aw_cql_w_object,
    training=replace(
        g1_29dof_wbt_aw_cql_w_object.training,
        name="g1_29dof_wbt_lse_aw_cql_w_object_manager",
    ),
    algo=AWCQLAlgoConfig(
        _target_="holosoma.agents.lse_aw_cql.lse_aw_cql_agent.LSEAWCQLAgent",
        _recursive_=False,
        config=AWCQLConfig(**asdict(g1_29dof_wbt_aw_cql_w_object.algo.config)),
    ),
)

g1_29dof_wbt_iql_w_object = replace(
    g1_29dof_wbt_iql,
    algo=replace(
        g1_29dof_wbt_iql.algo,
        config=replace(
            g1_29dof_wbt_iql.algo.config,
            num_learning_iterations=100000,
            critic_learning_rate=3e-4,
            value_learning_rate=3e-4,
            actor_learning_rate=3e-4,
            batch_size=g1_29dof_wbt_cql.algo.config.batch_size,
            num_updates=g1_29dof_wbt_cql.algo.config.num_updates,
            discount=g1_29dof_wbt_cql.algo.config.gamma,
            reward_scale=float(g1_29dof_wbt_cql.algo.config.reward_scale),
            bootstrap_truncations=g1_29dof_wbt_cql.algo.config.bootstrap_truncations,
            tau=0.005,
            offline_dataset_path=(
                "offline_data/g1_29dof_wbt_fastsac_w_object_episode_4m_env64.h5"
            ),
        ),
    ),
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_29dof_wbt_bc_w_object = replace(
    g1_29dof_wbt_bc,
    algo=replace(
        g1_29dof_wbt_bc.algo,
        config=replace(
            g1_29dof_wbt_bc.algo.config,
            num_learning_iterations= 100000,
            offline_dataset_path=(
                "offline_data/g1_29dof_wbt_fastsac_w_object_episode_4m_env64.h5"
            ),
        ),
    ),
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_29dof_wbt_w_bc_w_object = replace(
    g1_29dof_wbt_bc_w_object,
    training=replace(
        g1_29dof_wbt_bc_w_object.training,
        name="g1_29dof_wbt_w_bc_w_object_manager",
    ),
    algo=WBCAlgoConfig(
        _target_="holosoma.agents.w_bc.w_bc_agent.WBCAgent",
        _recursive_=False,
        config=WBCConfig(**asdict(g1_29dof_wbt_bc_w_object.algo.config)),
    ),
)

g1_29dof_wbt_td3_bc_w_object = replace(
    g1_29dof_wbt_td3_bc,
    training=replace(
        g1_29dof_wbt_td3_bc.training,
        name="g1_29dof_wbt_td3_bc_w_object_manager",
        num_envs=4096,
    ),
    algo=replace(
        g1_29dof_wbt_td3_bc.algo,
        config=replace(
            g1_29dof_wbt_td3_bc.algo.config,
            num_learning_iterations=100000,
            critic_learning_rate=3e-4,
            actor_learning_rate=3e-4,
            batch_size=1024,
            num_updates=4,
            discount=0.99,
            reward_scale=5.0,
            bootstrap_truncations=True,
            tau=0.005,
            offline_dataset_path="offline_data/g1_29dof_wbt_fastsac_w_object_episode_4m_env64.h5",
        ),
    ),
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

__all__ = [
    "g1_29dof_wbt",
    "g1_29dof_wbt_fast_sac",
    "g1_29dof_wbt_fast_sac_data",
    "g1_29dof_wbt_fixed_fast_sac_data",
    "g1_29dof_wbt_fast_sac_d3_seg_a_data",
    "g1_29dof_wbt_fast_sac_d3_seg_b_data",
    "g1_29dof_wbt_fast_sac_d3_seg_c_data",
    "g1_29dof_wbt_fast_sac_episode_data",
    "g1_29dof_wbt_iql",
    "g1_29dof_wbt_cql",
    "g1_29dof_wbt_lr_cql",
    "g1_29dof_wbt_vc_cql",
    "g1_29dof_wbt_aw_cql",
    "g1_29dof_wbt_os_aw_cql",
    "g1_29dof_wbt_lse_aw_cql",
    "g1_29dof_wbt_dw_cql",
    "g1_29dof_wbt_bf_cql",
    "g1_29dof_wbt_bc",
    "g1_29dof_wbt_w_bc",
    "g1_29dof_wbt_td3_bc",
    "g1_29dof_wbt_fast_sac_w_object",
    "g1_29dof_wbt_fast_sac_w_object_episode_data",
    "g1_29dof_wbt_w_object",
    "g1_29dof_wbt_cql_w_object",
    "g1_29dof_wbt_aw_cql_w_object",
    "g1_29dof_wbt_lse_aw_cql_w_object",
    "g1_29dof_wbt_iql_w_object",
    "g1_29dof_wbt_bc_w_object",
    "g1_29dof_wbt_w_bc_w_object",
    "g1_29dof_wbt_td3_bc_w_object",
    "g1_29dof_wbt_fast_sac_w_object_data",
]

"""
Example 1: Robot only:
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt

Example 2: Robot+Object:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object

Example 3: Robot+Terrain:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path="holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj" \
  --command.setup_terms.motion_command.params.motion_config.motion_file\
="holosoma/data/motions/g1_29dof/whole_body_tracking/motion_crawl_slope.npz" \
  --simulator.config.scene.env_spacing=0.0
"""
