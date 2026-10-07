"""Experiments for the independently configured G1 lafan-kick task."""

from dataclasses import asdict, replace

from holosoma.config_types.algo import (
    ACLQLAlgoConfig,
    ACLQLConfig,
    DWCQLAlgoConfig,
    DWCQLConfig,
    HongAdvantageCQLAlgoConfig,
    HongAdvantageCQLConfig,
    WBCAlgoConfig,
    WBCConfig,
)
from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import action, algo, robot, simulator, terrain
from holosoma.config_values.wbt.g1.lafan_kick.command import (
    g1_29dof_wbt_lafan_kick_command,
)
from holosoma.config_values.wbt.g1.lafan_kick.curriculum import (
    g1_29dof_wbt_lafan_kick_curriculum,
)
from holosoma.config_values.wbt.g1.lafan_kick.observation import (
    g1_29dof_wbt_lafan_kick_observation,
)
from holosoma.config_values.wbt.g1.lafan_kick.randomization import (
    g1_29dof_wbt_lafan_kick_randomization,
)
from holosoma.config_values.wbt.g1.lafan_kick.reward import (
    g1_29dof_wbt_lafan_kick_fast_sac_reward,
    g1_29dof_wbt_lafan_kick_reward,
)
from holosoma.config_values.wbt.g1.lafan_kick.termination import (
    g1_29dof_wbt_lafan_kick_termination,
)

# Collection targets: FastSAC writes these, nothing reads them for training.
_FAST_SAC_DATASET = "offline_data/g1_29dof_wbt_lafan_kick_fastsac_dataset.h5"
_EPISODE_COLLECT_DATASET = "offline_data/g1_29dof_wbt_lafan_kick_fastsac_1m_episode_env256_dataset.h5"

# Every offline learner trains on the collected episode dataset above, so the algorithm
# comparison is paired by construction. Kept as one assignment per algorithm rather than
# a single shared name, so one arm can be re-pointed without disturbing the others.
_CQL_DATASET = _EPISODE_COLLECT_DATASET
_IQL_DATASET = _EPISODE_COLLECT_DATASET
_TD3_BC_DATASET = _EPISODE_COLLECT_DATASET
_AW_CQL_DATASET = _EPISODE_COLLECT_DATASET
_AW_CQL_WEIGHTS = f"{_AW_CQL_DATASET}.aw_weights.npz"
_OS_AW_CQL_DATASET = _EPISODE_COLLECT_DATASET
_OS_AW_CQL_WEIGHTS = f"{_OS_AW_CQL_DATASET}.aw_weights.npz"
_LSE_AW_CQL_DATASET = _EPISODE_COLLECT_DATASET
_LSE_AW_CQL_WEIGHTS = f"{_LSE_AW_CQL_DATASET}.aw_weights.npz"
_DW_CQL_DATASET = _EPISODE_COLLECT_DATASET
_DW_CQL_WEIGHTS = f"{_DW_CQL_DATASET}.aw_weights.npz"
_BC_DATASET = _EPISODE_COLLECT_DATASET
_W_BC_DATASET = _EPISODE_COLLECT_DATASET
_W_BC_WEIGHTS = f"{_W_BC_DATASET}.aw_weights.npz"


_NIGHTLY = NightlyConfig(
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


def _simulator():
    return replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            scene=replace(simulator.isaacsim.config.scene, env_spacing=2.5),
            # The clip (195 frames @ 50 Hz) plus its 2 s prepend and 2 s append is 395 steps;
            # freeze_at_timestep_zero_prob holds timestep 0 for a geometric number of extra
            # steps (mean 20). 600 steps (12 s), the same budget as largebox and dance1.
            sim=replace(simulator.isaacsim.config.sim, max_episode_length_s=12.0),
        ),
    )


def _robot():
    return replace(
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_scale=1.0),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    )


g1_29dof_wbt_lafan_kick = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_lafan_kick_manager",
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
    simulator=_simulator(),
    robot=_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=g1_29dof_wbt_lafan_kick_observation,
    action=action.g1_29dof_joint_pos,
    termination=g1_29dof_wbt_lafan_kick_termination,
    randomization=g1_29dof_wbt_lafan_kick_randomization,
    command=g1_29dof_wbt_lafan_kick_command,
    curriculum=g1_29dof_wbt_lafan_kick_curriculum,
    reward=g1_29dof_wbt_lafan_kick_reward,
    nightly=_NIGHTLY,
)


g1_29dof_wbt_lafan_kick_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_lafan_kick_fast_sac_manager",
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
            offline_dataset_path=_FAST_SAC_DATASET,
        ),
    ),
    simulator=_simulator(),
    robot=_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=g1_29dof_wbt_lafan_kick_observation,
    action=action.g1_29dof_joint_pos,
    termination=g1_29dof_wbt_lafan_kick_termination,
    randomization=g1_29dof_wbt_lafan_kick_randomization,
    command=g1_29dof_wbt_lafan_kick_command,
    curriculum=g1_29dof_wbt_lafan_kick_curriculum,
    reward=g1_29dof_wbt_lafan_kick_fast_sac_reward,
    nightly=_NIGHTLY,
)


g1_29dof_wbt_lafan_kick_fast_sac_episode_data = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_lafan_kick_fast_sac_episode_data_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fast_sac_episode_data,
        config=replace(
            algo.fast_sac_episode_data.config,
            num_learning_iterations=10000,
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
            offline_dataset_path=_EPISODE_COLLECT_DATASET,
            episode_data_active_envs=256,
        ),
    ),
    simulator=_simulator(),
    robot=_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=g1_29dof_wbt_lafan_kick_observation,
    action=action.g1_29dof_joint_pos,
    termination=g1_29dof_wbt_lafan_kick_termination,
    randomization=g1_29dof_wbt_lafan_kick_randomization,
    command=g1_29dof_wbt_lafan_kick_command,
    curriculum=g1_29dof_wbt_lafan_kick_curriculum,
    reward=g1_29dof_wbt_lafan_kick_fast_sac_reward,
    nightly=_NIGHTLY,
)


g1_29dof_wbt_lafan_kick_cql = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_lafan_kick_cql_manager_seed1",
        num_envs=4096,
        eval_num_episodes=1,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.cql,
        config=replace(
            algo.cql.config,
            num_learning_iterations=100000,
            gamma=0.99,
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
            offline_dataset_path=_CQL_DATASET,
            use_gpu_cache=True,
            reward_scale=5.0,
            bellman_loss_type="mse",
            huber_beta=5.0,
            cql_max_target_backup=False,
        ),
    ),
    simulator=_simulator(),
    robot=_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=g1_29dof_wbt_lafan_kick_observation,
    action=action.g1_29dof_joint_pos,
    termination=g1_29dof_wbt_lafan_kick_termination,
    randomization=g1_29dof_wbt_lafan_kick_randomization,
    command=g1_29dof_wbt_lafan_kick_command,
    curriculum=g1_29dof_wbt_lafan_kick_curriculum,
    reward=g1_29dof_wbt_lafan_kick_fast_sac_reward,
    nightly=_NIGHTLY,
)


# ACL-QL on the lafan-kick dataset (cross-motion table): same ACL settings as the
# G1 main-table arm, paired with the lafan CQL arm (dataset, iterations, alpha=5).
# Needs offline_data/<lafan h5>.acl_quality.npz (scripts/acl_precompute_quality.py).
g1_29dof_wbt_lafan_kick_acl_ql = replace(
    g1_29dof_wbt_lafan_kick_cql,
    training=replace(g1_29dof_wbt_lafan_kick_cql.training, name="g1_29dof_wbt_lafan_kick_acl_ql_manager"),
    algo=ACLQLAlgoConfig(
        _target_="holosoma.agents.acl_ql.acl_ql_agent.ACLQLAgent",
        _recursive_=False,
        config=replace(
            ACLQLConfig(**asdict(g1_29dof_wbt_lafan_kick_cql.algo.config)),
            acl_weight_learning_rate=3e-4,
            acl_behavior_learning_rate=3e-4,
            acl_behavior_pretrain_steps=100000,
            acl_quality_lambda=0.5,
        ),
    ),
)


g1_29dof_wbt_lafan_kick_td3_bc = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_lafan_kick_td3_bc_manager",
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
            tau=0.005,
            policy_delay=2,
            use_symmetry=False,
            offline_dataset_path=_TD3_BC_DATASET,
        ),
    ),
    simulator=_simulator(),
    robot=_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=g1_29dof_wbt_lafan_kick_observation,
    action=action.g1_29dof_joint_pos,
    termination=g1_29dof_wbt_lafan_kick_termination,
    randomization=g1_29dof_wbt_lafan_kick_randomization,
    command=g1_29dof_wbt_lafan_kick_command,
    curriculum=g1_29dof_wbt_lafan_kick_curriculum,
    reward=g1_29dof_wbt_lafan_kick_fast_sac_reward,
    nightly=_NIGHTLY,
)


g1_29dof_wbt_lafan_kick_iql = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_lafan_kick_iql_manager_seed1",
        num_envs=4096,
        eval_num_episodes=1,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.iql,
        config=replace(
            algo.iql.config,
            num_learning_iterations=100000,
            discount=0.99,
            reward_scale=5.0,
            num_updates=4,
            tau=0.05,
            expectile=0.7,
            beta=3.0,
            max_weight=100.0,
            use_symmetry=False,
            offline_dataset_path=_IQL_DATASET,
        ),
    ),
    simulator=_simulator(),
    robot=_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=g1_29dof_wbt_lafan_kick_observation,
    action=action.g1_29dof_joint_pos,
    termination=g1_29dof_wbt_lafan_kick_termination,
    randomization=g1_29dof_wbt_lafan_kick_randomization,
    command=g1_29dof_wbt_lafan_kick_command,
    curriculum=g1_29dof_wbt_lafan_kick_curriculum,
    reward=g1_29dof_wbt_lafan_kick_fast_sac_reward,
    nightly=_NIGHTLY,
)


g1_29dof_wbt_lafan_kick_aw_cql = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_lafan_kick_aw_cql_manager_seed1",
        num_envs=4096,
        eval_num_episodes=1,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.aw_cql,
        config=replace(
            algo.aw_cql.config,
            num_learning_iterations=100000,
            gamma=0.99,
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
            offline_dataset_path=_AW_CQL_DATASET,
            aw_weights_path=_AW_CQL_WEIGHTS,
            use_gpu_cache=True,
            reward_scale=5.0,
            bellman_loss_type="mse",
            huber_beta=5.0,
            cql_max_target_backup=False,
        ),
    ),
    simulator=_simulator(),
    robot=_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=g1_29dof_wbt_lafan_kick_observation,
    action=action.g1_29dof_joint_pos,
    termination=g1_29dof_wbt_lafan_kick_termination,
    randomization=g1_29dof_wbt_lafan_kick_randomization,
    command=g1_29dof_wbt_lafan_kick_command,
    curriculum=g1_29dof_wbt_lafan_kick_curriculum,
    reward=g1_29dof_wbt_lafan_kick_fast_sac_reward,
    nightly=_NIGHTLY,
)


g1_29dof_wbt_lafan_kick_b_arm = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_lafan_kick_b_arm_manager",
        num_envs=4096,
        eval_num_episodes=1,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.b_arm,
        config=replace(
            algo.b_arm.config,
            num_learning_iterations=100000,
            gamma=0.99,
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
            offline_dataset_path=_OS_AW_CQL_DATASET,
            aw_weights_path=_OS_AW_CQL_WEIGHTS,
            use_gpu_cache=True,
            reward_scale=5.0,
            bellman_loss_type="mse",
            huber_beta=5.0,
            cql_max_target_backup=False,
        ),
    ),
    simulator=_simulator(),
    robot=_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=g1_29dof_wbt_lafan_kick_observation,
    action=action.g1_29dof_joint_pos,
    termination=g1_29dof_wbt_lafan_kick_termination,
    randomization=g1_29dof_wbt_lafan_kick_randomization,
    command=g1_29dof_wbt_lafan_kick_command,
    curriculum=g1_29dof_wbt_lafan_kick_curriculum,
    reward=g1_29dof_wbt_lafan_kick_fast_sac_reward,
    nightly=_NIGHTLY,
)


g1_29dof_wbt_lafan_kick_c_arm = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_lafan_kick_c_arm_manager",
        num_envs=4096,
        eval_num_episodes=1,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.c_arm,
        config=replace(
            algo.c_arm.config,
            num_learning_iterations=100000,
            gamma=0.99,
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
            offline_dataset_path=_LSE_AW_CQL_DATASET,
            aw_weights_path=_LSE_AW_CQL_WEIGHTS,
            use_gpu_cache=True,
            reward_scale=5.0,
            bellman_loss_type="mse",
            huber_beta=5.0,
            cql_max_target_backup=False,
        ),
    ),
    simulator=_simulator(),
    robot=_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=g1_29dof_wbt_lafan_kick_observation,
    action=action.g1_29dof_joint_pos,
    termination=g1_29dof_wbt_lafan_kick_termination,
    randomization=g1_29dof_wbt_lafan_kick_randomization,
    command=g1_29dof_wbt_lafan_kick_command,
    curriculum=g1_29dof_wbt_lafan_kick_curriculum,
    reward=g1_29dof_wbt_lafan_kick_fast_sac_reward,
    nightly=_NIGHTLY,
)


g1_29dof_wbt_lafan_kick_dw_cql = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_lafan_kick_dw_cql_manager",
        num_envs=4096,
        eval_num_episodes=1,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=DWCQLAlgoConfig(
        _target_="holosoma.agents.dw_cql.dw_cql_agent.DWCQLAgent",
        _recursive_=False,
        config=DWCQLConfig(
            **{
                **asdict(algo.aw_cql.config),
                "num_learning_iterations": 100000,
                "gamma": 0.99,
                "num_updates": 4,
                "policy_frequency": 1,
                "target_entropy_ratio": 0.5,
                "tau": 0.05,
                "cql_weight": 5.0,
                "cql_num_action_samples": 10,
                "use_symmetry": False,
                "use_lagrange": False,
                "batch_size": 1024,
                "cql_target_action_gap": 0.0,
                "offline_dataset_path": _DW_CQL_DATASET,
                "aw_weights_path": _DW_CQL_WEIGHTS,
                "use_gpu_cache": True,
                "reward_scale": 5.0,
                "bellman_loss_type": "mse",
                "huber_beta": 5.0,
                "cql_max_target_backup": False,
            }
        ),
    ),
    simulator=_simulator(),
    robot=_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=g1_29dof_wbt_lafan_kick_observation,
    action=action.g1_29dof_joint_pos,
    termination=g1_29dof_wbt_lafan_kick_termination,
    randomization=g1_29dof_wbt_lafan_kick_randomization,
    command=g1_29dof_wbt_lafan_kick_command,
    curriculum=g1_29dof_wbt_lafan_kick_curriculum,
    reward=g1_29dof_wbt_lafan_kick_fast_sac_reward,
    nightly=_NIGHTLY,
)


g1_29dof_wbt_lafan_kick_bc = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_lafan_kick_bc_manager",
        num_envs=4096,
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
            offline_dataset_path=_BC_DATASET,
        ),
    ),
    simulator=_simulator(),
    robot=_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=g1_29dof_wbt_lafan_kick_observation,
    action=action.g1_29dof_joint_pos,
    termination=g1_29dof_wbt_lafan_kick_termination,
    randomization=g1_29dof_wbt_lafan_kick_randomization,
    command=g1_29dof_wbt_lafan_kick_command,
    curriculum=g1_29dof_wbt_lafan_kick_curriculum,
    reward=g1_29dof_wbt_lafan_kick_fast_sac_reward,
    nightly=_NIGHTLY,
)


_W_BC_CONFIG = replace(
    WBCConfig(**asdict(g1_29dof_wbt_lafan_kick_bc.algo.config)),
    offline_dataset_path=_W_BC_DATASET,
    aw_weights_path=_W_BC_WEIGHTS,
)

g1_29dof_wbt_lafan_kick_w_bc = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_lafan_kick_w_bc_manager",
        num_envs=4096,
        eval_num_episodes=1,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=WBCAlgoConfig(
        _target_="holosoma.agents.w_bc.w_bc_agent.WBCAgent",
        _recursive_=False,
        config=_W_BC_CONFIG,
    ),
    simulator=_simulator(),
    robot=_robot(),
    terrain=terrain.terrain_locomotion_plane,
    observation=g1_29dof_wbt_lafan_kick_observation,
    action=action.g1_29dof_joint_pos,
    termination=g1_29dof_wbt_lafan_kick_termination,
    randomization=g1_29dof_wbt_lafan_kick_randomization,
    command=g1_29dof_wbt_lafan_kick_command,
    curriculum=g1_29dof_wbt_lafan_kick_curriculum,
    reward=g1_29dof_wbt_lafan_kick_fast_sac_reward,
    nightly=_NIGHTLY,
)


# Hong-Advantage CQL: paired with CQL (same dataset and hyperparameters, loss untouched); only the
# minibatch sampling distribution differs (Hong et al. ICLR 2023 trajectory AW, temperature 0.2).
g1_29dof_wbt_lafan_kick_hong_advantage_cql = replace(
    g1_29dof_wbt_lafan_kick_cql,
    training=replace(g1_29dof_wbt_lafan_kick_cql.training, name="g1_29dof_wbt_lafan_kick_hong_advantage_cql_manager"),
    algo=HongAdvantageCQLAlgoConfig(
        _target_="holosoma.agents.hong_advantage_cql.hong_advantage_cql_agent.HongAdvantageCQLAgent",
        _recursive_=False,
        config=HongAdvantageCQLConfig(**asdict(g1_29dof_wbt_lafan_kick_cql.algo.config)),
    ),
)


__all__ = [
    "g1_29dof_wbt_lafan_kick",
    "g1_29dof_wbt_lafan_kick_acl_ql",
    "g1_29dof_wbt_lafan_kick_aw_cql",
    "g1_29dof_wbt_lafan_kick_b_arm",
    "g1_29dof_wbt_lafan_kick_bc",
    "g1_29dof_wbt_lafan_kick_c_arm",
    "g1_29dof_wbt_lafan_kick_cql",
    "g1_29dof_wbt_lafan_kick_hong_advantage_cql",
    "g1_29dof_wbt_lafan_kick_dw_cql",
    "g1_29dof_wbt_lafan_kick_fast_sac",
    "g1_29dof_wbt_lafan_kick_fast_sac_episode_data",
    "g1_29dof_wbt_lafan_kick_iql",
    "g1_29dof_wbt_lafan_kick_td3_bc",
    "g1_29dof_wbt_lafan_kick_w_bc",
]
