"""Experiments for the independently configured K1 lafan-dance1 task."""

from dataclasses import asdict, replace

from holosoma.config_types.algo import (
    DWCQLAlgoConfig,
    DWCQLConfig,
    WBCAlgoConfig,
    WBCConfig,
)
from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import action, algo, robot, simulator, terrain
from holosoma.config_values.wbt.k1.lafan_dance1.command import (
    k1_23dof_wbt_lafan_dance1_command,
)
from holosoma.config_values.wbt.k1.lafan_dance1.curriculum import (
    k1_23dof_wbt_lafan_dance1_curriculum,
)
from holosoma.config_values.wbt.k1.lafan_dance1.observation import (
    k1_23dof_wbt_lafan_dance1_observation,
)
from holosoma.config_values.wbt.k1.lafan_dance1.randomization import (
    k1_23dof_wbt_lafan_dance1_randomization,
)
from holosoma.config_values.wbt.k1.lafan_dance1.reward import (
    k1_23dof_wbt_lafan_dance1_fast_sac_reward,
    k1_23dof_wbt_lafan_dance1_reward,
)
from holosoma.config_values.wbt.k1.lafan_dance1.termination import (
    k1_23dof_wbt_lafan_dance1_termination,
)

# Collection targets: FastSAC writes these, nothing reads them for training.
_FAST_SAC_DATASET = "offline_data/k1_23dof_wbt_lafan_dance1_fastsac_dataset.h5"
_EPISODE_COLLECT_DATASET = "offline_data/k1_23dof_wbt_lafan_dance1_fastsac_1m_episode_env256_dataset.h5"

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
            # The clip plus its 2s default-pose prepend is exactly 500 steps, and
            # freeze_at_timestep_zero_prob holds timestep 0 for a geometric number of
            # extra steps (mean 20). At a 500-step budget that overran the timeout on
            # 16% of training and 82% of eval episodes, which buried the motion_ends
            # success signal. 600 steps leaves 102 spare, covering 99.5% of the hold.
            sim=replace(simulator.isaacsim.config.sim, max_episode_length_s=12.0),
        ),
    )


def _robot():
    # NOTE: unlike G1 (action_scale=1.0), K1 keeps its per-joint action scale
    # (0.25 * effort_limit / kp) so exported policies match cyclo_lab's deployment stack.
    return replace(
        robot.k1_23dof,
        asset=replace(robot.k1_23dof.asset, enable_self_collisions=True),
        init_state=replace(robot.k1_23dof.init_state, pos=[0.0, 0.0, 0.764]),
    )


k1_23dof_wbt_lafan_dance1 = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_lafan_dance1_manager",
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
    observation=k1_23dof_wbt_lafan_dance1_observation,
    action=action.k1_23dof_joint_pos,
    termination=k1_23dof_wbt_lafan_dance1_termination,
    randomization=k1_23dof_wbt_lafan_dance1_randomization,
    command=k1_23dof_wbt_lafan_dance1_command,
    curriculum=k1_23dof_wbt_lafan_dance1_curriculum,
    reward=k1_23dof_wbt_lafan_dance1_reward,
    nightly=_NIGHTLY,
)


k1_23dof_wbt_lafan_dance1_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_lafan_dance1_fast_sac_manager",
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
    observation=k1_23dof_wbt_lafan_dance1_observation,
    action=action.k1_23dof_joint_pos,
    termination=k1_23dof_wbt_lafan_dance1_termination,
    randomization=k1_23dof_wbt_lafan_dance1_randomization,
    command=k1_23dof_wbt_lafan_dance1_command,
    curriculum=k1_23dof_wbt_lafan_dance1_curriculum,
    reward=k1_23dof_wbt_lafan_dance1_fast_sac_reward,
    nightly=_NIGHTLY,
)


k1_23dof_wbt_lafan_dance1_fast_sac_episode_data = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_lafan_dance1_fast_sac_episode_data_manager",
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
    observation=k1_23dof_wbt_lafan_dance1_observation,
    action=action.k1_23dof_joint_pos,
    termination=k1_23dof_wbt_lafan_dance1_termination,
    randomization=k1_23dof_wbt_lafan_dance1_randomization,
    command=k1_23dof_wbt_lafan_dance1_command,
    curriculum=k1_23dof_wbt_lafan_dance1_curriculum,
    reward=k1_23dof_wbt_lafan_dance1_fast_sac_reward,
    nightly=_NIGHTLY,
)


k1_23dof_wbt_lafan_dance1_cql = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_lafan_dance1_cql_manager_seed1",
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
    observation=k1_23dof_wbt_lafan_dance1_observation,
    action=action.k1_23dof_joint_pos,
    termination=k1_23dof_wbt_lafan_dance1_termination,
    randomization=k1_23dof_wbt_lafan_dance1_randomization,
    command=k1_23dof_wbt_lafan_dance1_command,
    curriculum=k1_23dof_wbt_lafan_dance1_curriculum,
    reward=k1_23dof_wbt_lafan_dance1_fast_sac_reward,
    nightly=_NIGHTLY,
)


k1_23dof_wbt_lafan_dance1_td3_bc = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_lafan_dance1_td3_bc_manager",
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
    observation=k1_23dof_wbt_lafan_dance1_observation,
    action=action.k1_23dof_joint_pos,
    termination=k1_23dof_wbt_lafan_dance1_termination,
    randomization=k1_23dof_wbt_lafan_dance1_randomization,
    command=k1_23dof_wbt_lafan_dance1_command,
    curriculum=k1_23dof_wbt_lafan_dance1_curriculum,
    reward=k1_23dof_wbt_lafan_dance1_fast_sac_reward,
    nightly=_NIGHTLY,
)


k1_23dof_wbt_lafan_dance1_iql = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_lafan_dance1_iql_manager_seed1",
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
    observation=k1_23dof_wbt_lafan_dance1_observation,
    action=action.k1_23dof_joint_pos,
    termination=k1_23dof_wbt_lafan_dance1_termination,
    randomization=k1_23dof_wbt_lafan_dance1_randomization,
    command=k1_23dof_wbt_lafan_dance1_command,
    curriculum=k1_23dof_wbt_lafan_dance1_curriculum,
    reward=k1_23dof_wbt_lafan_dance1_fast_sac_reward,
    nightly=_NIGHTLY,
)


k1_23dof_wbt_lafan_dance1_aw_cql = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_lafan_dance1_aw_cql_manager_seed1",
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
    observation=k1_23dof_wbt_lafan_dance1_observation,
    action=action.k1_23dof_joint_pos,
    termination=k1_23dof_wbt_lafan_dance1_termination,
    randomization=k1_23dof_wbt_lafan_dance1_randomization,
    command=k1_23dof_wbt_lafan_dance1_command,
    curriculum=k1_23dof_wbt_lafan_dance1_curriculum,
    reward=k1_23dof_wbt_lafan_dance1_fast_sac_reward,
    nightly=_NIGHTLY,
)


k1_23dof_wbt_lafan_dance1_b_arm = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_lafan_dance1_b_arm_manager",
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
    observation=k1_23dof_wbt_lafan_dance1_observation,
    action=action.k1_23dof_joint_pos,
    termination=k1_23dof_wbt_lafan_dance1_termination,
    randomization=k1_23dof_wbt_lafan_dance1_randomization,
    command=k1_23dof_wbt_lafan_dance1_command,
    curriculum=k1_23dof_wbt_lafan_dance1_curriculum,
    reward=k1_23dof_wbt_lafan_dance1_fast_sac_reward,
    nightly=_NIGHTLY,
)


k1_23dof_wbt_lafan_dance1_c_arm = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_lafan_dance1_c_arm_manager",
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
    observation=k1_23dof_wbt_lafan_dance1_observation,
    action=action.k1_23dof_joint_pos,
    termination=k1_23dof_wbt_lafan_dance1_termination,
    randomization=k1_23dof_wbt_lafan_dance1_randomization,
    command=k1_23dof_wbt_lafan_dance1_command,
    curriculum=k1_23dof_wbt_lafan_dance1_curriculum,
    reward=k1_23dof_wbt_lafan_dance1_fast_sac_reward,
    nightly=_NIGHTLY,
)


k1_23dof_wbt_lafan_dance1_dw_cql = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_lafan_dance1_dw_cql_manager",
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
    observation=k1_23dof_wbt_lafan_dance1_observation,
    action=action.k1_23dof_joint_pos,
    termination=k1_23dof_wbt_lafan_dance1_termination,
    randomization=k1_23dof_wbt_lafan_dance1_randomization,
    command=k1_23dof_wbt_lafan_dance1_command,
    curriculum=k1_23dof_wbt_lafan_dance1_curriculum,
    reward=k1_23dof_wbt_lafan_dance1_fast_sac_reward,
    nightly=_NIGHTLY,
)


k1_23dof_wbt_lafan_dance1_bc = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_lafan_dance1_bc_manager",
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
    observation=k1_23dof_wbt_lafan_dance1_observation,
    action=action.k1_23dof_joint_pos,
    termination=k1_23dof_wbt_lafan_dance1_termination,
    randomization=k1_23dof_wbt_lafan_dance1_randomization,
    command=k1_23dof_wbt_lafan_dance1_command,
    curriculum=k1_23dof_wbt_lafan_dance1_curriculum,
    reward=k1_23dof_wbt_lafan_dance1_fast_sac_reward,
    nightly=_NIGHTLY,
)


_W_BC_CONFIG = replace(
    WBCConfig(**asdict(k1_23dof_wbt_lafan_dance1_bc.algo.config)),
    offline_dataset_path=_W_BC_DATASET,
    aw_weights_path=_W_BC_WEIGHTS,
)

k1_23dof_wbt_lafan_dance1_w_bc = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="k1_23dof_wbt_lafan_dance1_w_bc_manager",
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
    observation=k1_23dof_wbt_lafan_dance1_observation,
    action=action.k1_23dof_joint_pos,
    termination=k1_23dof_wbt_lafan_dance1_termination,
    randomization=k1_23dof_wbt_lafan_dance1_randomization,
    command=k1_23dof_wbt_lafan_dance1_command,
    curriculum=k1_23dof_wbt_lafan_dance1_curriculum,
    reward=k1_23dof_wbt_lafan_dance1_fast_sac_reward,
    nightly=_NIGHTLY,
)


__all__ = [
    "k1_23dof_wbt_lafan_dance1",
    "k1_23dof_wbt_lafan_dance1_aw_cql",
    "k1_23dof_wbt_lafan_dance1_b_arm",
    "k1_23dof_wbt_lafan_dance1_bc",
    "k1_23dof_wbt_lafan_dance1_c_arm",
    "k1_23dof_wbt_lafan_dance1_cql",
    "k1_23dof_wbt_lafan_dance1_dw_cql",
    "k1_23dof_wbt_lafan_dance1_fast_sac",
    "k1_23dof_wbt_lafan_dance1_fast_sac_episode_data",
    "k1_23dof_wbt_lafan_dance1_iql",
    "k1_23dof_wbt_lafan_dance1_td3_bc",
    "k1_23dof_wbt_lafan_dance1_w_bc",
]
