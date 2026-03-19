"""Offline CODAC training script.

Usage:
    python train_offline_codac.py
    python train_offline_codac.py --dataset offline_data/fastsac_dataset.h5 --epochs 500
    python train_offline_codac.py --conservative-weight 1.0 --conservative-coef 5.0
    python train_offline_codac.py --resume logs/offline_codac/model_0000500.pt
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from holosoma.agents.offline_CODAC.offline_codac_agent import OfflineCODACAgent
from holosoma.agents.offline_CODAC.offline_codac_config import CODACConfig
from holosoma.config_types.algo import FastSACAlgoConfig, FastSACConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Offline CODAC agent from HDF5 dataset")

    # Dataset & dimensions
    parser.add_argument("--dataset", type=str, default="offline_data/fastsac_dataset.h5",
                        help="Path to the offline HDF5 dataset")
    parser.add_argument("--actor-obs-dim", type=int, default=154,
                        help="Actor observation dimension")
    parser.add_argument("--critic-obs-dim", type=int, default=298,
                        help="Critic observation dimension")
    parser.add_argument("--action-dim", type=int, default=29,
                        help="Action dimension")

    # Training
    parser.add_argument("--epochs", type=int, default=10000,
                        help="Number of training epochs (num_learning_iterations)")
    parser.add_argument("--updates-per-epoch", type=int, default=1000,
                        help="Gradient steps per epoch")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Batch size for sampling from replay buffer")
    parser.add_argument("--gamma", type=float, default=0.97,
                        help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.125,
                        help="Target network smoothing coefficient")

    # Normalizer
    parser.add_argument("--normalizer-mode", type=str, default="dataset",
                        choices=["dataset", "checkpoint", "none"],
                        help="How to initialize observation normalizer")

    # Conservative regularization (CODAC-specific)
    parser.add_argument("--conservative-weight", type=float, default=1.0,
                        help="Global conservative weight in FastSACConfig (parent hook multiplier)")
    parser.add_argument("--conservative-coef", type=float, default=5.0,
                        help="CODAC-side conservative coefficient (multiplied inside hook)")
    parser.add_argument("--num-conservative-actions", type=int, default=10,
                        help="Number of OOD actions per source per observation")
    parser.add_argument("--conservative-action-sample-mode", type=str, default="random_policy",
                        choices=["random", "policy", "random_policy", "random_policy_next"],
                        help="Which OOD action sources to use")
    parser.add_argument("--codac-risk-mode", type=str, default="neutral",
                        choices=["neutral", "cvar", "wang", "power", "quantile"],
                        help="Risk measure for distributional Q aggregation")
    parser.add_argument("--codac-risk-param", type=float, default=1.0,
                        help="Parameter for the risk measure")
    parser.add_argument("--codac-target-mode", type=str, default="mean",
                        choices=["mean", "min", "individual"],
                        help="How to combine Q-ensemble for penalty")
    parser.add_argument("--conservative-temp", type=float, default=1.0,
                        help="Logsumexp temperature for conservative penalty")
    parser.add_argument("--actor-bc-coef", type=float, default=0.0,
                        help="Behavior-cloning regularization coefficient (0 = disabled)")
    parser.add_argument("--log-codac-debug-metrics", action="store_true", default=True,
                        help="Log CODAC-specific diagnostic metrics")

    # Logging & checkpointing
    parser.add_argument("--log-dir", type=str, default="./logs/offline_codac",
                        help="Directory to save logs and checkpoints")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--logging-interval", type=int, default=10,
                        help="Log metrics every N epochs")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on (cuda or cpu)")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = FastSACConfig(
        offline_mode=True,
        offline_dataset_path=args.dataset,
        actor_obs_dim=args.actor_obs_dim,
        critic_obs_dim=args.critic_obs_dim,
        action_dim=args.action_dim,
        num_learning_iterations=args.epochs,
        offline_num_updates_per_epoch=args.updates_per_epoch,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        offline_normalizer_init_mode=args.normalizer_mode,
        conservative_weight=args.conservative_weight,
        save_interval=args.save_interval,
        logging_interval=args.logging_interval,
    )

    codac_config = CODACConfig(
        conservative_coef=args.conservative_coef,
        num_conservative_actions=args.num_conservative_actions,
        conservative_action_sample_mode=args.conservative_action_sample_mode,
        codac_risk_mode=args.codac_risk_mode,
        codac_risk_param=args.codac_risk_param,
        codac_target_mode=args.codac_target_mode,
        conservative_temp=args.conservative_temp,
        actor_bc_coef=args.actor_bc_coef,
        log_codac_debug_metrics=args.log_codac_debug_metrics,
    )

    agent = OfflineCODACAgent(
        config=config,
        device=args.device,
        log_dir=args.log_dir,
        codac_config=codac_config,
        env=None,
    )

    agent.setup()

    # Attach experiment config metadata so checkpoints match online FastSAC format
    from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_fast_sac_w_object

    experiment_config = replace(
        g1_29dof_wbt_fast_sac_w_object,
        algo=FastSACAlgoConfig(
            _target_="holosoma.agents.offline_CODAC.offline_codac_agent.OfflineCODACAgent",
            _recursive_=False,
            config=config,
        ),
    )
    agent.attach_checkpoint_metadata(experiment_config)

    if args.resume:
        agent.load(args.resume)

    agent.learn()


if __name__ == "__main__":
    main()
