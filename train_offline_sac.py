"""Offline FastSAC training script.

Usage:
    python train_offline_sac.py
    python train_offline_sac.py --dataset offline_data/fastsac_dataset.h5 --epochs 500
    python train_offline_sac.py --resume logs/offline_fastsac/model_0000500.pt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from holosoma.agents.offline_fast_sac.offline_fast_sac_agent import OfflineFastSACAgent
from holosoma.config_types.algo import FastSACConfig, FastSACAlgoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Offline FastSAC agent from HDF5 dataset")

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
    parser.add_argument("--epochs", type=int, default=10000, # 500,
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

    # Conservative (CQL-style) regularization
    parser.add_argument("--conservative-weight", type=float, default=0.0,
                        help="Weight for conservative critic loss (0 = disabled)")

    # Logging & checkpointing
    parser.add_argument("--log-dir", type=str, default="./logs/offline_fastsac",
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

    agent = OfflineFastSACAgent(
        config=config,
        device=args.device,
        log_dir=args.log_dir,
        env=None,
    )

    agent.setup()

    # Attach experiment config metadata so checkpoints match online FastSAC format
    from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_fast_sac_w_object

    experiment_config = replace(
        g1_29dof_wbt_fast_sac_w_object,
        algo=FastSACAlgoConfig(
            _target_="holosoma.agents.offline_fast_sac.offline_fast_sac_agent.OfflineFastSACAgent",
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
