"""PSC Part 0: dataset action-covariance spectrum measurement (standalone, no training).

Loads dataset actions from an offline HDF5, maps them into THE SAME action
representation the critic consumes in the target run config (env-scaled raw
actions by default — the current reference config uses no action normalization;
pass --space normalized only if the target run sets normalized_action_training),
eigendecomposes the covariance, reports the spectrum / effective rank / physical-
group alignment, and saves {mu, U, eigvals, meta} to a .pt consumed by PSCAgent.

Example invocations:
  # tracking (reference config: env-scaled actions, control action_scale = 1.0)
  python scripts/psc_spectrum.py \
      offline_data/g1_29dof_wbt_fastsac_offline_collect_5m_dataset.h5 \
      --space env --control-action-scale 1.0 \
      --out offline_data/psc_basis_wbt.pt --plot offline_data/psc_spectrum_wbt.png

  # locomotion (env-scaled actions, robot-config default action_scale = 0.25)
  python scripts/psc_spectrum.py \
      offline_data/g1_29dof_loco_fastsac_dataset.h5 \
      --space env \
      --out offline_data/psc_basis_loco.pt --plot offline_data/psc_spectrum_loco.png
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

from holosoma.agents.bf_cql.bf_cql import GROUP_PRESETS
from holosoma.config_values import robot as robot_values


def _action_boundaries(robot_name: str, control_action_scale: float | None) -> tuple[torch.Tensor, list[str], float]:
    """Replicate BFCQLEnv._compute_action_boundaries from the robot config."""
    robot_cfg = getattr(robot_values, robot_name)
    lower = torch.tensor(robot_cfg.dof_pos_lower_limit_list, dtype=torch.float64)
    upper = torch.tensor(robot_cfg.dof_pos_upper_limit_list, dtype=torch.float64)
    default = torch.zeros(len(robot_cfg.dof_names), dtype=torch.float64)
    for i, name in enumerate(robot_cfg.dof_names):
        if name in robot_cfg.init_state.default_joint_angles:
            default[i] = robot_cfg.init_state.default_joint_angles[name]
    scale = float(control_action_scale) if control_action_scale is not None else float(robot_cfg.control.action_scale)
    boundaries = torch.maximum((lower - default).abs(), (upper - default).abs()) / scale
    return boundaries, list(robot_cfg.dof_names), scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="offline HDF5 dataset with an 'actions' key")
    parser.add_argument("--space", choices=["env", "normalized"], default="env",
                        help="action representation of the TARGET run (env = no action normalization)")
    parser.add_argument("--robot", default="g1_29dof")
    parser.add_argument("--control-action-scale", type=float, default=None,
                        help="override robot-config control.action_scale (wbt experiments use 1.0)")
    parser.add_argument("--grouping", default="functional_9", choices=sorted(GROUP_PRESETS))
    parser.add_argument("--max-samples", type=int, default=1_000_000,
                        help="deterministic head slice of the dataset")
    parser.add_argument("--out", type=Path, required=True, help="output .pt basis file")
    parser.add_argument("--plot", type=Path, default=None, help="optional spectrum .png")
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as f:
        num_samples = int(f.attrs.get("num_samples", f["actions"].shape[0]))
        n = min(num_samples, args.max_samples)
        actions = torch.from_numpy(np.asarray(f["actions"][:n])).to(torch.float64)
    action_dim = actions.shape[1]
    print(f"dataset: {args.dataset} | samples used: {n}/{num_samples} | action_dim: {action_dim}")

    boundaries, dof_names, control_scale = _action_boundaries(args.robot, args.control_action_scale)
    assert boundaries.numel() == action_dim, (boundaries.numel(), action_dim)
    if args.space == "normalized":
        actions = (actions / (boundaries + 1e-6)).clamp(-1.0, 1.0)
        print(f"action space: NORMALIZED u-space [-1,1] (boundaries from {args.robot}, "
              f"control.action_scale={control_scale})")
    else:
        print(f"action space: ENV-scaled raw actions (reference no-normalization config; "
              f"control.action_scale={control_scale} only recorded in meta)")

    mu = actions.mean(dim=0)
    centered = actions - mu
    sigma = centered.T @ centered / max(n - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(sigma)  # ascending
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0.0)
    U = eigvecs[:, order]  # columns = eigen-directions, descending eigenvalue

    total = float(eigvals.sum())
    ratios = eigvals / max(total, 1e-12)
    cum = torch.cumsum(ratios, dim=0)
    p = ratios.clamp_min(1e-12)
    effective_rank = float(torch.exp(-(p * p.log()).sum()))

    print("\neigenvalue spectrum (descending):")
    for i in range(action_dim):
        print(f"  {i:2d}: eig={eigvals[i]:.6e}  var%={100*ratios[i]:6.2f}  cum%={100*cum[i]:6.2f}")
    print(f"\neffective rank exp(H(lambda)): {effective_rank:.2f} / {action_dim}")
    for frac in (0.5, 0.9, 0.95, 0.99):
        k = int((cum < frac).sum().item()) + 1
        print(f"  dims for {int(frac*100)}% variance: {k}")

    # physical-group alignment: projection energy of each group's coordinate
    # subspace onto the top-k eigenspace (1.0 = group lies inside the eigenspace)
    name_to_idx = {name: i for i, name in enumerate(dof_names)}
    groups = [(gname, [name_to_idx[j] for j in joints]) for gname, joints in GROUP_PRESETS[args.grouping]]
    print(f"\nphysical-group ({args.grouping}) projection energy onto top-k eigenspace:")
    ks = sorted({3, 9, 15, int((cum < 0.9).sum().item()) + 1})
    header = "  " + f"{'group':22s}" + "".join(f" k={k:<4d}" for k in ks)
    print(header)
    for gname, dims in groups:
        row = f"  {gname:22s}"
        for k in ks:
            energy = float((U[dims, :k] ** 2).sum() / len(dims))
            row += f" {energy:5.3f} "
        print(row)

    payload = {
        "mu": mu.to(torch.float32),
        "U": U.to(torch.float32),
        "eigvals": eigvals.to(torch.float32),
        "meta": {
            "space": args.space,
            "dataset_path": str(args.dataset),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "action_dim": action_dim,
            "robot": args.robot,
            "control_action_scale": control_scale,
            "num_samples": n,
            "effective_rank": effective_rank,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"\nsaved basis to {args.out}")

    if args.plot is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.semilogy(eigvals.numpy(), marker="o", ms=3)
        ax1.set_title(f"action covariance spectrum ({args.space})")
        ax1.set_xlabel("eigen index (desc)")
        ax1.set_ylabel("eigenvalue")
        ax2.plot(100 * cum.numpy(), marker="o", ms=3)
        ax2.axhline(90, color="gray", ls="--", lw=0.8)
        ax2.set_title(f"cumulative variance %  (eff. rank {effective_rank:.1f})")
        ax2.set_xlabel("eigen index (desc)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"saved plot to {args.plot}")


if __name__ == "__main__":
    main()
