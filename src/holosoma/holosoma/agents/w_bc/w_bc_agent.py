"""Advantage-weighted behavior cloning with AW-CQL's fixed sidecar weights.

W-BC changes only the per-transition contribution to likelihood BC:

    L_wBC = mean_i[w_i * (-log pi(a_i | s_i))]

The sidecar is generated once by ``scripts/aw_precompute_weights.py`` and is
loaded without recomputation or per-batch normalization. Its transition order
must exactly match the HDF5 dataset sampled by the base BC agent.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from loguru import logger

from holosoma.agents.bc.bc_agent import BCAgent
from holosoma.config_types.algo import WBCConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.safe_torch_import import TensorDict, torch


def weighted_bc_nll(
    log_prob_data: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return mean_i[w_i * (-log pi(a_i | s_i))]."""
    if log_prob_data.ndim != 1:
        raise ValueError(f"Expected one summed log-probability per transition, got {tuple(log_prob_data.shape)}.")
    if weights.ndim != 1 or weights.shape[0] != log_prob_data.shape[0]:
        raise ValueError(
            f"Expected one weight per transition, got weights={tuple(weights.shape)} "
            f"for log_prob_data={tuple(log_prob_data.shape)}."
        )
    per_transition_nll = -log_prob_data
    return (weights.to(dtype=per_transition_nll.dtype) * per_transition_nll).mean()


class WBCAgent(BCAgent):
    """Likelihood behavior cloning weighted by AW-CQL's exogenous transition weights."""

    config: WBCConfig

    def __init__(
        self,
        env: BaseTask,
        config: WBCConfig,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict | None = None,
    ):
        self._aw_weight_table: torch.Tensor | None = None
        super().__init__(env, config, device, log_dir, multi_gpu_cfg)

    def setup(self) -> None:
        super().setup()
        dataset_size = self._read_dataset_size(self._offline_dataset_path)
        sidecar_path = Path(self.config.aw_weights_path or f"{self._offline_dataset_path}.aw_weights.npz")
        with np.load(sidecar_path, allow_pickle=False) as sidecar:
            if "n" not in sidecar or "weight" not in sidecar:
                raise KeyError(f"W-BC sidecar '{sidecar_path}' must contain 'n' and 'weight'.")
            sidecar_size = int(sidecar["n"])
            weights = np.asarray(sidecar["weight"], dtype=np.float32)
            beta = float(sidecar["beta"]) if "beta" in sidecar else float("nan")
            ess_fraction = float(sidecar["ess_frac"]) if "ess_frac" in sidecar else float("nan")
            clip_fraction = float(sidecar["clip_frac"]) if "clip_frac" in sidecar else float("nan")

        if sidecar_size != dataset_size:
            raise ValueError(
                f"AW sidecar / H5 size mismatch: sidecar '{sidecar_path}' has n={sidecar_size}, "
                f"dataset '{self._offline_dataset_path}' has {dataset_size} transitions. "
                "Re-run scripts/aw_precompute_weights.py against this exact H5."
            )
        if weights.shape != (dataset_size,):
            raise ValueError(
                f"W-BC sidecar weight shape must be ({dataset_size},), got {weights.shape}."
            )
        if not np.isfinite(weights).all() or bool((weights < 0.0).any()):
            raise ValueError("W-BC sidecar weights must be finite and non-negative.")

        self._aw_weight_table = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        logger.info(
            "W-BC weights loaded from "
            f"'{sidecar_path}': n={sidecar_size}, beta={beta:.6f}, "
            f"ESS/N={ess_fraction:.3f}, clip%={100.0 * clip_fraction:.2f}, "
            f"mean(w)={float(weights.mean()):.6f}; no batch renormalization"
        )

    @staticmethod
    def _read_dataset_size(dataset_path: Path) -> int:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Offline dataset not found at '{dataset_path}'.")
        with h5py.File(dataset_path, "r") as h5_file:
            if "observations" not in h5_file:
                raise KeyError("Offline dataset is missing required key 'observations'.")
            return int(h5_file.attrs.get("num_samples", h5_file["observations"].shape[0]))

    def offline_dataset_random_sampling(
        self,
        batch_size: int,
        num_updates: int,
        normalize_obs,
    ) -> list[TensorDict]:
        batches = super().offline_dataset_random_sampling(batch_size, num_updates, normalize_obs)
        if self._aw_weight_table is None:
            raise RuntimeError("W-BC weight table is not initialized; call setup() first.")
        for data in batches:
            indices = data["dataset_index"].to(device=self.device, dtype=torch.long)
            data["aw_weight"] = self._aw_weight_table.index_select(0, indices)
        return batches

    def _compute_actor_loss(
        self,
        data: TensorDict,
        policy_actions_u: torch.Tensor,
        dataset_actions_u: torch.Tensor,
        log_prob_data: torch.Tensor,
    ) -> torch.Tensor:
        del policy_actions_u, dataset_actions_u
        return weighted_bc_nll(log_prob_data, data["aw_weight"])

    def _extra_batch_training_metrics(self, data: TensorDict) -> dict[str, torch.Tensor]:
        weights = data["aw_weight"].detach().float()
        normalized_ess = weights.sum().square() / (weights.square().sum().clamp_min(1e-12) * weights.numel())
        return {
            "w_bc/batch_ess": normalized_ess,
            "w_bc/batch_w_mean": weights.mean(),
            "w_bc/batch_w_std": weights.std(unbiased=False),
            "w_bc/batch_w_max": weights.max(),
            "w_bc/batch_w_p99": torch.quantile(weights, 0.99),
        }
