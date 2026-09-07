"""Asymmetric advantage-weighted CQL.

For each dataset transition this variant uses

    w_minus * logsumexp(Q) - w_plus * Q(s, a_data)

where both fixed weights are independently normalized to dataset mean one.
The positive weight is byte-for-byte the AW-CQL precompute formula.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from loguru import logger

from holosoma.agents.aw_cql.aw_cql_agent import AWCQLAgent
from holosoma.config_types.algo import AsymCQLConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.safe_torch_import import TensorDict, torch


class AsymCQLAgent(AWCQLAgent):
    config: AsymCQLConfig

    def __init__(self, env: BaseTask, config: AsymCQLConfig, device: str, log_dir: str,
                 multi_gpu_cfg: dict | None = None):
        self._asym_minus_weight_table: torch.Tensor | None = None
        self._asym_batch_minus_weight: torch.Tensor | None = None
        self._asym_phase_bin_table: torch.Tensor | None = None
        self._asym_dataset_metrics: dict[str, torch.Tensor] = {}
        self._asym_last_metrics: dict[str, torch.Tensor] = {}
        super().__init__(env, config, device, log_dir, multi_gpu_cfg)

    def setup(self) -> None:
        # Let the proven AW loader/index capture load w+ from the standard
        # ``weight`` key, while directing it at the dual-weight sidecar.
        sidecar_path = self.config.asym_weights_path or f"{self.config.offline_dataset_path}.asym_weights.npz"
        original_config = self.config
        self.config = replace(self.config, aw_weights_path=sidecar_path)
        try:
            super().setup()
        finally:
            self.config = original_config
        with np.load(sidecar_path, allow_pickle=False) as sidecar:
            plus = np.asarray(sidecar["weight"], dtype=np.float32)
            minus = np.asarray(sidecar["weight_minus"], dtype=np.float32)
            phase_bin = np.asarray(sidecar["phase_bin"], dtype=np.int64)
            if minus.shape != (self._offline_num_samples,):
                raise ValueError(f"Asym-CQL weight_minus has shape {minus.shape}; expected ({self._offline_num_samples},)")
            self._asym_minus_weight_table = torch.as_tensor(minus, device=self.device)
            self._asym_phase_bin_table = torch.as_tensor(phase_bin, device=self.device)
            delta = minus.astype(np.float64) - plus.astype(np.float64)
            self._asym_dataset_metrics = {
                "asym_cql/dataset_d_abs": torch.as_tensor(np.abs(delta).mean(), device=self.device),
                "asym_cql/dataset_d_rms": torch.as_tensor(np.sqrt(np.square(delta).mean()), device=self.device),
            }
            logger.info(
                f"Asym-CQL dual weights loaded: mean(w+)={float(self._aw_weight_table.mean()):.6f}, "
                f"mean(w-)={float(minus.mean()):.6f}, mean(w--w+)={float(minus.mean() - self._aw_weight_table.mean()):.6f}, "
                f"D_abs={float(np.abs(delta).mean()):.6f}, D_rms={float(np.sqrt(np.square(delta).mean())):.6f}"
            )

    def _sample_offline_batch(self, batch_size: int, normalize_obs, normalize_critic_obs) -> TensorDict:
        data = super()._sample_offline_batch(batch_size, normalize_obs, normalize_critic_obs)
        assert self._asym_minus_weight_table is not None and self._aw_last_dataset_index is not None
        indices = self._aw_last_dataset_index.to(device=self.device, dtype=torch.long)
        minus = self._asym_minus_weight_table[indices]
        effective_batch_size = int(data.batch_size[0])
        if effective_batch_size != minus.shape[0]:
            if effective_batch_size % minus.shape[0] != 0:
                raise RuntimeError("Cannot align Asym-CQL weights with the augmented batch")
            minus = minus.repeat(effective_batch_size // minus.shape[0])
        self._asym_batch_minus_weight = minus
        plus = data["aw_weight"]
        delta = minus - plus
        self._asym_last_metrics = {
            **self._asym_dataset_metrics,
            "asym_cql/batch_w_plus_mean": plus.mean(),
            "asym_cql/batch_w_minus_mean": minus.mean(),
            "asym_cql/batch_w_minus_minus_plus_mean": delta.mean(),
            "asym_cql/batch_d_abs": delta.abs().mean(),
            "asym_cql/batch_d_rms": delta.square().mean().sqrt(),
        }
        return data

    def _build_cql_per_sample_losses(self, q1_lse: torch.Tensor, q2_lse: torch.Tensor,
                                     q1_data: torch.Tensor, q2_data: torch.Tensor):
        assert self._aw_batch_weight is not None and self._asym_batch_minus_weight is not None
        plus = self._aw_batch_weight.to(dtype=q1_lse.dtype)
        minus = self._asym_batch_minus_weight.to(dtype=q1_lse.dtype)
        # Keep these components observable separately; their difference is the
        # exact per-critic conservative bracket returned below.
        self._asym_last_metrics.update({
            "asym_cql/lse_weighted_term": (0.5 * minus * (q1_lse + q2_lse)).mean().detach(),
            "asym_cql/anchor_weighted_term": (0.5 * plus * (q1_data + q2_data)).mean().detach(),
        })
        return minus * q1_lse - plus * q1_data, minus * q2_lse - plus * q2_data

    def _transform_cql_per_sample_losses(self, q1_gap: torch.Tensor, q2_gap: torch.Tensor):
        # AWCQLAgent's transform would multiply the completed bracket by w+ a
        # second time.  Asym-CQL places each weight explicitly above.
        return q1_gap, q2_gap

    @torch.no_grad()
    def _compute_action_ood_stats(self, data: TensorDict) -> dict[str, torch.Tensor]:
        stats = super()._compute_action_ood_stats(data)
        stats.update(self._asym_last_metrics)
        if self._asym_phase_bin_table is not None and self._aw_last_dataset_index is not None:
            bins = self._asym_phase_bin_table[self._aw_last_dataset_index.to(self.device, torch.long)]
            plus = self._aw_batch_weight[: bins.numel()]
            minus = self._asym_batch_minus_weight[: bins.numel()]
            for phase_bin in torch.unique(bins).tolist():
                mask = bins == phase_bin
                prefix = f"asym_cql/phase_bin_{phase_bin}"
                delta = minus[mask] - plus[mask]
                stats[f"{prefix}/w_plus_mean"] = plus[mask].mean()
                stats[f"{prefix}/w_minus_mean"] = minus[mask].mean()
                stats[f"{prefix}/w_minus_minus_plus_mean"] = delta.mean()
                stats[f"{prefix}/m_kappa"] = delta.mean()
                stats[f"{prefix}/d_abs"] = delta.abs().mean()
                stats[f"{prefix}/d_rms"] = delta.square().mean().sqrt()
        return stats
