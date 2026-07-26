"""Advantage-Weighted Conservative Q-Learning (AW-CQL v0).

Scalar CQL with an exogenous per-transition weight w multiplied into the
conservative penalty bracket only:

    conservative_loss = cql_alpha * ((w * pen1).mean() + (w * pen2).mean())

w is precomputed offline by ``scripts/aw_precompute_weights.py`` from truncated
H-step returns relative to a (motion_id, phase_bin) baseline, clipped and
globally mean-normalized to 1, and stored as a ``<h5>.aw_weights.npz`` sidecar
(the source H5 is never modified). Because w is a function of dataset indices
only — never of Q — it adds no contraction channel to the Bellman operator.

v0 invariants (violating any of these makes it a different method):
  1. TD path unweighted — bellman_loss untouched.
  2. Same w on both twin critics.
  3. No per-batch renormalization — global mean(w)=1 was fixed at precompute;
     renormalizing per batch would inject batch-composition noise.
  4. No source-differential weighting inside the logsumexp (rand/curr/next) —
     w multiplies the per-sample bracket scalar only. (By linearity,
     (w*(lse-qd)).mean() == (w*lse).mean() - (w*qd).mean(), so weighting the
     per-sample gap is exactly the bracket form.)
  5. alpha / Lagrange path untouched — global mean(w)=1 keeps the effective
     alpha aligned with the unweighted baseline, which is what makes this a
     one-variable comparison.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from holosoma.agents.cql.cql_agent import CQLAgent
from holosoma.config_types.algo import AWCQLConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.safe_torch_import import TensorDict, torch


class AWCQLAgent(CQLAgent):
    """Scalar CQL whose conservative bracket is scaled by precomputed advantage weights."""

    config: AWCQLConfig

    _AW_METRIC_KEYS = (
        "aw_cql/batch_ess",
        "aw_cql/batch_w_mean",
        "aw_cql/batch_w_max",
    )

    def __init__(
        self,
        env: BaseTask,
        config: AWCQLConfig,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict | None = None,
    ):
        self._aw_weight_table: torch.Tensor | None = None
        self._aw_batch_weight: torch.Tensor | None = None
        self._aw_last_dataset_index: torch.Tensor | None = None
        self._aw_last_metrics: dict[str, torch.Tensor] = {}
        super().__init__(env, config, device, log_dir, multi_gpu_cfg)

    def setup(self) -> None:
        super().setup()
        sidecar_path = self.config.aw_weights_path or f"{self._offline_dataset_path}.aw_weights.npz"
        logger.info(
            f"AW-CQL pairing paths: h5='{self._offline_dataset_path}', sidecar='{sidecar_path}'"
        )
        aw = np.load(sidecar_path)
        sidecar_n = int(aw["n"])
        if sidecar_n != self._offline_num_samples:
            raise ValueError(
                f"AW sidecar / H5 size mismatch: sidecar '{sidecar_path}' has n={sidecar_n}, "
                f"dataset '{self._offline_dataset_path}' has {self._offline_num_samples} transitions. "
                f"Re-run scripts/aw_precompute_weights.py against this exact H5."
            )
        weight = np.asarray(aw["weight"], dtype=np.float32)
        self._aw_weight_table = torch.as_tensor(weight, dtype=torch.float32, device=self.device)
        self._aw_last_metrics = self._zero_aw_metrics()
        self._install_aw_index_capture()
        logger.info(
            "AW-CQL weights loaded from "
            f"'{sidecar_path}': n={sidecar_n}, beta={float(aw['beta']):.6f}, "
            f"ESS/N={float(aw['ess_frac']):.3f}, clip%={100.0 * float(aw['clip_frac']):.2f}, "
            f"mean(w)={float(weight.mean()):.6f}"
        )

    def _zero_aw_metrics(self) -> dict[str, torch.Tensor]:
        return {key: torch.zeros((), device=self.device) for key in self._AW_METRIC_KEYS}

    def _install_aw_index_capture(self) -> None:
        """Capture each sampled batch's global H5 row indices ('dataset_index').

        Both offline samplers attach 'dataset_index' to raw batches, but the base
        agent drops it while assembling the training TensorDict, so we record it
        here instead of duplicating the assembly code.
        """
        sampler = self._offline_gpu_cache if self._offline_gpu_cache is not None else self._offline_shuffle_buffer
        if sampler is None:
            raise RuntimeError("AW-CQL requires an offline sampler; call setup() with an offline dataset configured.")
        original_sample = sampler.sample

        def sample_with_index(batch_size: int):
            batch = original_sample(batch_size=batch_size)
            if "dataset_index" not in batch:
                raise RuntimeError("Offline sampler batch has no 'dataset_index'; AW-CQL cannot map weights.")
            self._aw_last_dataset_index = batch["dataset_index"]
            return batch

        sampler.sample = sample_with_index  # type: ignore[method-assign]

    def _sample_offline_batch(
        self,
        batch_size: int,
        normalize_obs,
        normalize_critic_obs,
    ) -> TensorDict:
        data = super()._sample_offline_batch(batch_size, normalize_obs, normalize_critic_obs)
        assert self._aw_weight_table is not None and self._aw_last_dataset_index is not None
        indices = self._aw_last_dataset_index.to(device=self.device, dtype=torch.long)
        weight = self._aw_weight_table[indices]
        effective_batch_size = int(data.batch_size[0])
        if effective_batch_size != weight.shape[0]:
            # Symmetry augmentation repeats the sampled transitions num_aug times.
            if effective_batch_size % weight.shape[0] != 0:
                raise RuntimeError(
                    f"Effective batch size {effective_batch_size} is not a multiple of the sampled "
                    f"batch size {weight.shape[0]}; cannot align AW weights."
                )
            weight = weight.repeat(effective_batch_size // weight.shape[0])
        self._aw_batch_weight = weight
        data["aw_weight"] = weight
        # Batch ESS is logged from here (eager path) rather than inside the
        # torch.compile'd critic step.
        with torch.no_grad():
            batch_ess = weight.sum().square() / (weight.square().sum() * weight.numel())
            self._aw_last_metrics = {
                "aw_cql/batch_ess": batch_ess,
                "aw_cql/batch_w_mean": weight.mean(),
                "aw_cql/batch_w_max": weight.max(),
            }
        return data

    def _transform_cql_per_sample_losses(
        self,
        q1_gap: torch.Tensor,
        q2_gap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale the per-sample conservative bracket by the exogenous AW weight.

        (w * gap).mean() downstream is exactly cql_alpha * (w * pen).mean(); the
        TD, alpha, and Lagrange paths are untouched (invariants 1 and 5).
        """
        assert self._aw_batch_weight is not None
        weight = self._aw_batch_weight.to(dtype=q1_gap.dtype)
        return weight * q1_gap, weight * q2_gap

    @torch.no_grad()
    def _compute_action_ood_stats(self, data: TensorDict) -> dict[str, torch.Tensor]:
        stats = super()._compute_action_ood_stats(data)
        stats.update(self._aw_last_metrics)
        return stats
