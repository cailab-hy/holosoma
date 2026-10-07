"""ODPR-CQL: scalar CQL with ODPR-A (OPER-A) priority weights on the conservative bracket.

Baseline for AW-CQL that keeps the weight *placement* and swaps the weight *signal*:

    conservative_loss = cql_alpha * ((w * pen1).mean() + (w * pen2).mean()),   pen = LSE - Q_D

where ``w`` is the per-transition priority weight of ODPR ("Decoupled Prioritized Resampling",
Yue et al.): a TD(0) advantage from an iteratively re-fitted behaviour value function, seed-averaged
and passed through ODPR's load-time normalisation (linear rescale of the weight std, floor eps),
mean one over the dataset. The weights are precomputed and stored in a sidecar:

    python scripts/oper_precompute_weights.py <h5> --seed {1,2,3} --out <h5>.oper_a.seed{N}.npz
    python scripts/odpr_make_sidecar.py --oper <h5>.oper_a.seed*.npz --out <h5>.oper_a.odpr.npz

Invariants (same as AW-CQL so the two differ only in the weight table): the TD path, actor update,
alpha/Lagrange path and action sampling are plain CQL; both twin critics use the same w; there is no
per-batch renormalisation (global mean(w) = 1 is fixed at precompute time).

This agent is self-contained (it derives from CQLAgent, not from AWCQLAgent).
"""

from __future__ import annotations

import hashlib

import h5py
import numpy as np
from loguru import logger

from holosoma.agents.cql.cql_agent import CQLAgent
from holosoma.config_types.algo import ODPRCQLConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.safe_torch_import import TensorDict, torch


class ODPRCQLAgent(CQLAgent):
    """Scalar CQL whose conservative bracket is scaled by precomputed ODPR-A priority weights."""

    config: ODPRCQLConfig

    _METRIC_KEYS = ("odpr_cql/batch_ess", "odpr_cql/batch_w_mean", "odpr_cql/batch_w_max")

    def __init__(
        self,
        env: BaseTask,
        config: ODPRCQLConfig,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict | None = None,
    ):
        self._odpr_weight_table: torch.Tensor | None = None
        self._odpr_batch_weight: torch.Tensor | None = None
        self._odpr_last_dataset_index: torch.Tensor | None = None
        self._odpr_last_metrics: dict[str, torch.Tensor] = {}
        super().__init__(env, config, device, log_dir, multi_gpu_cfg)

    def setup(self) -> None:
        super().setup()
        sidecar_path = self.config.odpr_weights_path or f"{self._offline_dataset_path}.oper_a.odpr.npz"
        logger.info(f"ODPR-CQL pairing paths: h5='{self._offline_dataset_path}', sidecar='{sidecar_path}'")
        with np.load(sidecar_path, allow_pickle=False) as sidecar:
            weight = self._validated_weights(sidecar, sidecar_path)
            mode = str(np.asarray(sidecar["mode"]).item()) if "mode" in sidecar else "unknown"
            odpr_std = float(sidecar["odpr_std"]) if "odpr_std" in sidecar else float("nan")
            odpr_eps = float(sidecar["odpr_eps"]) if "odpr_eps" in sidecar else float("nan")
            num_seeds = int(sidecar["num_seeds"]) if "num_seeds" in sidecar else 1
        self._odpr_weight_table = torch.as_tensor(weight, dtype=torch.float32, device=self.device)
        self._odpr_last_metrics = {key: torch.zeros((), device=self.device) for key in self._METRIC_KEYS}
        self._install_index_capture()
        ess = float(weight.sum() ** 2 / (weight.size * np.square(weight, dtype=np.float64).sum()))
        logger.info(
            f"ODPR-CQL weights loaded from '{sidecar_path}': n={weight.size}, mode={mode}, seeds={num_seeds}, "
            f"std_param={odpr_std:.3f}, eps={odpr_eps:.3f}, ESS/N={ess:.3f}, "
            f"mean(w)={float(weight.mean()):.6f}, max(w)={float(weight.max()):.2f}"
        )

    def _validated_weights(self, sidecar, sidecar_path: str) -> np.ndarray:
        missing = [key for key in ("weight", "n", "rhash") if key not in sidecar]
        if missing:
            raise ValueError(f"ODPR sidecar '{sidecar_path}' is missing keys {missing}")
        weight = np.asarray(sidecar["weight"], dtype=np.float32)
        sidecar_n = int(sidecar["n"])
        if sidecar_n != self._offline_num_samples or weight.shape != (self._offline_num_samples,):
            raise ValueError(
                f"ODPR sidecar / H5 size mismatch: sidecar '{sidecar_path}' has n={sidecar_n}, "
                f"dataset '{self._offline_dataset_path}' has {self._offline_num_samples} transitions."
            )
        stored_hash = str(np.asarray(sidecar["rhash"]).item())
        actual_hash = self._h5_reward_fingerprint(self._offline_dataset_path)
        if stored_hash != actual_hash:
            raise ValueError(
                f"ODPR sidecar '{sidecar_path}' was computed for another dataset "
                f"(rhash {stored_hash} != {actual_hash}). Re-run scripts/odpr_make_sidecar.py."
            )
        if not np.isfinite(weight).all() or weight.min() < 0.0 or abs(float(weight.mean()) - 1.0) > 1e-3:
            raise ValueError(f"ODPR sidecar '{sidecar_path}' weights must be finite, non-negative and mean one")
        return weight

    @staticmethod
    def _h5_reward_fingerprint(h5_path) -> str:
        """Same reward fingerprint as scripts/oper_precompute_weights.py / aw_precompute_weights.py."""
        with h5py.File(h5_path, "r") as h5_file:
            rewards = h5_file["rewards" if "rewards" in h5_file else "reward"]
            num_rows = int(h5_file.attrs.get("num_samples", rewards.shape[0]))
            first = np.asarray(rewards[: min(1000, num_rows)], dtype=np.float64).reshape(-1)
            last = np.asarray(rewards[max(0, num_rows - 1000) : num_rows], dtype=np.float64).reshape(-1)
        return hashlib.sha256(
            np.ascontiguousarray(first).tobytes() + np.ascontiguousarray(last).tobytes()
        ).hexdigest()[:16]

    def _install_index_capture(self) -> None:
        """Record each sampled batch's global H5 row indices so weights can be looked up."""
        sampler = self._offline_gpu_cache if self._offline_gpu_cache is not None else self._offline_shuffle_buffer
        if sampler is None:
            raise RuntimeError("ODPR-CQL requires an offline sampler; call setup() with an offline dataset.")
        original_sample = sampler.sample

        def sample_with_index(batch_size: int):
            batch = original_sample(batch_size=batch_size)
            if "dataset_index" not in batch:
                raise RuntimeError("Offline sampler batch has no 'dataset_index'; ODPR-CQL cannot map weights.")
            self._odpr_last_dataset_index = batch["dataset_index"]
            return batch

        sampler.sample = sample_with_index  # type: ignore[method-assign]

    def _sample_offline_batch(self, batch_size: int, normalize_obs, normalize_critic_obs) -> TensorDict:
        data = super()._sample_offline_batch(batch_size, normalize_obs, normalize_critic_obs)
        assert self._odpr_weight_table is not None and self._odpr_last_dataset_index is not None
        indices = self._odpr_last_dataset_index.to(device=self.device, dtype=torch.long)
        weight = self._odpr_weight_table[indices]
        effective_batch_size = int(data.batch_size[0])
        if effective_batch_size != weight.shape[0]:
            # Symmetry augmentation repeats the sampled transitions num_aug times.
            if effective_batch_size % weight.shape[0] != 0:
                raise RuntimeError(
                    f"Effective batch size {effective_batch_size} is not a multiple of the sampled "
                    f"batch size {weight.shape[0]}; cannot align ODPR weights."
                )
            weight = weight.repeat(effective_batch_size // weight.shape[0])
        self._odpr_batch_weight = weight
        data["odpr_weight"] = weight
        with torch.no_grad():
            self._odpr_last_metrics = {
                "odpr_cql/batch_ess": weight.sum().square() / (weight.square().sum() * weight.numel()),
                "odpr_cql/batch_w_mean": weight.mean(),
                "odpr_cql/batch_w_max": weight.max(),
            }
        return data

    def _transform_cql_per_sample_losses(
        self,
        q1_gap: torch.Tensor,
        q2_gap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale the per-sample conservative bracket (LSE - Q_D) by the ODPR priority weight."""
        assert self._odpr_batch_weight is not None
        weight = self._odpr_batch_weight.to(dtype=q1_gap.dtype)
        return weight * q1_gap, weight * q2_gap

    @torch.no_grad()
    def _compute_action_ood_stats(self, data: TensorDict) -> dict[str, torch.Tensor]:
        stats = super()._compute_action_ood_stats(data)
        stats.update(self._odpr_last_metrics)
        return stats
